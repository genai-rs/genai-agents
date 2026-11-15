//! The `#[tool]` attribute of [genai-agents](https://docs.rs/genai-agents): a
//! `FunctionTool` from an async function.
//!
//! Use it through genai-agents, with its `macros` feature: `use genai_agents::tool;`. The
//! generated code refers to `::genai_agents`, so this crate does not work on its own.

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::ext::IdentExt;
use syn::spanned::Spanned;
use syn::{
    Attribute, Expr, ExprLit, FnArg, GenericArgument, Ident, ItemFn, Lit, LitStr, Meta, Pat,
    PathArguments, ReturnType, Type,
};

/// A [`FunctionTool`] from an async function: `#[tool]` keeps the function and adds a unit
/// struct named after it in UpperCamelCase (`get_weather` gives `GetWeather`), to register in a
/// `ToolRegistry`.
///
/// - The name of the tool is the name of the function.
/// - Its description is the function's doc comment.
/// - Its input is a struct named after the tool (`GetWeatherInput`), with a field for each
///   parameter. The schema is derived from it with
///   [`schema::for_type`](https://docs.rs/genai-agents/latest/genai_agents/tools/schema/fn.for_type.html):
///   the doc comment of a parameter is the description of its property, and an `Option`
///   parameter may be left out. Fields the schema does not list are refused
///   (`additionalProperties: false`). Other attributes of a parameter go to its field, such as
///   `#[serde(default)]`.
/// - Its output is the `T` of the function's `Result<T>` (genai-agents' `Result`).
///
/// The parameters' types implement `Deserialize` and `JsonSchema`, the output's `Serialize`; the
/// generated code itself needs no dependency on serde or schemars.
///
/// Options, as in `#[tool(name = "weather", parallel)]`:
///
/// - `name = "..."`: the name of the tool, instead of the function's;
/// - `description = "..."`: the description, instead of the doc comment;
/// - `parallel`: the tool may run at the same time as others (`supports_parallel`);
/// - `untrusted`: its output holds content from outside the application (`untrusted_output`).
///
/// Only free async functions with plain parameters (`name: Type`, owned types) are taken: not
/// methods, generic functions or functions that are not `async`. A tool with state implements
/// `FunctionTool` itself.
///
/// ```
/// use genai_agents::tools::ToolRegistry;
/// use genai_agents::{tool, Result, ToolCall};
///
/// /// Current weather of a city.
/// #[tool(parallel)]
/// async fn get_weather(
///     /// The city, such as "Ghent".
///     city: String,
///     /// Days ahead, 0 for today.
///     days: Option<u32>,
/// ) -> Result<String> {
///     Ok(format!("light rain in {city}, {} days ahead", days.unwrap_or(0)))
/// }
///
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() -> Result<()> {
/// let mut tools = ToolRegistry::new();
/// tools.register(GetWeather)?;
/// let answer = tools
///     .answer_call(ToolCall::function("call_1", "get_weather", r#"{"city":"Ghent"}"#))
///     .await?;
/// assert_eq!(answer.body.as_text(), Some(r#""light rain in Ghent, 0 days ahead""#));
/// # Ok(())
/// # }
/// ```
///
/// [`FunctionTool`]: https://docs.rs/genai-agents/latest/genai_agents/tools/trait.FunctionTool.html
#[proc_macro_attribute]
pub fn tool(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    expand(attr.into(), item.into())
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// The options of `#[tool(...)]`.
#[derive(Default)]
struct Options {
    name: Option<LitStr>,
    description: Option<LitStr>,
    parallel: bool,
    untrusted: bool,
}

impl Options {
    fn parse(attr: TokenStream) -> syn::Result<Self> {
        let mut options = Self::default();
        let parser = syn::meta::parser(|meta| {
            if meta.path.is_ident("name") {
                options.name = Some(meta.value()?.parse()?);
            } else if meta.path.is_ident("description") {
                options.description = Some(meta.value()?.parse()?);
            } else if meta.path.is_ident("parallel") {
                options.parallel = true;
            } else if meta.path.is_ident("untrusted") {
                options.untrusted = true;
            } else {
                return Err(meta.error(
                    "unknown option of #[tool]: use `name`, `description`, `parallel` or `untrusted`",
                ));
            }
            Ok(())
        });
        syn::parse::Parser::parse2(parser, attr)?;
        Ok(options)
    }
}

/// A parameter of the function: its name, type and attributes.
struct Param {
    ident: Ident,
    ty: Type,
    attrs: Vec<Attribute>,
}

fn expand(attr: TokenStream, item: TokenStream) -> syn::Result<TokenStream> {
    let options = Options::parse(attr)?;
    let mut function: ItemFn = syn::parse2(item)?;
    let sig = &function.sig;

    if sig.asyncness.is_none() {
        return Err(syn::Error::new_spanned(
            sig.fn_token,
            "#[tool] takes an `async fn`",
        ));
    }
    if !sig.generics.params.is_empty() || sig.generics.where_clause.is_some() {
        return Err(syn::Error::new_spanned(
            &sig.generics,
            "#[tool] takes a function without generic parameters",
        ));
    }
    if let Some(variadic) = &sig.variadic {
        return Err(syn::Error::new_spanned(
            variadic,
            "#[tool] takes no variadic function",
        ));
    }

    let fn_ident = sig.ident.clone();
    let tool_name = match &options.name {
        Some(name) => name.value(),
        None => fn_ident.unraw().to_string(),
    };
    check_name(
        &tool_name,
        options.name.as_ref().map_or(fn_ident.span(), LitStr::span),
    )?;

    let description = match &options.description {
        Some(description) => description.value(),
        None => doc_text(&function.attrs)?.ok_or_else(|| {
            syn::Error::new_spanned(
                &sig.ident,
                "a #[tool] function needs a doc comment, its description, or `#[tool(description = \"...\")]`",
            )
        })?,
    };
    if description.trim().is_empty() {
        return Err(syn::Error::new(
            options
                .description
                .as_ref()
                .map_or(fn_ident.span(), LitStr::span),
            "the description of a #[tool] is blank",
        ));
    }

    let output = output_type(&sig.output)?.clone();
    let params = take_params(&mut function)?;

    let vis = &function.vis;
    let struct_ident = format_ident!(
        "{}",
        upper_camel_case(&fn_ident.unraw().to_string()),
        span = fn_ident.span()
    );
    let input_ident = format_ident!("{}Input", struct_ident);
    let cfgs: Vec<&Attribute> = function
        .attrs
        .iter()
        .filter(|attr| attr.path().is_ident("cfg"))
        .collect();

    let field_idents: Vec<&Ident> = params.iter().map(|param| &param.ident).collect();
    let fields = params.iter().map(|param| {
        let Param { ident, ty, attrs } = param;
        quote! { #(#attrs)* #vis #ident: #ty }
    });

    let struct_doc = format!("The `{tool_name}` tool, from `#[tool]`.");
    let input_doc = format!("The arguments of the `{tool_name}` tool.");
    let parallel = options.parallel.then(|| {
        quote! {
            fn supports_parallel(&self) -> bool {
                true
            }
        }
    });
    let untrusted = options.untrusted.then(|| {
        quote! {
            fn untrusted_output(&self) -> bool {
                true
            }
        }
    });

    Ok(quote! {
        #function

        #(#cfgs)*
        #[doc = #struct_doc]
        #[derive(Debug, Clone, Copy, Default)]
        #vis struct #struct_ident;

        #(#cfgs)*
        #[doc = #input_doc]
        #[derive(
            Debug,
            ::genai_agents::__private::serde::Deserialize,
            ::genai_agents::__private::schemars::JsonSchema,
        )]
        #[serde(crate = "::genai_agents::__private::serde", deny_unknown_fields)]
        #[schemars(crate = "::genai_agents::__private::schemars")]
        #[allow(missing_docs)]
        #vis struct #input_ident {
            #(#fields,)*
        }

        #(#cfgs)*
        impl ::genai_agents::tools::FunctionTool for #struct_ident {
            type Input = #input_ident;
            type Output = #output;

            fn name(&self) -> &str {
                #tool_name
            }

            fn description(&self) -> &str {
                #description
            }

            fn parameters_schema(&self) -> ::genai_agents::__private::serde_json::Value {
                ::genai_agents::tools::schema::for_type::<#input_ident>()
            }

            async fn execute(&self, input: #input_ident) -> ::genai_agents::Result<#output> {
                let #input_ident { #(#field_idents),* } = input;
                #fn_ident(#(#field_idents),*).await
            }

            #parallel
            #untrusted
        }
    })
}

/// The parameters of the function, with their attributes taken off: Rust takes no doc comments
/// on parameters.
fn take_params(function: &mut ItemFn) -> syn::Result<Vec<Param>> {
    let mut params = Vec::new();
    for input in &mut function.sig.inputs {
        let typed = match input {
            FnArg::Receiver(receiver) => {
                return Err(syn::Error::new_spanned(
                    receiver,
                    "#[tool] takes a free function, not a method: implement `FunctionTool` for a tool with state",
                ))
            }
            FnArg::Typed(typed) => typed,
        };
        let ident = match &*typed.pat {
            Pat::Ident(pat) if pat.by_ref.is_none() && pat.subpat.is_none() => pat.ident.clone(),
            other => {
                return Err(syn::Error::new_spanned(
                    other,
                    "#[tool] takes parameters of the form `name: Type`",
                ))
            }
        };
        if let Type::Reference(reference) = &*typed.ty {
            return Err(syn::Error::new_spanned(
                reference,
                "the parameters of a #[tool] are owned: `String` instead of `&str`",
            ));
        }
        if let Type::ImplTrait(impl_trait) = &*typed.ty {
            return Err(syn::Error::new_spanned(
                impl_trait,
                "the parameters of a #[tool] have a concrete type",
            ));
        }
        params.push(Param {
            ident,
            ty: (*typed.ty).clone(),
            attrs: std::mem::take(&mut typed.attrs),
        });
    }
    Ok(params)
}

/// The `T` of `Result<T>`.
fn output_type(output: &ReturnType) -> syn::Result<&Type> {
    const MESSAGE: &str =
        "a #[tool] function returns `genai_agents::Result<T>`, with `T` the tool's output";
    let ty = match output {
        ReturnType::Default => return Err(syn::Error::new(output.span(), MESSAGE)),
        ReturnType::Type(_, ty) => ty,
    };
    let Type::Path(path) = &**ty else {
        return Err(syn::Error::new_spanned(ty, MESSAGE));
    };
    let last = match path.path.segments.last() {
        Some(last) if path.qself.is_none() && last.ident == "Result" => last,
        _ => return Err(syn::Error::new_spanned(ty, MESSAGE)),
    };
    let PathArguments::AngleBracketed(arguments) = &last.arguments else {
        return Err(syn::Error::new_spanned(ty, MESSAGE));
    };
    match arguments.args.first() {
        Some(GenericArgument::Type(output)) if arguments.args.len() == 1 => Ok(output),
        _ => Err(syn::Error::new_spanned(ty, MESSAGE)),
    }
}

/// The text of the doc comment, `None` without one.
fn doc_text(attrs: &[Attribute]) -> syn::Result<Option<String>> {
    let mut lines = Vec::new();
    for attr in attrs {
        if !attr.path().is_ident("doc") {
            continue;
        }
        let Meta::NameValue(name_value) = &attr.meta else {
            continue;
        };
        match &name_value.value {
            Expr::Lit(ExprLit {
                lit: Lit::Str(text),
                ..
            }) => lines.extend(
                text.value()
                    .split('\n')
                    .map(|line| line.strip_prefix(' ').unwrap_or(line).trim_end().to_owned()),
            ),
            other => return Err(syn::Error::new_spanned(
                other,
                "#[tool] reads only literal doc comments: give `#[tool(description = \"...\")]`",
            )),
        }
    }
    let text = lines.join("\n").trim().to_owned();
    Ok((!text.is_empty()).then_some(text))
}

/// The registry's rule: 1 to 64 ASCII letters, digits, underscores or hyphens.
fn check_name(name: &str, span: Span) -> syn::Result<()> {
    let valid = (1..=64).contains(&name.len())
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-');
    if valid {
        Ok(())
    } else {
        Err(syn::Error::new(
            span,
            format!(
                "the name of a tool is 1 to 64 ASCII letters, digits, underscores or hyphens, not `{name}`"
            ),
        ))
    }
}

/// `get_weather` as `GetWeather`.
fn upper_camel_case(snake: &str) -> String {
    snake
        .split('_')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            chars.next().map_or_else(String::new, |first| {
                first.to_uppercase().chain(chars).collect()
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;

    fn error(attr: TokenStream, item: TokenStream) -> String {
        match expand(attr, item) {
            Ok(tokens) => panic!("expected an error, got {tokens}"),
            Err(error) => error.to_string(),
        }
    }

    #[test]
    fn upper_camel_case_of_snake_case() {
        assert_eq!(upper_camel_case("get_weather"), "GetWeather");
        assert_eq!(upper_camel_case("search"), "Search");
        assert_eq!(upper_camel_case("_private__name_"), "PrivateName");
        assert_eq!(upper_camel_case("already_Camel"), "AlreadyCamel");
    }

    #[test]
    fn expands_into_a_struct_an_input_and_an_impl() {
        let tokens = expand(
            quote!(parallel),
            quote! {
                /// Current weather.
                async fn get_weather(
                    /// The city.
                    city: String,
                ) -> Result<String> { Ok(city) }
            },
        )
        .unwrap();
        let file: syn::File = syn::parse2(tokens).unwrap();
        let names: Vec<String> = file
            .items
            .iter()
            .map(|item| match item {
                syn::Item::Fn(item) => format!("fn {}", item.sig.ident),
                syn::Item::Struct(item) => format!("struct {}", item.ident),
                syn::Item::Impl(_) => "impl".to_owned(),
                _ => "other".to_owned(),
            })
            .collect();
        assert_eq!(
            names,
            [
                "fn get_weather",
                "struct GetWeather",
                "struct GetWeatherInput",
                "impl"
            ]
        );
        // The doc comment of the parameter moved to the field.
        let syn::Item::Fn(function) = &file.items[0] else {
            panic!("not a function")
        };
        let FnArg::Typed(city) = &function.sig.inputs[0] else {
            panic!("not a typed parameter")
        };
        assert!(city.attrs.is_empty());
        let syn::Item::Struct(input) = &file.items[2] else {
            panic!("not a struct")
        };
        let field = input.fields.iter().next().unwrap();
        assert_eq!(
            doc_text(&field.attrs).unwrap().as_deref(),
            Some("The city.")
        );
        let implementation = &file.items[3];
        let rendered = quote!(#implementation).to_string();
        assert!(rendered.contains("supports_parallel"));
        assert!(!rendered.contains("untrusted_output"));
    }

    #[test]
    fn doc_comment_lines_are_joined() {
        let function: ItemFn = syn::parse2(quote! {
            /// First line.
            ///
            ///   Indented.
            async fn f() -> Result<()> { Ok(()) }
        })
        .unwrap();
        assert_eq!(
            doc_text(&function.attrs).unwrap().as_deref(),
            Some("First line.\n\n  Indented.")
        );
    }

    #[test]
    fn options_override_name_and_description() {
        let tokens = expand(
            quote!(name = "weather", description = "The weather.", untrusted),
            quote!(
                async fn get_weather() -> genai_agents::Result<u32> {
                    Ok(1)
                }
            ),
        )
        .unwrap()
        .to_string();
        assert!(tokens.contains("\"weather\""));
        assert!(tokens.contains("\"The weather.\""));
        assert!(tokens.contains("untrusted_output"));
        assert!(!tokens.contains("supports_parallel"));
    }

    #[test]
    fn rejects_what_it_cannot_turn_into_a_tool() {
        let cases = [
            (
                quote!(),
                quote!(
                    /// Doc.
                    fn f() -> Result<()> {
                        Ok(())
                    }
                ),
                "takes an `async fn`",
            ),
            (
                quote!(),
                quote!(
                    async fn f() -> Result<()> {
                        Ok(())
                    }
                ),
                "needs a doc comment",
            ),
            (
                quote!(description = " "),
                quote!(
                    async fn f() -> Result<()> {
                        Ok(())
                    }
                ),
                "is blank",
            ),
            (
                quote!(),
                quote!(
                    /// Doc.
                    async fn f() {}
                ),
                "returns `genai_agents::Result<T>`",
            ),
            (
                quote!(),
                quote!(
                    /// Doc.
                    async fn f() -> u32 {
                        1
                    }
                ),
                "returns `genai_agents::Result<T>`",
            ),
            (
                quote!(),
                quote!(
                    /// Doc.
                    async fn f() -> std::result::Result<u32, String> {
                        Ok(1)
                    }
                ),
                "returns `genai_agents::Result<T>`",
            ),
            (
                quote!(),
                quote!(
                    /// Doc.
                    async fn f<T>(x: T) -> Result<()> {
                        Ok(())
                    }
                ),
                "without generic parameters",
            ),
            (
                quote!(),
                quote!(
                    /// Doc.
                    async fn f(&self) -> Result<()> {
                        Ok(())
                    }
                ),
                "not a method",
            ),
            (
                quote!(),
                quote!(
                    /// Doc.
                    async fn f((a, b): (u32, u32)) -> Result<()> {
                        Ok(())
                    }
                ),
                "`name: Type`",
            ),
            (
                quote!(),
                quote!(
                    /// Doc.
                    async fn f(city: &str) -> Result<()> {
                        Ok(())
                    }
                ),
                "are owned",
            ),
            (
                quote!(name = "a b"),
                quote!(
                    /// Doc.
                    async fn f() -> Result<()> {
                        Ok(())
                    }
                ),
                "the name of a tool",
            ),
            (
                quote!(retries = 3),
                quote!(
                    /// Doc.
                    async fn f() -> Result<()> {
                        Ok(())
                    }
                ),
                "unknown option",
            ),
            (
                quote!(),
                quote!(
                    #[doc = include_str!("doc.md")]
                    async fn f() -> Result<()> {
                        Ok(())
                    }
                ),
                "only literal doc comments",
            ),
        ];
        for (attr, item, expected) in cases {
            let message = error(attr, item.clone());
            assert!(message.contains(expected), "{item}: {message}");
        }
    }
}
