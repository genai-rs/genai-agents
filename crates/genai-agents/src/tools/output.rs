//! What the model sees of a tool's output: budgets and truncation.

use std::borrow::Cow;

use crate::item::{Content, ToolOutput, ToolOutputBody};

/// How much of a tool's output the model sees.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OutputBudget {
    /// At most this many bytes of text (UTF-8), cut on character boundaries.
    Bytes(usize),
    /// About this many tokens, counted as 4 bytes each.
    Tokens(usize),
    /// No limit.
    Unlimited,
}

impl OutputBudget {
    /// The budget in bytes; `None` when unlimited.
    #[must_use]
    pub fn max_bytes(self) -> Option<usize> {
        match self {
            Self::Bytes(bytes) => Some(bytes),
            Self::Tokens(tokens) => Some(tokens.saturating_mul(BYTES_PER_TOKEN)),
            Self::Unlimited => None,
        }
    }

    /// A budget of the same unit for `bytes` bytes.
    fn with_bytes(self, bytes: usize) -> Self {
        match self {
            Self::Tokens(_) => Self::Tokens(bytes / BYTES_PER_TOKEN),
            _ => Self::Bytes(bytes),
        }
    }
}

const BYTES_PER_TOKEN: usize = 4;

/// The truncation of tool outputs before they are sent to the model.
///
/// The default budget is 10 000 bytes, with a header line; a tool can override the budget
/// ([`FunctionTool::output_budget`](super::FunctionTool::output_budget)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct TruncationConfig {
    /// The budget of tools that do not set one.
    pub default: OutputBudget,
    /// Whether a truncated output starts with a line that says how long it was.
    pub header: bool,
}

impl Default for TruncationConfig {
    fn default() -> Self {
        Self {
            default: OutputBudget::Bytes(10_000),
            header: true,
        }
    }
}

impl TruncationConfig {
    /// A configuration with this default budget and the header line.
    #[must_use]
    pub fn new(default: OutputBudget) -> Self {
        Self {
            default,
            header: true,
        }
    }

    /// The same configuration with or without the header line.
    #[must_use]
    pub fn with_header(mut self, header: bool) -> Self {
        self.header = header;
        self
    }

    /// The budget of a tool: its own, else the default.
    #[must_use]
    pub fn budget_for(&self, tool_budget: Option<OutputBudget>) -> OutputBudget {
        tool_budget.unwrap_or(self.default)
    }
}

impl ToolOutputBody {
    /// The body cut to `budget`, keeping its head and its tail (see [`truncate_middle`]).
    ///
    /// The text of all text parts counts against the budget, in order: the part that crosses it
    /// is cut in the middle and later text parts are left out (a part says how many). Images and
    /// files are always kept. With `header`, a first line says how long the text was.
    #[must_use]
    pub fn truncated(&self, budget: OutputBudget, header: bool) -> Self {
        let Some(max) = budget.max_bytes() else {
            return self.clone();
        };
        match self {
            Self::Text(text) if text.len() <= max => self.clone(),
            Self::Text(text) => {
                let mut out = if header {
                    truncation_header(text.len(), text.lines().count())
                } else {
                    String::new()
                };
                out.push_str(&truncate_middle(text, budget));
                Self::Text(out)
            }
            Self::Items(items) => {
                let total: usize = items
                    .iter()
                    .filter_map(Content::as_text)
                    .map(str::len)
                    .sum();
                if total <= max {
                    return self.clone();
                }
                let lines = items
                    .iter()
                    .filter_map(Content::as_text)
                    .map(|text| text.lines().count())
                    .sum();
                let mut out = Vec::with_capacity(items.len() + 2);
                if header {
                    out.push(Content::text(truncation_header(total, lines)));
                }
                let mut left = max;
                let mut omitted = 0_usize;
                for item in items {
                    match item.as_text() {
                        None => out.push(item.clone()),
                        Some(_) if left == 0 => omitted += 1,
                        Some(text) if text.len() <= left => {
                            left -= text.len();
                            out.push(item.clone());
                        }
                        Some(text) => {
                            let budget = budget.with_bytes(left);
                            out.push(Content::text(truncate_middle(text, budget).into_owned()));
                            left = 0;
                        }
                    }
                }
                if omitted > 0 {
                    out.push(Content::text(format!(
                        "…{omitted} more text items omitted…"
                    )));
                }
                Self::Items(out)
            }
        }
    }
}

impl ToolOutput {
    /// The same answer, its body cut to `budget` ([`ToolOutputBody::truncated`]).
    #[must_use]
    pub fn truncated(mut self, budget: OutputBudget, header: bool) -> Self {
        self.body = self.body.truncated(budget, header);
        self
    }
}

fn truncation_header(bytes: usize, lines: usize) -> String {
    format!(
        "[truncated: original {bytes} bytes/~{} tokens, {lines} lines]\n",
        bytes.div_ceil(BYTES_PER_TOKEN)
    )
}

/// Cut `text` to `budget`, keeping its first and its last half.
///
/// The middle is replaced with `…N chars truncated…` (`…N tokens truncated…` for a token
/// budget). Cuts fall on character boundaries; text within the budget is returned as is.
///
/// ```
/// use genai_agents::tools::{truncate_middle, OutputBudget};
///
/// let text = "a".repeat(10) + &"b".repeat(10);
/// assert_eq!(truncate_middle(&text, OutputBudget::Bytes(8)), "aaaa…12 chars truncated…bbbb");
/// assert_eq!(truncate_middle("short", OutputBudget::Bytes(8)), "short");
/// ```
#[must_use]
pub fn truncate_middle(text: &str, budget: OutputBudget) -> Cow<'_, str> {
    let Some(max) = budget.max_bytes() else {
        return Cow::Borrowed(text);
    };
    if text.len() <= max {
        return Cow::Borrowed(text);
    }
    let mut head_end = max / 2;
    while !text.is_char_boundary(head_end) {
        head_end -= 1;
    }
    let mut tail_start = text.len() - (max - max / 2);
    while !text.is_char_boundary(tail_start) {
        tail_start += 1;
    }
    let removed = &text[head_end..tail_start];
    let marker = match budget {
        OutputBudget::Tokens(_) => {
            format!(
                "…{} tokens truncated…",
                removed.len().div_ceil(BYTES_PER_TOKEN)
            )
        }
        _ => format!("…{} chars truncated…", removed.chars().count()),
    };
    Cow::Owned(format!(
        "{}{marker}{}",
        &text[..head_end],
        &text[tail_start..]
    ))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;

    #[test]
    fn truncate_middle_keeps_char_boundaries() {
        let text = "é".repeat(10); // 20 bytes
        let cut = truncate_middle(&text, OutputBudget::Bytes(7));
        assert!(cut.starts_with('é'));
        assert!(cut.ends_with('é'));
        assert!(cut.contains("chars truncated"));
        let long = "x".repeat(100);
        let cut = truncate_middle(&long, OutputBudget::Tokens(2));
        assert_eq!(cut, "xxxx…23 tokens truncated…xxxx");
        assert_eq!(truncate_middle("abc", OutputBudget::Unlimited), "abc");
    }

    #[test]
    fn truncated_text_has_a_header() {
        let body = ToolOutputBody::text("line\n".repeat(10));
        let cut = body.truncated(OutputBudget::Bytes(10), true);
        let text = cut.as_text().unwrap();
        assert!(text.starts_with("[truncated: original 50 bytes/~13 tokens, 10 lines]\n"));
        assert_eq!(body.truncated(OutputBudget::Bytes(100), true), body);
        let cut = body.truncated(OutputBudget::Bytes(10), false);
        assert!(cut.as_text().unwrap().starts_with("line\n"));
    }

    #[test]
    fn truncated_items_keep_images() {
        let body = ToolOutputBody::Items(vec![
            Content::text("a".repeat(6)),
            Content::image_url("https://example.com/cat.png"),
            Content::text("b".repeat(10)),
            Content::text("c"),
        ]);
        let ToolOutputBody::Items(items) = body.truncated(OutputBudget::Bytes(10), false) else {
            panic!("items expected");
        };
        assert_eq!(items[0], Content::text("aaaaaa"));
        assert!(items[1].is_image());
        assert_eq!(items[2], Content::text("bb…6 chars truncated…bb"));
        assert_eq!(items[3], Content::text("…1 more text items omitted…"));
    }
}
