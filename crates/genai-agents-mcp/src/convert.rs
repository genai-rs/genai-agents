//! MCP tool results as tool outputs.

use genai_agents::{Content, ToolOutputBody};
use rmcp::model::{CallToolResult, ContentBlock, ResourceContents};
use serde_json::json;

/// What the model sees of a result: its content, or, when it is an error (`isError`), the text
/// of the failure.
pub(crate) fn output(result: CallToolResult) -> Result<ToolOutputBody, String> {
    let structured = result.structured_content;
    let parts: Vec<Content> = result.content.into_iter().map(content).collect();
    if result.is_error == Some(true) {
        let text = text_of(&parts);
        return Err(if text.is_empty() {
            structured.map_or_else(|| "the tool failed".to_string(), |value| value.to_string())
        } else {
            text
        });
    }
    if parts.is_empty() {
        // A result with only structured content (servers should send its JSON as text too).
        return Ok(ToolOutputBody::Text(
            structured
                .map(|value| value.to_string())
                .unwrap_or_default(),
        ));
    }
    if parts.iter().all(|part| part.as_text().is_some()) {
        return Ok(ToolOutputBody::Text(text_of(&parts)));
    }
    Ok(ToolOutputBody::Items(parts))
}

/// The text parts, one per line, and a note for each other part.
fn text_of(parts: &[Content]) -> String {
    parts
        .iter()
        .map(|part| match part {
            Content::Text { text } => text.clone(),
            Content::Image { .. } => "[image]".to_string(),
            _ => "[file]".to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// One content block: text as text, an image as an image (a `data:` URL), an embedded resource
/// as its text or as a file (an image when its MIME type says so), a resource link as a JSON
/// description of the link, and audio, which no model takes in tool outputs, as a note.
fn content(block: ContentBlock) -> Content {
    match block {
        ContentBlock::Text(text) => Content::text(text.text),
        ContentBlock::Image(image) => {
            Content::image_url(format!("data:{};base64,{}", image.mime_type, image.data))
        }
        ContentBlock::Audio(audio) => Content::text(format!(
            "[audio ({}) left out: tool outputs take no audio]",
            audio.mime_type
        )),
        ContentBlock::Resource(resource) => match resource.resource {
            ResourceContents::TextResourceContents { text, .. } => Content::text(text),
            ResourceContents::BlobResourceContents {
                uri,
                mime_type,
                blob,
                ..
            } => match mime_type {
                Some(mime) if mime.starts_with("image/") => {
                    Content::image_url(format!("data:{mime};base64,{blob}"))
                }
                _ => Content::file_data(file_name(&uri), blob),
            },
            // New kinds of resource contents: their JSON.
            #[allow(unreachable_patterns)]
            other => Content::text(serde_json::to_string(&other).unwrap_or_default()),
        },
        ContentBlock::ResourceLink(link) => {
            let mut value = json!({"type": "resource_link", "uri": link.uri, "name": link.name});
            if let Some(description) = link.description {
                value["description"] = description.into();
            }
            if let Some(mime_type) = link.mime_type {
                value["mimeType"] = mime_type.into();
            }
            Content::text(value.to_string())
        }
        // New kinds of content: their JSON.
        #[allow(unreachable_patterns)]
        other => Content::text(serde_json::to_string(&other).unwrap_or_default()),
    }
}

/// The last segment of a resource URI, as a file name.
fn file_name(uri: &str) -> String {
    uri.rsplit(['/', ':'])
        .find(|segment| !segment.is_empty())
        .unwrap_or("resource")
        .to_string()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]

    use super::*;
    use rmcp::model::{EmbeddedResource, Resource};

    #[test]
    fn text_images_resources_and_errors() {
        let text = CallToolResult::success(vec![ContentBlock::text("a"), ContentBlock::text("b")]);
        assert_eq!(output(text).unwrap(), ToolOutputBody::Text("a\nb".into()));

        let image = CallToolResult::success(vec![
            ContentBlock::text("a chart"),
            ContentBlock::image("aGk=", "image/png"),
            ContentBlock::Resource(EmbeddedResource::new(
                ResourceContents::blob("JVBERg==", "file:///reports/q3.pdf")
                    .with_mime_type("application/pdf"),
            )),
            ContentBlock::ResourceLink(Resource::new("file:///notes.txt", "notes")),
        ]);
        let ToolOutputBody::Items(parts) = output(image).unwrap() else {
            panic!("expected content parts");
        };
        assert_eq!(parts[0], Content::text("a chart"));
        assert_eq!(parts[1], Content::image_url("data:image/png;base64,aGk="));
        assert_eq!(parts[2], Content::file_data("q3.pdf", "JVBERg=="));
        assert!(parts[3]
            .as_text()
            .unwrap()
            .contains(r#""uri":"file:///notes.txt""#));

        let failed = CallToolResult::error(vec![ContentBlock::text("no such city")]);
        assert_eq!(output(failed).unwrap_err(), "no such city");

        let structured = CallToolResult::structured(json!({"temperature": 12}));
        assert_eq!(
            output(structured).unwrap(),
            ToolOutputBody::Text(r#"{"temperature":12}"#.into())
        );
    }
}
