//! The assistant's tools. Every path stays inside the workspace the assistant was started on:
//! a path that leaves it (`..`, an absolute path, a symbolic link out) is refused.

use std::path::{Path, PathBuf};

use genai_agents::tools::{FunctionTool, ToolRegistry};
use genai_agents::{tool, Error, Result};
use serde::Deserialize;
use serde_json::{json, Value};

/// The workspace: the directory the file tools work in.
#[derive(Debug, Clone)]
pub struct Workspace {
    root: PathBuf,
}

impl Workspace {
    pub fn new(root: &Path) -> Result<Self> {
        let root = root.canonicalize().map_err(Error::application)?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// `relative` inside the workspace, resolved; an error when it leaves it.
    fn resolve(&self, relative: &str) -> Result<PathBuf> {
        let relative = relative.trim_start_matches("./");
        if Path::new(relative).is_absolute() || relative.split(['/', '\\']).any(|part| part == "..")
        {
            return Err(refuse(format!(
                "{relative} is not a path inside the workspace"
            )));
        }
        let path = self.root.join(relative);
        let resolved = path
            .canonicalize()
            .map_err(|error| refuse(format!("{relative}: {error}")))?;
        if resolved.starts_with(&self.root) {
            Ok(resolved)
        } else {
            Err(refuse(format!("{relative} leads outside the workspace")))
        }
    }

    /// A path as the model sees it: relative, with `/` on every platform.
    fn relative(&self, path: &Path) -> String {
        match path.strip_prefix(&self.root) {
            Ok(relative) => relative
                .components()
                .map(|part| part.as_os_str().to_string_lossy())
                .collect::<Vec<_>>()
                .join("/"),
            Err(_) => path.display().to_string(),
        }
    }

    /// Every file of the workspace, relative, sorted; hidden ones left out.
    fn files(&self) -> Vec<PathBuf> {
        fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                let hidden = entry.file_name().to_string_lossy().starts_with('.');
                if hidden {
                    continue;
                }
                if path.is_dir() {
                    walk(&path, out);
                } else {
                    out.push(path);
                }
            }
        }
        let mut files = Vec::new();
        walk(&self.root, &mut files);
        files.sort();
        files
    }
}

fn refuse(message: String) -> Error {
    Error::application(std::io::Error::other(message))
}

/// The workspace's file tools, and the clock.
pub fn registry(workspace: &Workspace) -> Result<ToolRegistry> {
    let mut tools = ToolRegistry::new();
    tools.register(ListFiles(workspace.clone()))?;
    tools.register(ReadFile(workspace.clone()))?;
    tools.register(Search(workspace.clone()))?;
    tools.register(WriteNote(workspace.clone()))?;
    tools.register(CurrentTime)?;
    Ok(tools)
}

/// The current date and time, in UTC.
#[tool(parallel)]
async fn current_time() -> Result<String> {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(Error::application)?
        .as_secs();
    let days = seconds / 86_400;
    let (year, month, day) = civil_from_days(i64::try_from(days).unwrap_or_default());
    let rest = seconds % 86_400;
    Ok(format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rest / 3600,
        rest % 3600 / 60,
        rest % 60
    ))
}

/// A day count since 1970-01-01 as a date (Howard Hinnant's algorithm).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = u32::try_from(doy - (153 * mp + 2) / 5 + 1).unwrap_or(1);
    let month = u32::try_from(if mp < 10 { mp + 3 } else { mp - 9 }).unwrap_or(1);
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

struct ListFiles(Workspace);

impl FunctionTool for ListFiles {
    type Input = Value;
    type Output = Vec<String>;

    fn name(&self) -> &str {
        "list_files"
    }
    fn description(&self) -> &str {
        "List the files of the workspace, with their paths relative to it."
    }
    fn parameters_schema(&self) -> Value {
        json!({"type": "object", "properties": {}, "additionalProperties": false})
    }
    fn supports_parallel(&self) -> bool {
        true
    }
    async fn execute(&self, _: Value) -> Result<Vec<String>> {
        Ok(self
            .0
            .files()
            .iter()
            .map(|path| self.0.relative(path))
            .collect())
    }
}

#[derive(Deserialize)]
struct PathInput {
    path: String,
}

struct ReadFile(Workspace);

impl FunctionTool for ReadFile {
    type Input = PathInput;
    type Output = String;

    fn name(&self) -> &str {
        "read_file"
    }
    fn description(&self) -> &str {
        "Read a text file of the workspace."
    }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {"path": {"type": "string", "description": "The file, relative to the workspace"}},
            "required": ["path"],
            "additionalProperties": false
        })
    }
    fn supports_parallel(&self) -> bool {
        true
    }
    async fn execute(&self, input: PathInput) -> Result<String> {
        let path = self.0.resolve(&input.path)?;
        std::fs::read_to_string(path).map_err(Error::application)
    }
}

#[derive(Deserialize)]
struct SearchInput {
    text: String,
}

struct Search(Workspace);

impl FunctionTool for Search {
    type Input = SearchInput;
    type Output = Vec<String>;

    fn name(&self) -> &str {
        "search"
    }
    fn description(&self) -> &str {
        "Search the workspace's files for words (not case-sensitive): the lines with the most of \
         the words first, as `path:line: text`. Use a few plain words, such as `paint kitchen`."
    }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {"text": {"type": "string", "description": "The words to look for"}},
            "required": ["text"],
            "additionalProperties": false
        })
    }
    fn supports_parallel(&self) -> bool {
        true
    }
    async fn execute(&self, input: SearchInput) -> Result<Vec<String>> {
        let words = search_words(&input.text);
        let mut hits: Vec<(usize, String)> = Vec::new();
        for path in self.0.files() {
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            for (number, line) in text.lines().enumerate() {
                let lower = line.to_lowercase();
                let found = words
                    .iter()
                    .filter(|word| lower.contains(word.as_str()))
                    .count();
                if found > 0 {
                    let hit = format!("{}:{}: {}", self.0.relative(&path), number + 1, line.trim());
                    hits.push((found, hit));
                }
            }
        }
        // The lines with the most of the words first; within those, in file order.
        hits.sort_by_key(|hit| std::cmp::Reverse(hit.0));
        Ok(hits.into_iter().take(20).map(|(_, hit)| hit).collect())
    }
}

/// The words of a query worth looking for: lowercase, without short words, and cut to a stem
/// (`painting` finds `paint`, `plumbers` finds `plumber`).
fn search_words(query: &str) -> Vec<String> {
    const STOP: &[&str] = &[
        "the", "and", "for", "are", "how", "much", "when", "what", "where", "our", "with",
    ];
    query
        .split(|c: char| !c.is_alphanumeric())
        .map(str::to_lowercase)
        .filter(|word| word.len() > 2 && !STOP.contains(&word.as_str()))
        .map(|word| {
            for suffix in ["ing", "ers", "es", "s"] {
                if let Some(stem) = word.strip_suffix(suffix) {
                    if stem.len() >= 3 {
                        return stem.to_owned();
                    }
                }
            }
            word
        })
        .collect()
}

#[derive(Deserialize)]
struct NoteInput {
    name: String,
    text: String,
}

/// Writes a note: the one tool that changes something, so the agent asks before it runs.
struct WriteNote(Workspace);

impl FunctionTool for WriteNote {
    type Input = NoteInput;
    type Output = String;

    fn name(&self) -> &str {
        "write_note"
    }
    fn description(&self) -> &str {
        "Write a note into the workspace's notes/ directory (a person approves it first)."
    }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "A file name, such as `groceries.md`"},
                "text": {"type": "string", "description": "The note"}
            },
            "required": ["name", "text"],
            "additionalProperties": false
        })
    }
    async fn execute(&self, input: NoteInput) -> Result<String> {
        let name = input.name.trim();
        if name.is_empty() || name.contains(['/', '\\']) || name.starts_with('.') {
            return Err(refuse(format!("{name:?} is not a plain file name")));
        }
        let dir = self.0.root().join("notes");
        std::fs::create_dir_all(&dir).map_err(Error::application)?;
        std::fs::write(dir.join(name), input.text).map_err(Error::application)?;
        Ok(format!("wrote notes/{name}"))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(20_407), (2025, 11, 15));
    }

    #[tokio::test]
    async fn paths_stay_inside_the_workspace() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("workspace");
        let workspace = Workspace::new(&root).unwrap();
        let read = ReadFile(workspace.clone());
        assert!(read
            .execute(PathInput {
                path: "README.md".into()
            })
            .await
            .is_ok());
        for path in ["../Cargo.toml", "/etc/hosts", "notes/../../Cargo.toml"] {
            assert!(
                read.execute(PathInput { path: path.into() }).await.is_err(),
                "{path}"
            );
        }
        let hits = Search(workspace)
            .execute(SearchInput {
                text: "PAINT".into(),
            })
            .await
            .unwrap();
        assert!(
            hits.iter().any(|hit| hit.starts_with("notes/house.md:")),
            "{hits:?}"
        );
    }
}
