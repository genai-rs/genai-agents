//! Skills in the [Agent Skills](https://agentskills.io) format, and instruction files
//! (`AGENTS.md`).
//!
//! A skill is a directory with a `SKILL.md`: a front matter with its `name` and `description`,
//! then instructions in Markdown, and any files the instructions refer to.
//!
//! ```markdown
//! ---
//! name: release-notes
//! description: Write release notes from a list of merged pull requests.
//! ---
//! Group the changes by kind, newest first. Use `template.md` for the layout.
//! ```
//!
//! The model does not get every skill's instructions up front: it gets a list of the skills
//! ([`SkillSet::instructions`], within a budget), and loads the one it needs with the
//! `load_skill` tool ([`SkillSet::tool`]), which also hands over the skill's files.
//!
//! Skills are untrusted content:
//!
//! - they are read only from the directories the application lists ([`SkillSet::load`]);
//! - a `SKILL.md` or a file that is a symbolic link out of its skill's directory is refused;
//! - nothing in a skill is run when it is loaded, and a skill cannot add tools.
//!
//! Front matter fields besides `name` and `description`:
//!
//! - `disable-model-invocation: true`: left out of the model's list (the application loads it
//!   itself, [`SkillSet::get`]);
//! - `requires:` with `bins` (programs on the `PATH`), `env` (variables that are set) and `os`
//!   (`linux`, `macos`, `windows`): a skill whose requirements are not met is skipped, with a
//!   warning.
//!
//! ```
//! use genai_agents::skills::{Skill, SkillSet};
//!
//! let skill = Skill::parse(
//!     "---\nname: haiku\ndescription: Answer in a haiku.\n---\nThree lines: 5, 7, 5 syllables.",
//! )?;
//! let skills = SkillSet::from_skills(vec![skill]);
//! let list = skills.instructions(1_000);
//! assert!(list.contains("- haiku: Answer in a haiku."));
//! # Ok::<(), genai_agents::skills::SkillError>(())
//! ```

use std::collections::BTreeMap;
#[cfg(not(target_family = "wasm"))]
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::error::Result;
use crate::tools::FunctionTool;

/// Why a skill could not be read.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SkillError {
    /// `SKILL.md` has no front matter, or a field the format needs is missing or invalid.
    #[error("invalid SKILL.md: {0}")]
    Invalid(String),
    /// A file of the skill could not be read, or is outside its directory.
    #[error("skill file: {0}")]
    File(String),
}

/// What a skill needs to work.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Requirements {
    /// Programs that must be on the `PATH`.
    pub bins: Vec<String>,
    /// Environment variables that must be set.
    pub env: Vec<String>,
    /// The operating systems it works on (`linux`, `macos`, `windows`); empty for any.
    pub os: Vec<String>,
}

impl Requirements {
    /// What is missing on this machine, if anything.
    #[must_use]
    pub fn missing(&self) -> Vec<String> {
        let mut missing = Vec::new();
        if !self.os.is_empty() && !self.os.iter().any(|os| os == std::env::consts::OS) {
            missing.push(format!("an operating system of {}", self.os.join(", ")));
        }
        for variable in &self.env {
            if std::env::var_os(variable).is_none_or(|value| value.is_empty()) {
                missing.push(format!("the variable {variable}"));
            }
        }
        for bin in &self.bins {
            if !on_path(bin) {
                missing.push(format!("the program {bin}"));
            }
        }
        missing
    }
}

#[cfg(not(target_family = "wasm"))]
fn on_path(bin: &str) -> bool {
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&path).any(|dir| {
        let candidate = dir.join(bin);
        candidate.is_file() || (cfg!(windows) && dir.join(format!("{bin}.exe")).is_file())
    })
}

#[cfg(target_family = "wasm")]
fn on_path(_: &str) -> bool {
    false
}

/// A skill: its name, description and instructions, and where its files are.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Skill {
    /// Its name: lowercase letters, digits and `-`, 1 to 64 characters.
    pub name: String,
    /// What it does and when to use it: what the model decides by.
    pub description: String,
    /// The instructions (`SKILL.md` without its front matter).
    pub instructions: String,
    /// Whether it is left out of the model's list.
    pub hidden: bool,
    /// What it needs.
    pub requires: Requirements,
    /// The other front matter fields, as text.
    pub fields: BTreeMap<String, String>,
    /// Its directory, when it was loaded from one.
    #[cfg(not(target_family = "wasm"))]
    pub dir: Option<PathBuf>,
    /// Its files by relative path, when it was made in memory.
    pub files: BTreeMap<String, String>,
}

impl Skill {
    /// A skill from the text of a `SKILL.md`.
    ///
    /// # Errors
    ///
    /// [`SkillError::Invalid`] when the front matter is missing, or `name` or `description` is
    /// missing or invalid.
    pub fn parse(text: &str) -> Result<Self, SkillError> {
        let text = text.strip_prefix('\u{feff}').unwrap_or(text);
        let rest = text
            .strip_prefix("---\n")
            .or_else(|| text.strip_prefix("---\r\n"))
            .ok_or_else(|| SkillError::Invalid("no front matter (`---` first)".into()))?;
        let end = rest
            .find("\n---")
            .ok_or_else(|| SkillError::Invalid("the front matter does not end (`---`)".into()))?;
        let front = &rest[..end];
        let body = rest[end + 4..]
            .trim_start_matches(['-'])
            .trim_start_matches(['\r', '\n']);
        let matter = FrontMatter::parse(front)?;
        let name = matter
            .scalar("name")
            .ok_or_else(|| SkillError::Invalid("`name` is missing".into()))?;
        if name.is_empty()
            || name.len() > 64
            || !name
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
            || name.starts_with('-')
            || name.ends_with('-')
            || name.contains("--")
        {
            return Err(SkillError::Invalid(format!(
                "`name` must be 1 to 64 lowercase letters, digits and single hyphens, not {name:?}"
            )));
        }
        let description = matter
            .scalar("description")
            .ok_or_else(|| SkillError::Invalid("`description` is missing".into()))?;
        if description.trim().is_empty() || description.chars().count() > 1024 {
            return Err(SkillError::Invalid(
                "`description` must be 1 to 1024 characters".into(),
            ));
        }
        let hidden = matter
            .scalar("disable-model-invocation")
            .is_some_and(|value| value == "true");
        let requires = Requirements {
            bins: matter.list("requires.bins"),
            env: matter.list("requires.env"),
            os: matter.list("requires.os"),
        };
        let fields = matter
            .scalars
            .into_iter()
            .filter(|(key, _)| !matches!(key.as_str(), "name" | "description"))
            .collect();
        Ok(Self {
            name,
            description: description.trim().to_owned(),
            instructions: body.trim_end().to_owned(),
            hidden,
            requires,
            fields,
            #[cfg(not(target_family = "wasm"))]
            dir: None,
            files: BTreeMap::new(),
        })
    }

    /// The same skill, with a file (made in memory: on WebAssembly, or in tests).
    #[must_use]
    pub fn with_file(mut self, path: impl Into<String>, content: impl Into<String>) -> Self {
        self.files.insert(path.into(), content.into());
        self
    }

    /// A file of the skill, by its path relative to the skill's directory.
    ///
    /// # Errors
    ///
    /// [`SkillError::File`] when the path leaves the directory (absolute, `..`, a symbolic link
    /// out of it), or the file cannot be read.
    pub fn file(&self, path: &str) -> Result<String, SkillError> {
        if path.is_empty()
            || path.starts_with('/')
            || path.starts_with('\\')
            || path.contains(':')
            || path.split(['/', '\\']).any(|part| part == "..")
        {
            return Err(SkillError::File(format!(
                "{path:?} is not a path inside the skill"
            )));
        }
        if let Some(content) = self.files.get(path) {
            return Ok(content.clone());
        }
        #[cfg(not(target_family = "wasm"))]
        if let Some(dir) = &self.dir {
            let file = inside(dir, &dir.join(path))?;
            return std::fs::read_to_string(&file)
                .map_err(|error| SkillError::File(format!("{path}: {error}")));
        }
        Err(SkillError::File(format!("{path}: no such file")))
    }

    /// Read the skill in `dir` (its `SKILL.md`); its name must be the directory's.
    ///
    /// # Errors
    ///
    /// As [`parse`](Self::parse), and [`SkillError::File`] when `SKILL.md` cannot be read or
    /// leaves the directory.
    #[cfg(not(target_family = "wasm"))]
    pub fn load(dir: &Path) -> Result<Self, SkillError> {
        let dir = dir
            .canonicalize()
            .map_err(|error| SkillError::File(format!("{}: {error}", dir.display())))?;
        let file = inside(&dir, &dir.join("SKILL.md"))?;
        let text = std::fs::read_to_string(&file)
            .map_err(|error| SkillError::File(format!("SKILL.md: {error}")))?;
        let mut skill = Self::parse(&text)?;
        let folder = dir
            .file_name()
            .map(|name| name.to_string_lossy().into_owned());
        if folder.as_deref() != Some(skill.name.as_str()) {
            return Err(SkillError::Invalid(format!(
                "the skill {:?} is in a directory named {:?}",
                skill.name,
                folder.unwrap_or_default()
            )));
        }
        skill.dir = Some(dir);
        Ok(skill)
    }
}

/// `path`, resolved, when it stays inside `dir` (resolved); symbolic links are followed first.
#[cfg(not(target_family = "wasm"))]
fn inside(dir: &Path, path: &Path) -> Result<PathBuf, SkillError> {
    let resolved = path
        .canonicalize()
        .map_err(|error| SkillError::File(format!("{}: {error}", path.display())))?;
    if resolved.starts_with(dir) {
        Ok(resolved)
    } else {
        Err(SkillError::File(format!(
            "{} leads outside the skill's directory",
            path.display()
        )))
    }
}

/// The front matter: a small subset of YAML (`key: value`, quoted or not; lists inline
/// `[a, b]` or as `- item` lines; one level of nesting under a key).
struct FrontMatter {
    scalars: BTreeMap<String, String>,
    lists: BTreeMap<String, Vec<String>>,
}

impl FrontMatter {
    fn parse(text: &str) -> Result<Self, SkillError> {
        let mut scalars = BTreeMap::new();
        let mut lists: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut parent: Option<String> = None;
        let mut list_key: Option<String> = None;
        let mut folded: Option<(String, Vec<String>)> = None;
        for line in text.lines() {
            if line.trim().is_empty() || line.trim_start().starts_with('#') {
                continue;
            }
            let indent = line.len() - line.trim_start().len();
            let content = line.trim();
            if let Some((key, lines)) = &mut folded {
                if indent > 0 {
                    lines.push(content.to_owned());
                    continue;
                }
                scalars.insert(key.clone(), lines.join(" "));
                folded = None;
            }
            if let Some(item) = content.strip_prefix("- ") {
                let Some(key) = &list_key else {
                    return Err(SkillError::Invalid(format!(
                        "a list item without a key: {line}"
                    )));
                };
                lists.entry(key.clone()).or_default().push(unquote(item));
                continue;
            }
            let Some((key, value)) = content.split_once(':') else {
                return Err(SkillError::Invalid(format!("not `key: value`: {line}")));
            };
            let key = key.trim().to_owned();
            let value = value.trim();
            if indent == 0 {
                parent = None;
            }
            let full = match (&parent, indent > 0) {
                (Some(parent), true) => format!("{parent}.{key}"),
                _ => key.clone(),
            };
            if value.is_empty() {
                if indent == 0 {
                    parent = Some(key);
                }
                list_key = Some(full);
            } else if value == ">" || value == "|" || value == ">-" || value == "|-" {
                folded = Some((full, Vec::new()));
            } else if let Some(inner) = value.strip_prefix('[').and_then(|v| v.strip_suffix(']')) {
                lists.insert(
                    full,
                    inner
                        .split(',')
                        .map(unquote)
                        .filter(|item| !item.is_empty())
                        .collect(),
                );
            } else {
                scalars.insert(full, unquote(value));
            }
        }
        if let Some((key, lines)) = folded {
            scalars.insert(key, lines.join(" "));
        }
        Ok(Self { scalars, lists })
    }

    fn scalar(&self, key: &str) -> Option<String> {
        self.scalars.get(key).cloned()
    }

    fn list(&self, key: &str) -> Vec<String> {
        self.lists
            .get(key)
            .cloned()
            .or_else(|| self.scalars.get(key).map(|value| vec![value.clone()]))
            .unwrap_or_default()
    }
}

fn unquote(value: &str) -> String {
    let value = value.trim();
    for quote in ['"', '\''] {
        if let Some(inner) = value
            .strip_prefix(quote)
            .and_then(|value| value.strip_suffix(quote))
        {
            return inner.to_owned();
        }
    }
    value.to_owned()
}

/// A skill that was not loaded, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct SkillWarning {
    /// Where it is.
    pub location: String,
    /// Why it was skipped.
    pub reason: String,
}

/// The skills an agent can use.
#[derive(Debug, Clone, Default)]
pub struct SkillSet {
    skills: BTreeMap<String, Skill>,
}

impl SkillSet {
    /// These skills (a later one of the same name wins).
    #[must_use]
    pub fn from_skills(skills: Vec<Skill>) -> Self {
        Self {
            skills: skills
                .into_iter()
                .map(|skill| (skill.name.clone(), skill))
                .collect(),
        }
    }

    /// The skills in these directories: each subdirectory with a `SKILL.md` is a skill. A later
    /// directory's skill wins over an earlier one of the same name (a project's over a user's).
    /// Skills that do not parse, leave their directory, or whose requirements are not met are
    /// skipped, with a warning each.
    #[cfg(not(target_family = "wasm"))]
    #[must_use]
    pub fn load(dirs: &[impl AsRef<Path>]) -> (Self, Vec<SkillWarning>) {
        let mut skills = BTreeMap::new();
        let mut warnings = Vec::new();
        for root in dirs {
            let root = root.as_ref();
            let Ok(entries) = std::fs::read_dir(root) else {
                continue;
            };
            let Ok(canonical_root) = root.canonicalize() else {
                continue;
            };
            let mut dirs: Vec<PathBuf> = entries
                .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                .filter(|path| path.join("SKILL.md").exists())
                .collect();
            dirs.sort();
            for dir in dirs {
                let warn = |reason: String| SkillWarning {
                    location: dir.display().to_string(),
                    reason,
                };
                // A skill directory that is a link out of the root is refused.
                if let Err(error) = inside(&canonical_root, &dir) {
                    warnings.push(warn(error.to_string()));
                    continue;
                }
                match Skill::load(&dir) {
                    Ok(skill) => {
                        let missing = skill.requires.missing();
                        if missing.is_empty() {
                            skills.insert(skill.name.clone(), skill);
                        } else {
                            warnings.push(warn(format!("needs {}", missing.join(", "))));
                        }
                    }
                    Err(error) => warnings.push(warn(error.to_string())),
                }
            }
        }
        (Self { skills }, warnings)
    }

    /// A skill by name, hidden ones included.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Skill> {
        self.skills.get(name)
    }

    /// The skills, by name.
    pub fn iter(&self) -> impl Iterator<Item = &Skill> {
        self.skills.values()
    }

    /// The number of skills, hidden ones included.
    #[must_use]
    pub fn len(&self) -> usize {
        self.skills.len()
    }

    /// Whether there is no skill.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    /// The list of skills for the instructions, at most `budget` characters: each skill's name
    /// and description; when they do not fit, the descriptions are left out, from the last skill
    /// back, then the names that still do not fit (with a count of them). Hidden skills are not
    /// listed. Empty when there is no skill to list.
    #[must_use]
    pub fn instructions(&self, budget: usize) -> String {
        let visible: Vec<&Skill> = self.skills.values().filter(|skill| !skill.hidden).collect();
        if visible.is_empty() {
            return String::new();
        }
        let head = "Skills are instructions for particular tasks. When one fits the task, load it \
                    with the load_skill tool before you start, and follow it.\n\nSkills:";
        let entry = |skill: &Skill, describe: bool| {
            if describe {
                format!("\n- {}: {}", skill.name, skill.description)
            } else {
                format!("\n- {}", skill.name)
            }
        };
        let mut described = visible.len();
        loop {
            let text: String = visible
                .iter()
                .enumerate()
                .map(|(index, skill)| entry(skill, index < described))
                .collect();
            if head.len() + text.len() <= budget || described == 0 {
                if head.len() + text.len() <= budget {
                    return format!("{head}{text}");
                }
                break;
            }
            described -= 1;
        }
        // Names only, and as many as fit.
        let mut out = head.to_owned();
        for (index, skill) in visible.iter().enumerate() {
            let line = entry(skill, false);
            let more = format!("\n- and {} more", visible.len() - index);
            if out.len() + line.len() + more.len() > budget {
                out.push_str(&more);
                return out;
            }
            out.push_str(&line);
        }
        out
    }

    /// [`instructions`](Self::instructions) within `percent` of the model's context window (four
    /// characters a token); 2 000 characters when the window is not known.
    #[must_use]
    pub fn instructions_for(&self, info: &crate::ModelInfo, percent: u8) -> String {
        let budget = info.context_window.map_or(2_000, |window| {
            usize::try_from(u64::from(window) * u64::from(percent) * 4 / 100).unwrap_or(usize::MAX)
        });
        self.instructions(budget)
    }

    /// The `load_skill` tool over these skills.
    #[must_use]
    pub fn tool(&self) -> LoadSkill {
        LoadSkill {
            skills: self.clone(),
        }
    }
}

/// The `load_skill` tool: a skill's instructions, or one of its files.
#[derive(Debug, Clone)]
pub struct LoadSkill {
    skills: SkillSet,
}

/// The input of [`LoadSkill`].
#[derive(Debug, Deserialize)]
pub struct LoadSkillInput {
    /// The skill.
    pub name: String,
    /// A file of the skill, relative to its directory; none for its instructions.
    #[serde(default)]
    pub file: Option<String>,
}

impl FunctionTool for LoadSkill {
    type Input = LoadSkillInput;
    type Output = String;

    fn name(&self) -> &str {
        "load_skill"
    }

    fn description(&self) -> &str {
        "Load a skill's instructions by name, or one of the files its instructions refer to."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The skill's name"},
                "file": {
                    "type": "string",
                    "description": "A file the skill's instructions refer to, relative to the skill; leave out for the instructions"
                }
            },
            "required": ["name"],
            "additionalProperties": false,
        })
    }

    fn supports_parallel(&self) -> bool {
        true
    }

    fn untrusted_output(&self) -> bool {
        true
    }

    async fn execute(&self, input: LoadSkillInput) -> Result<String> {
        let skill = self
            .skills
            .get(&input.name)
            .filter(|skill| !skill.hidden)
            .ok_or_else(|| {
                crate::Error::application(std::io::Error::other(format!(
                    "no skill named {:?}",
                    input.name
                )))
            })?;
        match input.file {
            None => Ok(skill.instructions.clone()),
            Some(file) => skill.file(&file).map_err(|error| {
                crate::Error::application(std::io::Error::other(error.to_string()))
            }),
        }
    }
}

/// The instruction files (`AGENTS.md`) from `root` down to `dir`: the root's first, the most
/// specific last, each under a line naming it, within `budget` bytes in all (the deepest kept
/// first: it is the most specific). `dir` must be inside `root`; files that are links out of
/// `root` are left out.
///
/// # Errors
///
/// [`SkillError::File`] when `dir` is not inside `root`.
#[cfg(not(target_family = "wasm"))]
pub fn instruction_files(root: &Path, dir: &Path, budget: usize) -> Result<String, SkillError> {
    let root = root
        .canonicalize()
        .map_err(|error| SkillError::File(format!("{}: {error}", root.display())))?;
    let dir = inside(&root, dir)?;
    let mut dirs: Vec<PathBuf> = dir
        .ancestors()
        .take_while(|ancestor| ancestor.starts_with(&root))
        .map(Path::to_path_buf)
        .collect();
    dirs.reverse();
    let mut found: Vec<(String, String)> = Vec::new();
    for current in dirs {
        let file = current.join("AGENTS.md");
        if !file.is_file() {
            continue;
        }
        let Ok(file) = inside(&root, &file) else {
            continue;
        };
        let Ok(text) = std::fs::read_to_string(&file) else {
            continue;
        };
        // The same on every platform: the model sees it, and `/` is what it knows.
        let label = file.strip_prefix(&root).map_or_else(
            |_| "AGENTS.md".to_owned(),
            |path| {
                path.components()
                    .map(|part| part.as_os_str().to_string_lossy())
                    .collect::<Vec<_>>()
                    .join("/")
            },
        );
        found.push((label, text.trim().to_owned()));
    }
    // From the deepest up, while the budget lasts; the last one in may be cut.
    let mut kept: Vec<String> = Vec::new();
    let mut left = budget;
    for (label, text) in found.iter().rev() {
        let section = format!("# {label}\n\n{text}");
        if section.len() <= left {
            left -= section.len();
            kept.push(section);
        } else {
            let mut end = left;
            while end > 0 && !section.is_char_boundary(end) {
                end -= 1;
            }
            if end > label.len() + 4 {
                kept.push(format!("{}\n[…cut to fit]", &section[..end]));
            }
            break;
        }
    }
    kept.reverse();
    Ok(kept.join("\n\n"))
}
