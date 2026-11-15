//! Skills: SKILL.md parsing, loading from directories (and what is refused), the model's list
//! within a budget, the load_skill tool, and AGENTS.md files.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg(not(target_family = "wasm"))]

use std::fs;
use std::path::{Path, PathBuf};

use genai_agents::skills::{instruction_files, Skill, SkillError, SkillSet};
use genai_agents::testing::{Reply, ScriptedModel};
use genai_agents::tools::ToolRegistry;
use genai_agents::{Conversation, Item, Runner, ToolCall};

/// A directory of its own per test, removed afterwards.
struct Scratch(PathBuf);

impl Scratch {
    fn new(test: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "genai-agents-skills-{test}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn write(&self, relative: &str, text: &str) -> PathBuf {
        let path = self.0.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, text).unwrap();
        path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn skill_md(name: &str, description: &str, extra: &str, body: &str) -> String {
    format!("---\nname: {name}\ndescription: {description}\n{extra}---\n{body}\n")
}

#[test]
fn skill_md_is_parsed() {
    let skill = Skill::parse(
        "---\n\
         name: pdf-forms\n\
         description: >\n  Fill PDF forms\n  from a JSON file.\n\
         license: \"MIT\"\n\
         disable-model-invocation: true\n\
         requires:\n  bins: [qpdf, 'pdftk']\n  env:\n    - FORMS_KEY\n  os: linux\n\
         ---\n\
         Use `fill.md`.\n",
    )
    .unwrap();
    assert_eq!(skill.name, "pdf-forms");
    assert_eq!(skill.description, "Fill PDF forms from a JSON file.");
    assert_eq!(skill.instructions, "Use `fill.md`.");
    assert!(skill.hidden);
    assert_eq!(skill.requires.bins, ["qpdf", "pdftk"]);
    assert_eq!(skill.requires.env, ["FORMS_KEY"]);
    assert_eq!(skill.requires.os, ["linux"]);
    assert_eq!(skill.fields["license"], "MIT");

    for (text, why) in [
        ("no front matter", "front matter"),
        ("---\nname: x\n", "does not end"),
        ("---\ndescription: d\n---\n", "`name` is missing"),
        ("---\nname: Bad_Name\ndescription: d\n---\n", "lowercase"),
        ("---\nname: a--b\ndescription: d\n---\n", "lowercase"),
        ("---\nname: ok\n---\n", "`description` is missing"),
    ] {
        let error = Skill::parse(text).unwrap_err();
        assert!(
            matches!(&error, SkillError::Invalid(message) if message.contains(why)),
            "{text:?}: {error}"
        );
    }
}

#[test]
fn skills_load_from_the_listed_directories_only() {
    let user = Scratch::new("user");
    let project = Scratch::new("project");
    user.write(
        "notes/SKILL.md",
        &skill_md("notes", "Old notes skill.", "", "old"),
    );
    project.write(
        "notes/SKILL.md",
        &skill_md("notes", "Take notes.", "", "Write them down."),
    );
    project.write("notes/template.md", "# Notes\n");
    project.write(
        "mismatch/SKILL.md",
        &skill_md("other-name", "Wrong directory.", "", ""),
    );
    project.write(
        "needs-env/SKILL.md",
        &skill_md(
            "needs-env",
            "Needs a variable.",
            "requires:\n  env: [GENAI_AGENTS_TEST_UNSET_VARIABLE]\n",
            "",
        ),
    );
    project.write(
        "secret/SKILL.md",
        &skill_md(
            "secret",
            "Only for the app.",
            "disable-model-invocation: true\n",
            "s",
        ),
    );
    project.write("broken/SKILL.md", "no front matter");

    let (skills, warnings) = SkillSet::load(&[&user.0, &project.0]);
    // The project's skill wins over the user's.
    let notes = skills.get("notes").unwrap();
    assert_eq!(notes.description, "Take notes.");
    assert_eq!(notes.file("template.md").unwrap(), "# Notes\n");
    assert!(skills.get("secret").unwrap().hidden);
    assert_eq!(skills.len(), 2);
    let reasons: Vec<String> = warnings.iter().map(|w| w.reason.clone()).collect();
    assert_eq!(warnings.len(), 3, "{reasons:?}");
    assert!(reasons.iter().any(|r| r.contains("directory named")));
    assert!(reasons
        .iter()
        .any(|r| r.contains("GENAI_AGENTS_TEST_UNSET_VARIABLE")));
    assert!(reasons.iter().any(|r| r.contains("front matter")));

    // Paths out of the skill are refused.
    for path in ["../mismatch/SKILL.md", "/etc/passwd", "a/../../x", "C:\\x"] {
        assert!(
            matches!(notes.file(path), Err(SkillError::File(_))),
            "{path}"
        );
    }
}

#[cfg(unix)]
#[test]
fn symbolic_links_out_of_a_skill_are_refused() {
    use std::os::unix::fs::symlink;
    let outside = Scratch::new("outside");
    let secret = outside.write("secret.txt", "do not read");
    outside.write(
        "elsewhere/SKILL.md",
        &skill_md("elsewhere", "Lives outside.", "", "x"),
    );
    let root = Scratch::new("root");
    root.write("inside/SKILL.md", &skill_md("inside", "Inside.", "", "y"));
    symlink(&secret, root.0.join("inside/leak.txt")).unwrap();
    symlink(outside.0.join("elsewhere"), root.0.join("elsewhere")).unwrap();

    let (skills, warnings) = SkillSet::load(&[&root.0]);
    assert!(skills.get("elsewhere").is_none());
    assert!(warnings.iter().any(|w| w.reason.contains("outside")));
    let inside = skills.get("inside").unwrap();
    let error = inside.file("leak.txt").unwrap_err();
    assert!(error.to_string().contains("outside"), "{error}");
}

#[test]
fn the_list_fits_its_budget() {
    let skills = SkillSet::from_skills(
        (0..5)
            .map(|index| {
                Skill::parse(&skill_md(
                    &format!("skill-{index}"),
                    &format!("Does thing number {index}, at some length."),
                    "",
                    "body",
                ))
                .unwrap()
            })
            .collect(),
    );
    let full = skills.instructions(10_000);
    assert!(full.contains("- skill-4: Does thing number 4, at some length."));
    // Tighter: the last descriptions go first.
    let tighter = skills.instructions(full.len() - 30);
    assert!(tighter.contains("- skill-0: Does thing number 0"));
    assert!(tighter.contains("\n- skill-4\n") || tighter.ends_with("\n- skill-4"));
    assert!(tighter.len() <= full.len() - 30);
    // Tightest: names only, then a count of the rest.
    let head = full.find("\n- ").unwrap();
    let tightest = skills.instructions(head + 40);
    assert!(tightest.contains("more"), "{tightest}");
    assert!(!tightest.contains("Does thing"));
    assert_eq!(SkillSet::default().instructions(1_000), "");
}

#[tokio::test]
async fn the_model_loads_a_skill_with_the_tool() {
    let skills = SkillSet::from_skills(vec![
        Skill::parse(&skill_md(
            "haiku",
            "Answer in a haiku.",
            "",
            "5, 7, 5 syllables.",
        ))
        .unwrap()
        .with_file("examples.md", "Old pond, frog jumps in"),
        Skill::parse(&skill_md(
            "hidden",
            "x",
            "disable-model-invocation: true\n",
            "no",
        ))
        .unwrap(),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(skills.tool()).unwrap();
    let model = ScriptedModel::new(|index, _| match index {
        0 => Reply::items(vec![
            Item::ToolCall(ToolCall::function(
                "l1",
                "load_skill",
                r#"{"name":"haiku"}"#,
            )),
            Item::ToolCall(ToolCall::function(
                "l2",
                "load_skill",
                r#"{"name":"haiku","file":"examples.md"}"#,
            )),
            Item::ToolCall(ToolCall::function(
                "l3",
                "load_skill",
                r#"{"name":"hidden"}"#,
            )),
        ]),
        _ => Reply::text("Autumn rain."),
    });
    let mut runner = Runner::new(model.clone(), tools);
    runner
        .run(&mut Conversation::user("A poem?"))
        .await
        .unwrap();
    let outputs = model.requests()[1].outputs();
    assert_eq!(outputs[0].body.as_text(), Some("\"5, 7, 5 syllables.\""));
    assert_eq!(
        outputs[1].body.as_text(),
        Some("\"Old pond, frog jumps in\"")
    );
    assert!(!outputs[2].success);
    // A skill's text is outside content; the failure is the runner's own.
    assert!(outputs[0].untrusted && outputs[1].untrusted);
}

#[test]
fn agents_md_from_the_root_down() {
    let repo = Scratch::new("repo");
    repo.write("AGENTS.md", "Use cargo.");
    repo.write("crates/AGENTS.md", "Crates follow the workspace lints.");
    let deep = repo.0.join("crates/core/src");
    fs::create_dir_all(&deep).unwrap();
    let text = instruction_files(&repo.0, &deep, 10_000).unwrap();
    assert_eq!(
        text,
        "# AGENTS.md\n\nUse cargo.\n\n# crates/AGENTS.md\n\nCrates follow the workspace lints."
    );
    // A small budget keeps the most specific file.
    let small = instruction_files(&repo.0, &deep, 60).unwrap();
    assert!(small.starts_with("# crates/AGENTS.md"), "{small}");
    // A directory outside the root is refused.
    let other = Scratch::new("other");
    assert!(instruction_files(&repo.0, &other.0, 100).is_err());
    let _ = Path::new("unused");
}
