# Skills and instruction files

## Skills

A skill is a directory with a `SKILL.md`, in the [Agent Skills](https://agentskills.io) format: a front matter with
its `name` (lowercase letters, digits and single hyphens, the directory's name) and `description`, then instructions,
and any files the instructions refer to.

```markdown
---
name: release-notes
description: Write release notes from a list of merged pull requests.
requires:
  bins: [git]
---
Group the changes by kind, newest first. Use `template.md` for the layout.
```

The model gets a list of the skills in its instructions, and loads the one it needs with the `load_skill` tool:

```rust,ignore
use genai_agents::skills::SkillSet;

let (skills, warnings) = SkillSet::load(&[user_skills_dir, project_skills_dir]);
for warning in &warnings {
    tracing::warn!(skill = %warning.location, "{}", warning.reason);
}
let mut tools = ToolRegistry::new();
tools.register(skills.tool())?;
let agent = Agent::new("assistant", model)
    .with_tools(tools)
    .instructions(format!("You are an assistant.\n\n{}", skills.instructions_for(&info, 2)));
```

- **The list** (`instructions(budget)`, or `instructions_for(&info, percent)` within a share of the context window)
  names each skill with its description. When it does not fit, descriptions go first, from the last skill back, then
  the names that still do not fit, with a count of them.
- **`load_skill`** takes a skill's `name`, and optionally a `file` relative to the skill: its instructions or that
  file. Its answers are marked untrusted.
- **Front matter:** `disable-model-invocation: true` leaves a skill out of the model's list (the application uses it
  through `SkillSet::get`); `requires` with `bins` (on the `PATH`), `env` (set) and `os` (`linux`, `macos`,
  `windows`) skips a skill whose needs are not met, with a warning. Other fields are kept as text (`Skill::fields`).

Skills are untrusted content, and loading them is guarded:

- they are read only from the directories given to `SkillSet::load` (a later directory's skill wins, so a project's
  replaces a user's);
- a skill directory, a `SKILL.md` or a file that is a symbolic link out of its directory is refused, and so is a
  path with `..` or an absolute one;
- nothing in a skill is run when it is loaded, and a skill cannot add tools.

On WebAssembly there is no file system: make skills in memory with `Skill::parse` and `Skill::with_file`, and
`SkillSet::from_skills`.

## Instruction files

`instruction_files(root, dir, budget)` reads the `AGENTS.md` files from the project's root down to the working
directory: the root's first, the most specific last, each under a line that names it. Within the budget the deepest
are kept first, since they are the most specific; files that are links out of the root are left out. Put the text in
the agent's instructions.
