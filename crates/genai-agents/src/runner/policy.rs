//! The tool policy: which registered tools a run may use.

use crate::tools::ToolKey;

/// Which registered tools a run may use, by name pattern.
///
/// A pattern is a tool's name, or `namespace.name` for a tool in a namespace, where `*` stands
/// for any run of characters: `"get_*"`, `"fs.*"`, `"*"`.
///
/// - A tool that matches a [`deny`](Self::deny) pattern is left out, whatever else matches.
/// - When [`allow`](Self::allow) is not empty, only the tools that match one of its patterns
///   are kept.
///
/// A tool left out is not advertised, a tool search does not find it, and a call of it (a
/// model that guessed its name) is answered "not allowed" without running.
///
/// ```
/// use genai_agents::tools::ToolKey;
/// use genai_agents::ToolPolicy;
///
/// let policy = ToolPolicy::new().allow("fs.*").allow("get_weather").deny("fs.delete_*");
/// assert!(policy.permits(&ToolKey::namespaced("fs", "read")));
/// assert!(!policy.permits(&ToolKey::namespaced("fs", "delete_tree")));
/// assert!(policy.permits(&ToolKey::new("get_weather")));
/// assert!(!policy.permits(&ToolKey::new("send_email")));
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ToolPolicy {
    /// The patterns of the tools to keep; empty keeps every tool not denied.
    pub allow: Vec<String>,
    /// The patterns of the tools to leave out.
    pub deny: Vec<String>,
}

impl ToolPolicy {
    /// A policy that keeps every tool.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The same policy, keeping the tools that match `pattern` (and, once any pattern is
    /// allowed, only those).
    #[must_use]
    pub fn allow(mut self, pattern: impl Into<String>) -> Self {
        self.allow.push(pattern.into());
        self
    }

    /// The same policy, leaving out the tools that match `pattern`.
    #[must_use]
    pub fn deny(mut self, pattern: impl Into<String>) -> Self {
        self.deny.push(pattern.into());
        self
    }

    /// Whether the policy keeps this tool.
    #[must_use]
    pub fn permits(&self, key: &ToolKey) -> bool {
        let name = match &key.namespace {
            Some(namespace) => format!("{namespace}.{}", key.name),
            None => key.name.clone(),
        };
        let matches = |pattern: &String| glob(pattern, &name);
        !self.deny.iter().any(matches) && (self.allow.is_empty() || self.allow.iter().any(matches))
    }

    /// Whether the policy keeps every tool.
    #[must_use]
    pub fn is_open(&self) -> bool {
        self.allow.is_empty() && self.deny.is_empty()
    }
}

/// Whether `text` matches `pattern`, where `*` matches any run of characters.
fn glob(pattern: &str, text: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == text;
    }
    let (first, rest) = (parts[0], &parts[1..]);
    let Some(mut remaining) = text.strip_prefix(first) else {
        return false;
    };
    let last = rest.len() - 1;
    for (index, part) in rest.iter().enumerate() {
        if index == last {
            return remaining.ends_with(part);
        }
        match remaining.find(part) {
            Some(at) => remaining = &remaining[at + part.len()..],
            None => return false,
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn globs() {
        assert!(glob("*", "anything"));
        assert!(glob("get_*", "get_weather"));
        assert!(!glob("get_*", "set_weather"));
        assert!(glob("fs.*", "fs.read"));
        assert!(glob("*_file", "read_file"));
        assert!(glob("a*b*c", "aXbYc"));
        assert!(!glob("a*b*c", "aXbY"));
        assert!(glob("exact", "exact"));
        assert!(!glob("exact", "exactly"));
    }
}
