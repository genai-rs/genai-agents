//! The test server as a child process over stdio: this test binary is both. Run with `--serve`,
//! it is the server; otherwise it starts itself with `--serve` and calls a tool. (It has no
//! test harness, whose output would mix with the protocol on stdout.)
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[cfg(feature = "stdio")]
mod support;

#[cfg(feature = "stdio")]
#[tokio::main]
async fn main() {
    use genai_agents::tools::ToolRegistry;
    use genai_agents::ToolCall;
    use genai_agents_mcp::{McpOptions, McpServer};
    use rmcp::ServiceExt;

    if std::env::args().any(|arg| arg == "--serve") {
        let running = support::TestServer::default()
            .serve(rmcp::transport::stdio())
            .await
            .unwrap();
        let _ = running.waiting().await;
        return;
    }

    let mut command = tokio::process::Command::new(std::env::current_exe().unwrap());
    command.arg("--serve");
    let server = McpServer::stdio("household", command, McpOptions::new())
        .await
        .unwrap();
    let mut tools = ToolRegistry::new();
    let registered = server.register(&mut tools).await.unwrap();
    assert_eq!(registered.len(), 10);
    let output = tools
        .answer_call(
            ToolCall::function("c1", "get_weather", r#"{"city":"Ghent"}"#)
                .in_namespace("household"),
        )
        .await
        .unwrap();
    assert!(output.success);
    assert_eq!(output.body.as_text(), Some("12 degrees in Ghent"));
    server.close();
    println!("test stdio ... ok");
}

#[cfg(not(feature = "stdio"))]
fn main() {}
