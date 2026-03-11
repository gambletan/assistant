use crate::error::AssistantError;
use crate::providers::{ToolInfo, ToolProvider};
use serde_json::Value;
use std::process::Command;

/// A ToolProvider with built-in system tools.
pub struct BuiltinTools {
    /// Allowlist of shell commands that can be executed.
    shell_allowlist: Vec<String>,
}

impl BuiltinTools {
    pub fn new() -> Self {
        Self {
            shell_allowlist: vec![
                "ls".to_string(),
                "cat".to_string(),
                "echo".to_string(),
                "date".to_string(),
                "pwd".to_string(),
                "whoami".to_string(),
                "wc".to_string(),
                "head".to_string(),
                "tail".to_string(),
                "find".to_string(),
                "grep".to_string(),
                "which".to_string(),
                "uname".to_string(),
                "df".to_string(),
                "du".to_string(),
                "env".to_string(),
                "printenv".to_string(),
            ],
        }
    }

    pub fn with_shell_allowlist(mut self, commands: Vec<String>) -> Self {
        self.shell_allowlist = commands;
        self
    }

    fn exec_shell(&self, args: &Value) -> Result<Value, AssistantError> {
        let command = args["command"]
            .as_str()
            .ok_or_else(|| AssistantError::Tool("shell_exec: missing 'command' arg".to_string()))?;

        // Check first word against allowlist
        let first_word = command.split_whitespace().next().unwrap_or("");
        if !self.shell_allowlist.iter().any(|a| a == first_word) {
            return Err(AssistantError::Tool(format!(
                "shell_exec: command '{}' not in allowlist",
                first_word
            )));
        }

        let output = Command::new("sh")
            .arg("-c")
            .arg(command)
            .output()
            .map_err(|e| AssistantError::Tool(format!("shell_exec failed: {}", e)))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        Ok(serde_json::json!({
            "exit_code": output.status.code().unwrap_or(-1),
            "stdout": stdout,
            "stderr": stderr,
        }))
    }

    fn exec_read_file(&self, args: &Value) -> Result<Value, AssistantError> {
        let path = args["path"]
            .as_str()
            .ok_or_else(|| AssistantError::Tool("read_file: missing 'path' arg".to_string()))?;

        let content = std::fs::read_to_string(path)
            .map_err(|e| AssistantError::Tool(format!("read_file '{}': {}", path, e)))?;

        Ok(serde_json::json!({
            "path": path,
            "content": content,
        }))
    }

    fn exec_write_file(&self, args: &Value) -> Result<Value, AssistantError> {
        let path = args["path"]
            .as_str()
            .ok_or_else(|| AssistantError::Tool("write_file: missing 'path' arg".to_string()))?;
        let content = args["content"]
            .as_str()
            .ok_or_else(|| {
                AssistantError::Tool("write_file: missing 'content' arg".to_string())
            })?;

        std::fs::write(path, content)
            .map_err(|e| AssistantError::Tool(format!("write_file '{}': {}", path, e)))?;

        Ok(serde_json::json!({
            "path": path,
            "bytes_written": content.len(),
        }))
    }

    fn exec_http_get(&self, args: &Value) -> Result<Value, AssistantError> {
        let url = args["url"]
            .as_str()
            .ok_or_else(|| AssistantError::Tool("http_get: missing 'url' arg".to_string()))?;

        let response = ureq::get(url)
            .call()
            .map_err(|e| AssistantError::Tool(format!("http_get '{}': {}", url, e)))?;

        let status = response.status();
        let body = response
            .into_string()
            .map_err(|e| AssistantError::Tool(format!("http_get read body: {}", e)))?;

        Ok(serde_json::json!({
            "url": url,
            "status": status,
            "body": body,
        }))
    }

    fn exec_current_time(&self, _args: &Value) -> Result<Value, AssistantError> {
        let now = chrono::Utc::now();
        Ok(serde_json::json!({
            "utc": now.to_rfc3339(),
            "unix": now.timestamp(),
        }))
    }

    fn exec_list_files(&self, args: &Value) -> Result<Value, AssistantError> {
        let path = args["path"].as_str().unwrap_or(".");

        let entries: Vec<String> = std::fs::read_dir(path)
            .map_err(|e| AssistantError::Tool(format!("list_files '{}': {}", path, e)))?
            .filter_map(|entry| entry.ok())
            .map(|entry| {
                let name = entry.file_name().to_string_lossy().to_string();
                let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
                if is_dir {
                    format!("{}/", name)
                } else {
                    name
                }
            })
            .collect();

        Ok(serde_json::json!({
            "path": path,
            "entries": entries,
        }))
    }
}

impl Default for BuiltinTools {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolProvider for BuiltinTools {
    fn available_tools(&self) -> Vec<ToolInfo> {
        vec![
            ToolInfo {
                name: "shell_exec".to_string(),
                description: "Run a shell command (allowlisted commands only). Args: {\"command\": \"...\"}".to_string(),
            },
            ToolInfo {
                name: "read_file".to_string(),
                description: "Read a file's contents. Args: {\"path\": \"...\"}".to_string(),
            },
            ToolInfo {
                name: "write_file".to_string(),
                description: "Write content to a file. Args: {\"path\": \"...\", \"content\": \"...\"}".to_string(),
            },
            ToolInfo {
                name: "http_get".to_string(),
                description: "Fetch a URL via HTTP GET. Args: {\"url\": \"...\"}".to_string(),
            },
            ToolInfo {
                name: "current_time".to_string(),
                description: "Get current date and time in UTC. Args: {}".to_string(),
            },
            ToolInfo {
                name: "list_files".to_string(),
                description: "List directory contents. Args: {\"path\": \"...\"}".to_string(),
            },
        ]
    }

    fn invoke(&self, tool_name: &str, args: Value) -> Result<Value, AssistantError> {
        match tool_name {
            "shell_exec" => self.exec_shell(&args),
            "read_file" => self.exec_read_file(&args),
            "write_file" => self.exec_write_file(&args),
            "http_get" => self.exec_http_get(&args),
            "current_time" => self.exec_current_time(&args),
            "list_files" => self.exec_list_files(&args),
            _ => Err(AssistantError::Tool(format!(
                "Unknown tool: {}",
                tool_name
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_current_time() {
        let tools = BuiltinTools::new();
        let result = tools.invoke("current_time", serde_json::json!({})).unwrap();
        assert!(result["utc"].as_str().is_some());
        assert!(result["unix"].as_i64().is_some());
    }

    #[test]
    fn test_read_file() {
        let tools = BuiltinTools::new();
        // Write a temp file and read it back
        let tmp = std::env::temp_dir().join("assistant_test_read.txt");
        std::fs::write(&tmp, "[workspace]\nmembers = []").unwrap();

        let result = tools
            .invoke(
                "read_file",
                serde_json::json!({"path": tmp.to_str().unwrap()}),
            )
            .unwrap();
        let content = result["content"].as_str().unwrap();
        assert!(content.contains("[workspace]"));

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_read_file_not_found() {
        let tools = BuiltinTools::new();
        let result = tools.invoke(
            "read_file",
            serde_json::json!({"path": "/nonexistent/file.txt"}),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_write_and_read_file() {
        let tools = BuiltinTools::new();
        let tmp = std::env::temp_dir().join("assistant_test_write.txt");
        let tmp_path = tmp.to_str().unwrap();

        let write_result = tools
            .invoke(
                "write_file",
                serde_json::json!({"path": tmp_path, "content": "hello from test"}),
            )
            .unwrap();
        assert_eq!(write_result["bytes_written"], 15);

        let read_result = tools
            .invoke("read_file", serde_json::json!({"path": tmp_path}))
            .unwrap();
        assert_eq!(read_result["content"], "hello from test");

        // Cleanup
        let _ = std::fs::remove_file(tmp_path);
    }

    #[test]
    fn test_list_files() {
        let tools = BuiltinTools::new();
        let result = tools
            .invoke("list_files", serde_json::json!({"path": "."}))
            .unwrap();
        let entries = result["entries"].as_array().unwrap();
        assert!(!entries.is_empty());
    }

    #[test]
    fn test_shell_exec_allowed() {
        let tools = BuiltinTools::new();
        let result = tools
            .invoke("shell_exec", serde_json::json!({"command": "echo hello"}))
            .unwrap();
        assert_eq!(result["exit_code"], 0);
        assert!(result["stdout"].as_str().unwrap().contains("hello"));
    }

    #[test]
    fn test_shell_exec_blocked() {
        let tools = BuiltinTools::new();
        let result = tools.invoke(
            "shell_exec",
            serde_json::json!({"command": "rm -rf /"}),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_unknown_tool() {
        let tools = BuiltinTools::new();
        let result = tools.invoke("nonexistent", serde_json::json!({}));
        assert!(result.is_err());
    }

    #[test]
    fn test_available_tools_count() {
        let tools = BuiltinTools::new();
        assert_eq!(tools.available_tools().len(), 6);
    }
}
