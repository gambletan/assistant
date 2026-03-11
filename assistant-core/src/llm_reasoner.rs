use crate::error::AssistantError;
use crate::providers::{ReasonerProvider, ReasoningResult, ToolInfo};
use crate::types::Context;

/// A ReasonerProvider that calls an OpenAI-compatible chat completions API.
pub struct LlmReasoner {
    base_url: String,
    api_key: String,
    model: String,
    system_prompt: String,
}

impl LlmReasoner {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            model: model.into(),
            system_prompt: "You are a helpful assistant. When you need to use a tool, respond with a JSON object like {\"tool_call\": {\"name\": \"tool_name\", \"args\": {...}}}. Otherwise, respond with plain text.".to_string(),
        }
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    fn build_messages(
        &self,
        context: &Context,
        input: &str,
        tools: &[ToolInfo],
    ) -> Vec<serde_json::Value> {
        let mut messages = Vec::new();

        // System prompt with context
        let mut system = self.system_prompt.clone();

        if !context.memories.is_empty() {
            system.push_str("\n\n## Relevant Memories\n");
            for m in &context.memories {
                system.push_str(&format!("- {}\n", m));
            }
        }

        if !context.knowledge.is_empty() {
            system.push_str("\n\n## Relevant Knowledge\n");
            for k in &context.knowledge {
                system.push_str(&format!("- {}\n", k));
            }
        }

        if !tools.is_empty() {
            system.push_str("\n\n## Available Tools\n");
            for t in tools {
                system.push_str(&format!("- `{}`: {}\n", t.name, t.description));
            }
        }

        messages.push(serde_json::json!({
            "role": "system",
            "content": system,
        }));

        // Conversation history
        for (role, content) in &context.conversation_history {
            let api_role = if role.starts_with("[tool:") {
                "assistant"
            } else {
                "user"
            };
            messages.push(serde_json::json!({
                "role": api_role,
                "content": content,
            }));
        }

        // Current user input
        messages.push(serde_json::json!({
            "role": "user",
            "content": input,
        }));

        messages
    }
}

impl ReasonerProvider for LlmReasoner {
    fn reason(
        &self,
        context: &Context,
        input: &str,
        tools: &[ToolInfo],
    ) -> Result<ReasoningResult, AssistantError> {
        let messages = self.build_messages(context, input, tools);

        let url = format!("{}/v1/chat/completions", self.base_url.trim_end_matches('/'));

        let body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
        });

        let response = ureq::post(&url)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", "application/json")
            .send_json(body)
            .map_err(|e| AssistantError::Reasoning(format!("HTTP request failed: {}", e)))?;

        let resp_json: serde_json::Value = response
            .into_json()
            .map_err(|e| AssistantError::Reasoning(format!("Failed to parse response: {}", e)))?;

        let content = resp_json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        // Check if the response contains a tool call JSON
        if let Some(tool_call) = Self::parse_tool_call(&content) {
            return Ok(tool_call);
        }

        Ok(ReasoningResult::Respond(content))
    }
}

impl LlmReasoner {
    fn parse_tool_call(content: &str) -> Option<ReasoningResult> {
        // Try to find a JSON object with tool_call
        let trimmed = content.trim();

        // Try parsing the whole content as JSON
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
            if let Some(tc) = v.get("tool_call") {
                if let (Some(name), Some(args)) = (tc.get("name"), tc.get("args")) {
                    return Some(ReasoningResult::CallTool {
                        name: name.as_str().unwrap_or_default().to_string(),
                        args: args.clone(),
                    });
                }
            }
        }

        // Try to find JSON embedded in text (between { and })
        if let Some(start) = trimmed.find("{\"tool_call\"") {
            if let Some(end) = trimmed[start..].rfind('}') {
                let json_str = &trimmed[start..=start + end];
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(json_str) {
                    if let Some(tc) = v.get("tool_call") {
                        if let (Some(name), Some(args)) = (tc.get("name"), tc.get("args")) {
                            return Some(ReasoningResult::CallTool {
                                name: name.as_str().unwrap_or_default().to_string(),
                                args: args.clone(),
                            });
                        }
                    }
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tool_call_plain_text() {
        assert!(LlmReasoner::parse_tool_call("Hello, how can I help?").is_none());
    }

    #[test]
    fn test_parse_tool_call_json() {
        let content = r#"{"tool_call": {"name": "shell_exec", "args": {"command": "ls"}}}"#;
        let result = LlmReasoner::parse_tool_call(content);
        assert!(result.is_some());
        if let Some(ReasoningResult::CallTool { name, args }) = result {
            assert_eq!(name, "shell_exec");
            assert_eq!(args["command"], "ls");
        }
    }

    #[test]
    fn test_parse_tool_call_embedded() {
        let content = r#"I'll run that for you. {"tool_call": {"name": "read_file", "args": {"path": "/tmp/test.txt"}}}"#;
        let result = LlmReasoner::parse_tool_call(content);
        assert!(result.is_some());
    }

    #[test]
    fn test_build_messages_includes_context() {
        let reasoner = LlmReasoner::new("http://localhost", "key", "model");
        let mut ctx = Context::new();
        ctx.memories.push("previous conversation".to_string());
        ctx.knowledge.push("[Doc] some info".to_string());
        let tools = vec![ToolInfo {
            name: "test".to_string(),
            description: "a test tool".to_string(),
        }];
        let messages = reasoner.build_messages(&ctx, "hello", &tools);
        assert_eq!(messages.len(), 2); // system + user
        let system = messages[0]["content"].as_str().unwrap();
        assert!(system.contains("previous conversation"));
        assert!(system.contains("some info"));
        assert!(system.contains("`test`"));
    }
}
