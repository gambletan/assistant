use crate::config::AssistantConfig;
use crate::cortex_memory::SimpleMemoryStore;
use crate::error::AssistantError;
use crate::llm_reasoner::LlmReasoner;
use crate::shell_tools::BuiltinTools;
use crate::simple_knowledge::SimpleKnowledgeStore;
use crate::types::UserInput;
use crate::Assistant;
use std::collections::HashMap;
use std::io::{self, BufRead, Write};

/// Configuration file format for TOML-based setup.
#[derive(Debug, serde::Deserialize)]
struct RunnerConfig {
    name: Option<String>,
    persona: Option<String>,
    llm_url: String,
    llm_key: String,
    model: String,
    db_path: Option<String>,
    system_prompt: Option<String>,
    shell_allowlist: Option<Vec<String>>,
}

/// The main entry point that wires everything together and runs the assistant.
pub struct AssistantRunner {
    assistant: Assistant,
}

impl AssistantRunner {
    /// Build an AssistantRunner from a TOML config file.
    pub fn from_config(config_path: &str) -> Result<Self, AssistantError> {
        let content = std::fs::read_to_string(config_path)
            .map_err(|e| AssistantError::Config(format!("Failed to read config '{}': {}", config_path, e)))?;

        let rc: RunnerConfig = toml::from_str(&content)
            .map_err(|e| AssistantError::Config(format!("Failed to parse config: {}", e)))?;

        let db_path = rc.db_path.unwrap_or_else(|| "assistant.db".to_string());

        let config = AssistantConfig {
            name: rc.name.unwrap_or_else(|| "Assistant".to_string()),
            persona: rc.persona,
            max_reasoning_steps: 10,
            memory_enabled: true,
            knowledge_enabled: true,
        };

        let mut reasoner = LlmReasoner::new(&rc.llm_url, &rc.llm_key, &rc.model);
        if let Some(prompt) = rc.system_prompt {
            reasoner = reasoner.with_system_prompt(prompt);
        }

        let memory = SimpleMemoryStore::new(&db_path)?;
        let knowledge = SimpleKnowledgeStore::new(&db_path)?;

        let mut tools = BuiltinTools::new();
        if let Some(allowlist) = rc.shell_allowlist {
            tools = tools.with_shell_allowlist(allowlist);
        }

        let assistant = Assistant::new(config)
            .with_reasoner(Box::new(reasoner))
            .with_memory(Box::new(memory))
            .with_knowledge(Box::new(knowledge))
            .with_tools(Box::new(tools));

        Ok(Self { assistant })
    }

    /// Build an AssistantRunner from environment variables.
    ///
    /// Required: ASSISTANT_LLM_URL, ASSISTANT_LLM_KEY, ASSISTANT_MODEL
    /// Optional: ASSISTANT_DB_PATH (defaults to "assistant.db")
    pub fn from_env() -> Result<Self, AssistantError> {
        let llm_url = std::env::var("ASSISTANT_LLM_URL")
            .map_err(|_| AssistantError::Config("ASSISTANT_LLM_URL not set".to_string()))?;
        let llm_key = std::env::var("ASSISTANT_LLM_KEY")
            .map_err(|_| AssistantError::Config("ASSISTANT_LLM_KEY not set".to_string()))?;
        let model = std::env::var("ASSISTANT_MODEL")
            .map_err(|_| AssistantError::Config("ASSISTANT_MODEL not set".to_string()))?;
        let db_path =
            std::env::var("ASSISTANT_DB_PATH").unwrap_or_else(|_| "assistant.db".to_string());

        let config = AssistantConfig::default();
        let reasoner = LlmReasoner::new(&llm_url, &llm_key, &model);
        let memory = SimpleMemoryStore::new(&db_path)?;
        let knowledge = SimpleKnowledgeStore::new(&db_path)?;
        let tools = BuiltinTools::new();

        let assistant = Assistant::new(config)
            .with_reasoner(Box::new(reasoner))
            .with_memory(Box::new(memory))
            .with_knowledge(Box::new(knowledge))
            .with_tools(Box::new(tools));

        Ok(Self { assistant })
    }

    /// Simple one-shot: text in, text out.
    pub fn chat(&self, input: &str) -> Result<String, AssistantError> {
        let user_input = UserInput {
            text: input.to_string(),
            user_id: "user".to_string(),
            channel: "api".to_string(),
            metadata: HashMap::new(),
        };

        let response = self.assistant.process(user_input)?;
        Ok(response.text)
    }

    /// REPL loop: read stdin, process, print response, loop.
    pub fn run_interactive(&self) -> Result<(), AssistantError> {
        let stdin = io::stdin();
        let mut stdout = io::stdout();

        println!("Assistant ready. Type 'quit' or 'exit' to stop.\n");

        loop {
            print!("> ");
            stdout.flush().unwrap();

            let mut line = String::new();
            match stdin.lock().read_line(&mut line) {
                Ok(0) => break, // EOF
                Ok(_) => {}
                Err(e) => {
                    eprintln!("Read error: {}", e);
                    break;
                }
            }

            let input = line.trim();
            if input.is_empty() {
                continue;
            }
            if input == "quit" || input == "exit" {
                println!("Goodbye!");
                break;
            }

            match self.chat(input) {
                Ok(response) => println!("\n{}\n", response),
                Err(e) => eprintln!("\nError: {}\n", e),
            }
        }

        Ok(())
    }

    /// Get a reference to the underlying Assistant.
    pub fn assistant(&self) -> &Assistant {
        &self.assistant
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_env_missing_vars() {
        // Unset the vars to make sure this fails properly
        std::env::remove_var("ASSISTANT_LLM_URL");
        let result = AssistantRunner::from_env();
        assert!(result.is_err());
    }

    #[test]
    fn test_from_config_missing_file() {
        let result = AssistantRunner::from_config("/nonexistent/config.toml");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_config_valid() {
        let tmp = std::env::temp_dir().join("assistant_test_config.toml");
        let config = r#"
llm_url = "http://localhost:8080"
llm_key = "test-key"
model = "test-model"
db_path = ":memory:"
name = "TestBot"
"#;
        std::fs::write(&tmp, config).unwrap();

        let result = AssistantRunner::from_config(tmp.to_str().unwrap());
        assert!(result.is_ok());

        let _ = std::fs::remove_file(&tmp);
    }
}
