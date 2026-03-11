pub mod comfyui;
pub mod config;
pub mod cortex_memory;
pub mod error;
pub mod llm_reasoner;
pub mod pipeline;
pub mod providers;
pub mod runner;
pub mod shell_tools;
pub mod simple_knowledge;
pub mod types;

use config::AssistantConfig;
use error::AssistantError;
use pipeline::Pipeline;
use providers::{ChannelProvider, KnowledgeProvider, MemoryProvider, ReasonerProvider, ToolProvider};
use types::{AssistantResponse, UserInput};

/// The main assistant runtime — wires together all provider modules.
pub struct Assistant {
    pub config: AssistantConfig,
    pub(crate) memory: Option<Box<dyn MemoryProvider>>,
    pub(crate) knowledge: Option<Box<dyn KnowledgeProvider>>,
    pub(crate) tools: Option<Box<dyn ToolProvider>>,
    pub(crate) reasoner: Option<Box<dyn ReasonerProvider>>,
    pub(crate) channels: Vec<Box<dyn ChannelProvider>>,
}

impl Assistant {
    pub fn new(config: AssistantConfig) -> Self {
        Self {
            config,
            memory: None,
            knowledge: None,
            tools: None,
            reasoner: None,
            channels: Vec::new(),
        }
    }

    pub fn with_memory(mut self, memory: Box<dyn MemoryProvider>) -> Self {
        self.memory = Some(memory);
        self
    }

    pub fn with_knowledge(mut self, kb: Box<dyn KnowledgeProvider>) -> Self {
        self.knowledge = Some(kb);
        self
    }

    pub fn with_tools(mut self, registry: Box<dyn ToolProvider>) -> Self {
        self.tools = Some(registry);
        self
    }

    pub fn with_reasoner(mut self, reasoner: Box<dyn ReasonerProvider>) -> Self {
        self.reasoner = Some(reasoner);
        self
    }

    pub fn with_channel(mut self, channel: Box<dyn ChannelProvider>) -> Self {
        self.channels.push(channel);
        self
    }

    /// Process a single user input through the full pipeline.
    pub fn process(&self, input: UserInput) -> Result<AssistantResponse, AssistantError> {
        Pipeline::process(self, &input)
    }

    /// Start listening on all registered channels (placeholder).
    pub fn start(&self) -> Result<(), AssistantError> {
        if self.channels.is_empty() {
            return Err(AssistantError::Channel(
                "no channels registered".to_string(),
            ));
        }
        log::info!(
            "Assistant '{}' starting with {} channel(s)",
            self.config.name,
            self.channels.len()
        );
        // Future: spawn channel receive loops
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::*;
    use crate::types::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    // --- Mock implementations ---

    struct MockMemory {
        store: Mutex<Vec<String>>,
    }

    impl MockMemory {
        fn new() -> Self {
            Self {
                store: Mutex::new(Vec::new()),
            }
        }
    }

    impl MemoryProvider for MockMemory {
        fn store(&self, text: &str, _source: &str) -> Result<(), AssistantError> {
            self.store.lock().unwrap().push(text.to_string());
            Ok(())
        }

        fn recall(&self, query: &str, limit: usize) -> Result<Vec<String>, AssistantError> {
            let store = self.store.lock().unwrap();
            Ok(store
                .iter()
                .filter(|s| s.contains(query) || !query.is_empty())
                .take(limit)
                .cloned()
                .collect())
        }
    }

    struct MockKnowledge {
        docs: Vec<(String, String)>,
    }

    impl MockKnowledge {
        fn new(docs: Vec<(String, String)>) -> Self {
            Self { docs }
        }
    }

    impl KnowledgeProvider for MockKnowledge {
        fn query(&self, _text: &str, top_k: usize) -> Result<Vec<Source>, AssistantError> {
            Ok(self
                .docs
                .iter()
                .take(top_k)
                .map(|(title, content)| Source {
                    title: title.clone(),
                    chunk_preview: content.clone(),
                    relevance_score: 0.9,
                })
                .collect())
        }

        fn ingest(&self, _title: &str, _content: &str) -> Result<(), AssistantError> {
            Ok(())
        }
    }

    struct MockTools;

    impl ToolProvider for MockTools {
        fn available_tools(&self) -> Vec<ToolInfo> {
            vec![ToolInfo {
                name: "echo".to_string(),
                description: "Echoes input back".to_string(),
            }]
        }

        fn invoke(
            &self,
            _tool_name: &str,
            args: serde_json::Value,
        ) -> Result<serde_json::Value, AssistantError> {
            // Echo tool: return the args as the result
            Ok(args)
        }
    }

    struct MockReasoner {
        /// If true, issues a tool call first, then responds.
        use_tool: bool,
        call_count: Mutex<usize>,
    }

    impl MockReasoner {
        fn simple() -> Self {
            Self {
                use_tool: false,
                call_count: Mutex::new(0),
            }
        }

        fn with_tool_call() -> Self {
            Self {
                use_tool: true,
                call_count: Mutex::new(0),
            }
        }
    }

    impl ReasonerProvider for MockReasoner {
        fn reason(
            &self,
            _context: &Context,
            input: &str,
            _tools: &[ToolInfo],
        ) -> Result<ReasoningResult, AssistantError> {
            let mut count = self.call_count.lock().unwrap();
            *count += 1;

            if self.use_tool && *count == 1 {
                Ok(ReasoningResult::CallTool {
                    name: "echo".to_string(),
                    args: serde_json::json!({"input": input}),
                })
            } else {
                Ok(ReasoningResult::Respond(format!(
                    "Processed: {}",
                    input
                )))
            }
        }
    }

    struct MockChannel {
        channel_name: String,
    }

    impl ChannelProvider for MockChannel {
        fn name(&self) -> &str {
            &self.channel_name
        }
    }

    fn test_input(text: &str) -> UserInput {
        UserInput {
            text: text.to_string(),
            user_id: "test-user".to_string(),
            channel: "test".to_string(),
            metadata: HashMap::new(),
        }
    }

    // --- Tests ---

    #[test]
    fn test_full_pipeline_with_mocks() {
        let assistant = Assistant::new(AssistantConfig::default())
            .with_memory(Box::new(MockMemory::new()))
            .with_knowledge(Box::new(MockKnowledge::new(vec![(
                "Rust".to_string(),
                "A systems programming language".to_string(),
            )])))
            .with_tools(Box::new(MockTools))
            .with_reasoner(Box::new(MockReasoner::simple()));

        let response = assistant.process(test_input("Hello")).unwrap();

        assert_eq!(response.text, "Processed: Hello");
        assert!(response.actions_taken.is_empty());
        assert_eq!(response.memories_stored, 1);
        assert_eq!(response.sources.len(), 1);
        assert_eq!(response.sources[0].title, "Rust");
    }

    #[test]
    fn test_pipeline_with_tool_call() {
        let assistant = Assistant::new(AssistantConfig::default())
            .with_memory(Box::new(MockMemory::new()))
            .with_tools(Box::new(MockTools))
            .with_reasoner(Box::new(MockReasoner::with_tool_call()));

        let response = assistant.process(test_input("echo this")).unwrap();

        assert_eq!(response.text, "Processed: echo this");
        assert_eq!(response.actions_taken.len(), 1);
        assert_eq!(response.actions_taken[0].tool_name, "echo");
    }

    #[test]
    fn test_memory_storage_after_conversation() {
        let assistant = Assistant::new(AssistantConfig::default())
            .with_memory(Box::new(MockMemory::new()))
            .with_reasoner(Box::new(MockReasoner::simple()));

        let response = assistant.process(test_input("remember this")).unwrap();
        assert_eq!(response.memories_stored, 1);
    }

    #[test]
    fn test_knowledge_retrieval_in_context() {
        let docs = vec![
            ("Doc A".to_string(), "First document".to_string()),
            ("Doc B".to_string(), "Second document".to_string()),
        ];

        let assistant = Assistant::new(AssistantConfig::default())
            .with_knowledge(Box::new(MockKnowledge::new(docs)))
            .with_reasoner(Box::new(MockReasoner::simple()));

        let response = assistant.process(test_input("search")).unwrap();

        assert_eq!(response.sources.len(), 2);
        assert_eq!(response.sources[0].title, "Doc A");
        assert_eq!(response.sources[1].title, "Doc B");
    }

    #[test]
    fn test_no_reasoner_returns_error() {
        let assistant = Assistant::new(AssistantConfig::default());

        let result = assistant.process(test_input("hello"));
        assert!(result.is_err());
    }

    #[test]
    fn test_start_without_channels_returns_error() {
        let assistant = Assistant::new(AssistantConfig::default());

        let result = assistant.start();
        assert!(result.is_err());
    }

    #[test]
    fn test_start_with_channel() {
        let assistant = Assistant::new(AssistantConfig::default()).with_channel(Box::new(
            MockChannel {
                channel_name: "telegram".to_string(),
            },
        ));

        let result = assistant.start();
        assert!(result.is_ok());
    }

    #[test]
    fn test_disabled_memory() {
        let config = AssistantConfig {
            memory_enabled: false,
            ..AssistantConfig::default()
        };

        let assistant = Assistant::new(config)
            .with_memory(Box::new(MockMemory::new()))
            .with_reasoner(Box::new(MockReasoner::simple()));

        let response = assistant.process(test_input("hello")).unwrap();
        assert_eq!(response.memories_stored, 0);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::cortex_memory::SimpleMemoryStore;
    use crate::providers::*;
    use crate::shell_tools::BuiltinTools;
    use crate::simple_knowledge::SimpleKnowledgeStore;
    use crate::types::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// A mock reasoner for integration tests that returns a plain response.
    struct IntegrationMockReasoner {
        use_tool: bool,
        call_count: Mutex<usize>,
    }

    impl IntegrationMockReasoner {
        fn simple() -> Self {
            Self {
                use_tool: false,
                call_count: Mutex::new(0),
            }
        }

        #[allow(dead_code)]
        fn with_tool(name: &str) -> (Self, String) {
            (
                Self {
                    use_tool: true,
                    call_count: Mutex::new(0),
                },
                name.to_string(),
            )
        }
    }

    impl ReasonerProvider for IntegrationMockReasoner {
        fn reason(
            &self,
            _context: &Context,
            input: &str,
            _tools: &[ToolInfo],
        ) -> Result<ReasoningResult, AssistantError> {
            let mut count = self.call_count.lock().unwrap();
            *count += 1;

            if self.use_tool && *count == 1 {
                Ok(ReasoningResult::CallTool {
                    name: "current_time".to_string(),
                    args: serde_json::json!({}),
                })
            } else {
                Ok(ReasoningResult::Respond(format!("Response to: {}", input)))
            }
        }
    }

    fn make_input(text: &str) -> UserInput {
        UserInput {
            text: text.to_string(),
            user_id: "integration-test".to_string(),
            channel: "test".to_string(),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_simple_memory_store_integration() {
        let store = SimpleMemoryStore::new(":memory:").unwrap();

        // Store several memories
        store.store("I like Rust programming", "chat").unwrap();
        store.store("My favorite color is blue", "chat").unwrap();
        store.store("I work on distributed systems", "chat").unwrap();

        // Recall by keyword
        let results = store.recall("Rust", 5).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].contains("Rust"));

        // Recall with broad match
        let results = store.recall("I", 10).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_simple_knowledge_store_integration() {
        let store = SimpleKnowledgeStore::new(":memory:").unwrap();

        store
            .ingest(
                "Rust Ownership",
                "Rust uses ownership rules to manage memory.\n\nEach value has exactly one owner.\n\nWhen the owner goes out of scope, the value is dropped.",
            )
            .unwrap();

        store
            .ingest(
                "Rust Borrowing",
                "References allow borrowing without taking ownership.\n\nMutable references are exclusive.",
            )
            .unwrap();

        // Query for ownership
        let results = store.query("ownership", 5).unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().any(|s| s.title == "Rust Borrowing"));

        // Query for scope
        let results = store.query("scope", 5).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].title, "Rust Ownership");
    }

    #[test]
    fn test_builtin_tools_current_time() {
        let tools = BuiltinTools::new();
        let result = tools.invoke("current_time", serde_json::json!({})).unwrap();
        assert!(result["utc"].as_str().unwrap().contains("T"));
        assert!(result["unix"].as_i64().unwrap() > 0);
    }

    #[test]
    fn test_builtin_tools_read_file() {
        let tools = BuiltinTools::new();

        // Write a temp file, then read it
        let tmp = std::env::temp_dir().join("assistant_integration_test.txt");
        std::fs::write(&tmp, "integration test content").unwrap();

        let result = tools
            .invoke(
                "read_file",
                serde_json::json!({"path": tmp.to_str().unwrap()}),
            )
            .unwrap();
        assert_eq!(result["content"], "integration test content");

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_full_pipeline_with_real_providers() {
        let memory = SimpleMemoryStore::new(":memory:").unwrap();
        let knowledge = SimpleKnowledgeStore::new(":memory:").unwrap();
        let tools = BuiltinTools::new();

        // Pre-populate knowledge
        knowledge
            .ingest(
                "Project Info",
                "The assistant project is built in Rust.\n\nIt uses a modular provider architecture.",
            )
            .unwrap();

        // Pre-populate memory
        memory
            .store("User previously asked about Rust", "chat")
            .unwrap();

        let assistant = Assistant::new(AssistantConfig::default())
            .with_memory(Box::new(memory))
            .with_knowledge(Box::new(knowledge))
            .with_tools(Box::new(tools))
            .with_reasoner(Box::new(IntegrationMockReasoner::simple()));

        let response = assistant.process(make_input("Rust")).unwrap();

        assert!(response.text.contains("Response to:"));
        assert_eq!(response.memories_stored, 1);
        // Knowledge query for "Rust" matches chunks containing "Rust"
        assert!(!response.sources.is_empty());
    }

    #[test]
    fn test_full_pipeline_with_tool_call() {
        let memory = SimpleMemoryStore::new(":memory:").unwrap();
        let tools = BuiltinTools::new();

        let reasoner = IntegrationMockReasoner {
            use_tool: true,
            call_count: Mutex::new(0),
        };

        let assistant = Assistant::new(AssistantConfig::default())
            .with_memory(Box::new(memory))
            .with_tools(Box::new(tools))
            .with_reasoner(Box::new(reasoner));

        let response = assistant
            .process(make_input("What time is it?"))
            .unwrap();

        assert!(!response.text.is_empty());
        assert_eq!(response.actions_taken.len(), 1);
        assert_eq!(response.actions_taken[0].tool_name, "current_time");
        // The tool result should contain a timestamp
        assert!(response.actions_taken[0].result_summary.contains("utc"));
    }
}
