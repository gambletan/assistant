pub mod config;
pub mod error;
pub mod pipeline;
pub mod providers;
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
