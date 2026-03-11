use crate::error::AssistantError;
use crate::types::{Context, Source};
use serde::{Deserialize, Serialize};

/// Long-term memory storage and retrieval (wraps cortex).
pub trait MemoryProvider: Send + Sync {
    fn store(&self, text: &str, source: &str) -> Result<(), AssistantError>;
    fn recall(&self, query: &str, limit: usize) -> Result<Vec<String>, AssistantError>;
}

/// Knowledge base for RAG retrieval (wraps knowledge-base).
pub trait KnowledgeProvider: Send + Sync {
    fn query(&self, text: &str, top_k: usize) -> Result<Vec<Source>, AssistantError>;
    fn ingest(&self, title: &str, content: &str) -> Result<(), AssistantError>;
}

/// Tool registry for executing actions (wraps tool-registry).
pub trait ToolProvider: Send + Sync {
    fn available_tools(&self) -> Vec<ToolInfo>;
    fn invoke(
        &self,
        tool_name: &str,
        args: serde_json::Value,
    ) -> Result<serde_json::Value, AssistantError>;
}

/// Reasoning engine (wraps agent-engine).
pub trait ReasonerProvider: Send + Sync {
    fn reason(
        &self,
        context: &Context,
        input: &str,
        tools: &[ToolInfo],
    ) -> Result<ReasoningResult, AssistantError>;
}

/// Communication channel (wraps unified-channel).
pub trait ChannelProvider: Send + Sync {
    fn name(&self) -> &str;
    // Future: receive/send message loops
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInfo {
    pub name: String,
    pub description: String,
}

#[derive(Debug, Clone)]
pub enum ReasoningResult {
    Respond(String),
    CallTool { name: String, args: serde_json::Value },
    Done(String),
}
