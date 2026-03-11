use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInput {
    pub text: String,
    pub user_id: String,
    pub channel: String,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantResponse {
    pub text: String,
    pub actions_taken: Vec<ActionRecord>,
    pub memories_stored: usize,
    pub sources: Vec<Source>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionRecord {
    pub tool_name: String,
    pub args_summary: String,
    pub result_summary: String,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub title: String,
    pub chunk_preview: String,
    pub relevance_score: f64,
}

#[derive(Debug, Clone)]
pub struct Context {
    pub memories: Vec<String>,
    pub knowledge: Vec<String>,
    pub conversation_history: Vec<(String, String)>,
}

impl Context {
    pub fn new() -> Self {
        Self {
            memories: Vec::new(),
            knowledge: Vec::new(),
            conversation_history: Vec::new(),
        }
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}
