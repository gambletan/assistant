use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantConfig {
    pub name: String,
    pub persona: Option<String>,
    pub max_reasoning_steps: usize,
    pub memory_enabled: bool,
    pub knowledge_enabled: bool,
}

impl Default for AssistantConfig {
    fn default() -> Self {
        Self {
            name: "Assistant".to_string(),
            persona: None,
            max_reasoning_steps: 10,
            memory_enabled: true,
            knowledge_enabled: true,
        }
    }
}
