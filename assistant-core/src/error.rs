use thiserror::Error;

#[derive(Error, Debug)]
pub enum AssistantError {
    #[error("memory error: {0}")]
    Memory(String),

    #[error("knowledge error: {0}")]
    Knowledge(String),

    #[error("tool error: {0}")]
    Tool(String),

    #[error("reasoning error: {0}")]
    Reasoning(String),

    #[error("channel error: {0}")]
    Channel(String),

    #[error("config error: {0}")]
    Config(String),
}
