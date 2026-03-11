use crate::error::AssistantError;
use crate::providers::ReasoningResult;
use crate::types::{ActionRecord, AssistantResponse, Context, UserInput};
use log::{debug, info};
use std::time::Instant;

/// The processing pipeline that orchestrates input through all providers.
pub struct Pipeline;

impl Pipeline {
    /// Process a user input through the full assistant pipeline.
    ///
    /// Steps:
    /// 1. Recall memories relevant to input
    /// 2. Query knowledge base
    /// 3. Build context
    /// 4. Reason (may loop: reason -> tool call -> observe -> reason)
    /// 5. Store new memories from conversation
    /// 6. Return response
    pub fn process(
        assistant: &crate::Assistant,
        input: &UserInput,
    ) -> Result<AssistantResponse, AssistantError> {
        info!("Processing input from user={} channel={}", input.user_id, input.channel);

        let mut context = Context::new();
        let mut actions_taken = Vec::new();
        let mut memories_stored = 0usize;
        let mut sources = Vec::new();

        // Step 1: Recall memories
        if assistant.config.memory_enabled {
            if let Some(ref memory) = assistant.memory {
                debug!("Recalling memories for: {}", &input.text);
                match memory.recall(&input.text, 5) {
                    Ok(memories) => context.memories = memories,
                    Err(e) => debug!("Memory recall failed (non-fatal): {}", e),
                }
            }
        }

        // Step 2: Query knowledge base
        if assistant.config.knowledge_enabled {
            if let Some(ref kb) = assistant.knowledge {
                debug!("Querying knowledge base for: {}", &input.text);
                match kb.query(&input.text, 5) {
                    Ok(results) => {
                        context.knowledge = results
                            .iter()
                            .map(|s| format!("[{}] {}", s.title, s.chunk_preview))
                            .collect();
                        sources = results;
                    }
                    Err(e) => debug!("Knowledge query failed (non-fatal): {}", e),
                }
            }
        }

        // Step 3: Build context (conversation history would be added here)

        // Step 4: Reasoning loop
        let tools_list = assistant
            .tools
            .as_ref()
            .map(|t| t.available_tools())
            .unwrap_or_default();

        let reasoner = assistant
            .reasoner
            .as_ref()
            .ok_or_else(|| AssistantError::Reasoning("no reasoner configured".to_string()))?;

        let response_text;
        let mut steps = 0;

        loop {
            if steps >= assistant.config.max_reasoning_steps {
                return Err(AssistantError::Reasoning(format!(
                    "exceeded max reasoning steps ({})",
                    assistant.config.max_reasoning_steps
                )));
            }

            let result = reasoner.reason(&context, &input.text, &tools_list)?;
            steps += 1;

            match result {
                ReasoningResult::Respond(text) | ReasoningResult::Done(text) => {
                    response_text = text;
                    break;
                }
                ReasoningResult::CallTool { name, args } => {
                    let start = Instant::now();
                    let tool_result = if let Some(ref tools) = assistant.tools {
                        tools.invoke(&name, args.clone())?
                    } else {
                        return Err(AssistantError::Tool(
                            "no tool provider configured".to_string(),
                        ));
                    };
                    let duration_ms = start.elapsed().as_millis() as u64;

                    actions_taken.push(ActionRecord {
                        tool_name: name.clone(),
                        args_summary: args.to_string(),
                        result_summary: tool_result.to_string(),
                        duration_ms,
                    });

                    // Feed tool result back into context for next reasoning step
                    context
                        .conversation_history
                        .push((format!("[tool:{}]", name), tool_result.to_string()));
                }
            }
        }

        // Step 5: Store new memories
        if assistant.config.memory_enabled {
            if let Some(ref memory) = assistant.memory {
                let memory_text = format!("User: {}\nAssistant: {}", input.text, response_text);
                match memory.store(&memory_text, &input.channel) {
                    Ok(()) => memories_stored = 1,
                    Err(e) => debug!("Memory store failed (non-fatal): {}", e),
                }
            }
        }

        Ok(AssistantResponse {
            text: response_text,
            actions_taken,
            memories_stored,
            sources,
        })
    }
}
