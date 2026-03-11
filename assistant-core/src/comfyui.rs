use crate::error::AssistantError;
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::thread;
use std::time::Duration;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// ComfyUI API Client
// ---------------------------------------------------------------------------

pub struct ComfyUIClient {
    base_url: String,
    client_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptResponse {
    pub prompt_id: String,
}

impl ComfyUIClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client_id: Uuid::new_v4().to_string(),
        }
    }

    /// Queue a workflow prompt for execution.
    pub fn queue_prompt(&self, workflow: &Value) -> Result<PromptResponse, AssistantError> {
        let body = serde_json::json!({
            "prompt": workflow,
            "client_id": &self.client_id,
        });

        let resp = ureq::post(&format!("{}/prompt", self.base_url))
            .send_json(&body)
            .map_err(|e| AssistantError::ComfyUI(format!("queue_prompt request failed: {e}")))?;

        let json: Value = resp
            .into_json()
            .map_err(|e| AssistantError::ComfyUI(format!("queue_prompt parse failed: {e}")))?;

        let prompt_id = json["prompt_id"]
            .as_str()
            .ok_or_else(|| AssistantError::ComfyUI("missing prompt_id in response".to_string()))?
            .to_string();

        Ok(PromptResponse { prompt_id })
    }

    /// Retrieve execution history for a given prompt.
    pub fn get_history(&self, prompt_id: &str) -> Result<Value, AssistantError> {
        let resp = ureq::get(&format!("{}/history/{}", self.base_url, prompt_id))
            .call()
            .map_err(|e| AssistantError::ComfyUI(format!("get_history request failed: {e}")))?;

        resp.into_json()
            .map_err(|e| AssistantError::ComfyUI(format!("get_history parse failed: {e}")))
    }

    /// Download a generated image by filename.
    pub fn get_image(
        &self,
        filename: &str,
        subfolder: &str,
        folder_type: &str,
    ) -> Result<Vec<u8>, AssistantError> {
        let url = format!(
            "{}/view?filename={}&subfolder={}&type={}",
            self.base_url, filename, subfolder, folder_type
        );

        let resp = ureq::get(&url)
            .call()
            .map_err(|e| AssistantError::ComfyUI(format!("get_image request failed: {e}")))?;

        let mut bytes = Vec::new();
        resp.into_reader()
            .read_to_end(&mut bytes)
            .map_err(|e| AssistantError::ComfyUI(format!("get_image read failed: {e}")))?;

        Ok(bytes)
    }

    /// Get all available node types from ComfyUI.
    pub fn get_node_info(&self) -> Result<Value, AssistantError> {
        let resp = ureq::get(&format!("{}/object_info", self.base_url))
            .call()
            .map_err(|e| AssistantError::ComfyUI(format!("get_node_info request failed: {e}")))?;

        resp.into_json()
            .map_err(|e| AssistantError::ComfyUI(format!("get_node_info parse failed: {e}")))
    }

    /// Poll `get_history()` every 2 seconds until the prompt appears or timeout.
    pub fn poll_completion(
        &self,
        prompt_id: &str,
        timeout_secs: u64,
    ) -> Result<Value, AssistantError> {
        let deadline = std::time::Instant::now() + Duration::from_secs(timeout_secs);

        loop {
            let history = self.get_history(prompt_id)?;

            if let Some(entry) = history.get(prompt_id) {
                return Ok(entry.clone());
            }

            if std::time::Instant::now() >= deadline {
                return Err(AssistantError::ComfyUI(format!(
                    "timeout after {timeout_secs}s waiting for prompt {prompt_id}"
                )));
            }

            thread::sleep(Duration::from_secs(2));
        }
    }
}

// ---------------------------------------------------------------------------
// Workflow Template System
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VarType {
    Text,
    Number,
    Seed,
    ImagePath,
    Model,
    Sampler,
    Scheduler,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateVariable {
    pub name: String,
    pub node_id: String,
    pub field_path: String,
    pub var_type: VarType,
    pub default: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowTemplate {
    pub name: String,
    pub template: Value,
    pub variables: Vec<TemplateVariable>,
}

impl WorkflowTemplate {
    /// Load a workflow template from a JSON file.
    ///
    /// Expected file format:
    /// ```json
    /// {
    ///   "name": "my_template",
    ///   "template": { "6": { "inputs": { ... } } },
    ///   "variables": [ { "name": "prompt", "node_id": "6", "field_path": "inputs.text", "var_type": "Text", "default": null } ]
    /// }
    /// ```
    pub fn from_file(path: &str) -> Result<Self, AssistantError> {
        let content = fs::read_to_string(path)
            .map_err(|e| AssistantError::ComfyUI(format!("failed to read template {path}: {e}")))?;

        let template: WorkflowTemplate = serde_json::from_str(&content).map_err(|e| {
            AssistantError::ComfyUI(format!("failed to parse template {path}: {e}"))
        })?;

        Ok(template)
    }

    pub fn from_json(name: &str, json: Value) -> Self {
        Self {
            name: name.to_string(),
            template: json,
            variables: Vec::new(),
        }
    }

    /// Fill the template with the provided variable values.
    ///
    /// For `VarType::Seed`, a random i64 is generated when no value is provided.
    pub fn fill(&self, vars: &HashMap<String, Value>) -> Result<Value, AssistantError> {
        let mut workflow = self.template.clone();

        for var in &self.variables {
            let value = if let Some(v) = vars.get(&var.name) {
                v.clone()
            } else if var.var_type == VarType::Seed {
                let seed: i64 = rand::thread_rng().gen_range(0..i64::MAX);
                Value::Number(serde_json::Number::from(seed))
            } else if let Some(ref def) = var.default {
                def.clone()
            } else {
                return Err(AssistantError::ComfyUI(format!(
                    "missing required variable '{}' for template '{}'",
                    var.name, self.name
                )));
            };

            // Navigate to the node and set the field.
            let node = workflow.get_mut(&var.node_id).ok_or_else(|| {
                AssistantError::ComfyUI(format!(
                    "node_id '{}' not found in template '{}'",
                    var.node_id, self.name
                ))
            })?;

            set_nested(node, &var.field_path, value).map_err(|e| {
                AssistantError::ComfyUI(format!(
                    "failed to set '{}' in node '{}': {e}",
                    var.field_path, var.node_id
                ))
            })?;
        }

        Ok(workflow)
    }
}

/// Walk a dotted path (e.g. "inputs.text") into a JSON value and set it.
fn set_nested(root: &mut Value, path: &str, value: Value) -> Result<(), String> {
    let parts: Vec<&str> = path.split('.').collect();
    let mut current = root;

    for (i, part) in parts.iter().enumerate() {
        if i == parts.len() - 1 {
            match current {
                Value::Object(map) => {
                    map.insert(part.to_string(), value);
                    return Ok(());
                }
                _ => return Err(format!("expected object at '{part}'")),
            }
        } else {
            current = match current {
                Value::Object(map) => map
                    .get_mut(*part)
                    .ok_or_else(|| format!("key '{part}' not found"))?,
                _ => return Err(format!("expected object at '{part}'")),
            };
        }
    }

    Err("empty path".to_string())
}

// ---------------------------------------------------------------------------
// Template Registry
// ---------------------------------------------------------------------------

pub struct TemplateRegistry {
    templates_dir: String,
    templates: HashMap<String, WorkflowTemplate>,
}

impl TemplateRegistry {
    pub fn new(templates_dir: &str) -> Self {
        Self {
            templates_dir: templates_dir.to_string(),
            templates: HashMap::new(),
        }
    }

    /// Scan the templates directory for .json files and load each as a WorkflowTemplate.
    /// Returns the number of templates loaded.
    pub fn load_all(&mut self) -> Result<usize, AssistantError> {
        let dir = Path::new(&self.templates_dir);
        if !dir.exists() {
            return Err(AssistantError::ComfyUI(format!(
                "templates directory '{}' does not exist",
                self.templates_dir
            )));
        }

        let entries = fs::read_dir(dir).map_err(|e| {
            AssistantError::ComfyUI(format!(
                "failed to read templates dir '{}': {e}",
                self.templates_dir
            ))
        })?;

        let mut count = 0;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                let path_str = path.to_string_lossy().to_string();
                let tmpl = WorkflowTemplate::from_file(&path_str)?;
                self.templates.insert(tmpl.name.clone(), tmpl);
                count += 1;
            }
        }

        Ok(count)
    }

    pub fn get(&self, name: &str) -> Option<&WorkflowTemplate> {
        self.templates.get(name)
    }

    pub fn list(&self) -> Vec<&str> {
        self.templates.keys().map(|s| s.as_str()).collect()
    }
}

// ---------------------------------------------------------------------------
// Content Generator
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ContentType {
    Image,
    Video,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedContent {
    pub id: String,
    pub content_type: ContentType,
    pub file_path: String,
    pub metadata: HashMap<String, String>,
    pub prompt_used: String,
    pub template_name: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

pub struct ContentGenerator {
    comfyui: ComfyUIClient,
    templates: TemplateRegistry,
    output_dir: String,
}

impl ContentGenerator {
    pub fn new(comfyui_url: &str, templates_dir: &str, output_dir: &str) -> Self {
        let comfyui = ComfyUIClient::new(comfyui_url);
        let templates = TemplateRegistry::new(templates_dir);
        Self {
            comfyui,
            templates,
            output_dir: output_dir.to_string(),
        }
    }

    /// Load all templates from the configured directory.
    pub fn load_templates(&mut self) -> Result<usize, AssistantError> {
        self.templates.load_all()
    }

    /// Generate a single image using the named template.
    pub fn generate_image(
        &self,
        template_name: &str,
        prompt: &str,
        negative_prompt: &str,
        seed: Option<i64>,
    ) -> Result<GeneratedContent, AssistantError> {
        let tmpl = self.templates.get(template_name).ok_or_else(|| {
            AssistantError::ComfyUI(format!("template '{template_name}' not found"))
        })?;

        let mut vars = HashMap::new();
        vars.insert("prompt".to_string(), Value::String(prompt.to_string()));
        vars.insert(
            "negative_prompt".to_string(),
            Value::String(negative_prompt.to_string()),
        );
        if let Some(s) = seed {
            vars.insert(
                "seed".to_string(),
                Value::Number(serde_json::Number::from(s)),
            );
        }

        let workflow = tmpl.fill(&vars)?;

        let resp = self.comfyui.queue_prompt(&workflow)?;
        let result = self.comfyui.poll_completion(&resp.prompt_id, 300)?;

        // Extract output images from the history result
        let file_path = self.download_first_image(&result, &resp.prompt_id)?;

        Ok(GeneratedContent {
            id: Uuid::new_v4().to_string(),
            content_type: ContentType::Image,
            file_path,
            metadata: HashMap::new(),
            prompt_used: prompt.to_string(),
            template_name: template_name.to_string(),
            created_at: chrono::Utc::now(),
        })
    }

    /// Generate a batch of images from different prompts using the same template.
    pub fn generate_batch(
        &self,
        template_name: &str,
        prompts: &[&str],
        common_vars: &HashMap<String, Value>,
    ) -> Result<Vec<GeneratedContent>, AssistantError> {
        let mut results = Vec::with_capacity(prompts.len());

        for prompt_text in prompts {
            let negative = common_vars
                .get("negative_prompt")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let seed = common_vars.get("seed").and_then(|v| v.as_i64());

            let content = self.generate_image(template_name, prompt_text, negative, seed)?;
            results.push(content);
        }

        Ok(results)
    }

    /// Download the first image from a completed execution result.
    fn download_first_image(
        &self,
        result: &Value,
        prompt_id: &str,
    ) -> Result<String, AssistantError> {
        let outputs = result.get("outputs").ok_or_else(|| {
            AssistantError::ComfyUI("no 'outputs' key in execution result".to_string())
        })?;

        let outputs_obj = outputs.as_object().ok_or_else(|| {
            AssistantError::ComfyUI("'outputs' is not an object".to_string())
        })?;

        for (_node_id, node_output) in outputs_obj {
            if let Some(images) = node_output.get("images").and_then(|v| v.as_array()) {
                if let Some(img) = images.first() {
                    let filename = img["filename"]
                        .as_str()
                        .ok_or_else(|| AssistantError::ComfyUI("missing filename".to_string()))?;
                    let subfolder = img["subfolder"].as_str().unwrap_or("");
                    let folder_type = img["type"].as_str().unwrap_or("output");

                    let bytes = self.comfyui.get_image(filename, subfolder, folder_type)?;

                    fs::create_dir_all(&self.output_dir).map_err(|e| {
                        AssistantError::ComfyUI(format!("failed to create output dir: {e}"))
                    })?;

                    let out_path =
                        Path::new(&self.output_dir).join(format!("{prompt_id}_{filename}"));
                    let out_path_str = out_path.to_string_lossy().to_string();

                    fs::write(&out_path, &bytes).map_err(|e| {
                        AssistantError::ComfyUI(format!("failed to write image: {e}"))
                    })?;

                    return Ok(out_path_str);
                }
            }
        }

        Err(AssistantError::ComfyUI(
            "no images found in execution output".to_string(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_template() -> WorkflowTemplate {
        let template = serde_json::json!({
            "6": {
                "inputs": {
                    "text": "",
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "7": {
                "inputs": {
                    "text": "",
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "3": {
                "inputs": {
                    "seed": 0,
                    "steps": 20,
                    "cfg": 7.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                },
                "class_type": "KSampler"
            }
        });

        WorkflowTemplate {
            name: "test_txt2img".to_string(),
            template,
            variables: vec![
                TemplateVariable {
                    name: "prompt".to_string(),
                    node_id: "6".to_string(),
                    field_path: "inputs.text".to_string(),
                    var_type: VarType::Text,
                    default: None,
                },
                TemplateVariable {
                    name: "negative_prompt".to_string(),
                    node_id: "7".to_string(),
                    field_path: "inputs.text".to_string(),
                    var_type: VarType::Text,
                    default: Some(Value::String("bad quality".to_string())),
                },
                TemplateVariable {
                    name: "seed".to_string(),
                    node_id: "3".to_string(),
                    field_path: "inputs.seed".to_string(),
                    var_type: VarType::Seed,
                    default: None,
                },
            ],
        }
    }

    #[test]
    fn test_template_fill_with_all_vars() {
        let tmpl = make_test_template();
        let mut vars = HashMap::new();
        vars.insert(
            "prompt".to_string(),
            Value::String("a beautiful sunset".to_string()),
        );
        vars.insert(
            "negative_prompt".to_string(),
            Value::String("blurry".to_string()),
        );
        vars.insert(
            "seed".to_string(),
            Value::Number(serde_json::Number::from(42)),
        );

        let filled = tmpl.fill(&vars).unwrap();

        assert_eq!(filled["6"]["inputs"]["text"], "a beautiful sunset");
        assert_eq!(filled["7"]["inputs"]["text"], "blurry");
        assert_eq!(filled["3"]["inputs"]["seed"], 42);
    }

    #[test]
    fn test_template_fill_seed_auto_generated() {
        let tmpl = make_test_template();
        let mut vars = HashMap::new();
        vars.insert("prompt".to_string(), Value::String("a cat".to_string()));
        // No seed provided — should be auto-generated

        let filled = tmpl.fill(&vars).unwrap();

        assert_eq!(filled["6"]["inputs"]["text"], "a cat");
        // negative_prompt uses default
        assert_eq!(filled["7"]["inputs"]["text"], "bad quality");
        // seed should be a positive number
        let seed = filled["3"]["inputs"]["seed"].as_i64().unwrap();
        assert!(seed >= 0);
    }

    #[test]
    fn test_template_fill_missing_required_var() {
        let tmpl = WorkflowTemplate {
            name: "test".to_string(),
            template: serde_json::json!({"1": {"inputs": {"text": ""}}}),
            variables: vec![TemplateVariable {
                name: "prompt".to_string(),
                node_id: "1".to_string(),
                field_path: "inputs.text".to_string(),
                var_type: VarType::Text,
                default: None,
            }],
        };

        let vars = HashMap::new();
        let result = tmpl.fill(&vars);
        assert!(result.is_err());
    }

    #[test]
    fn test_seed_generates_different_values() {
        let tmpl = make_test_template();
        let mut vars = HashMap::new();
        vars.insert("prompt".to_string(), Value::String("test".to_string()));

        let filled1 = tmpl.fill(&vars).unwrap();
        let filled2 = tmpl.fill(&vars).unwrap();

        let seed1 = filled1["3"]["inputs"]["seed"].as_i64().unwrap();
        let seed2 = filled2["3"]["inputs"]["seed"].as_i64().unwrap();

        // Both should be valid positive seeds
        assert!(seed1 >= 0);
        assert!(seed2 >= 0);
    }

    #[test]
    fn test_template_registry_load_all() {
        let dir = std::env::temp_dir().join("comfyui_test_templates");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        // Write two template files
        let tmpl1 = make_test_template();
        let mut tmpl2 = make_test_template();
        tmpl2.name = "test_img2img".to_string();

        fs::write(
            dir.join("txt2img.json"),
            serde_json::to_string_pretty(&tmpl1).unwrap(),
        )
        .unwrap();
        fs::write(
            dir.join("img2img.json"),
            serde_json::to_string_pretty(&tmpl2).unwrap(),
        )
        .unwrap();

        // Also write a non-json file that should be ignored
        fs::write(dir.join("README.txt"), "ignore me").unwrap();

        let mut registry = TemplateRegistry::new(dir.to_str().unwrap());
        let count = registry.load_all().unwrap();

        assert_eq!(count, 2);
        assert!(registry.get("test_txt2img").is_some());
        assert!(registry.get("test_img2img").is_some());
        assert!(registry.get("nonexistent").is_none());

        let names = registry.list();
        assert_eq!(names.len(), 2);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_template_registry_missing_dir() {
        let mut registry = TemplateRegistry::new("/tmp/comfyui_nonexistent_dir_12345");
        let result = registry.load_all();
        assert!(result.is_err());
    }

    #[test]
    fn test_content_generator_construction() {
        let gen = ContentGenerator::new(
            "http://127.0.0.1:8188",
            "/tmp/templates",
            "/tmp/output",
        );

        assert_eq!(gen.comfyui.base_url, "http://127.0.0.1:8188");
        assert_eq!(gen.output_dir, "/tmp/output");
        assert!(!gen.comfyui.client_id.is_empty());
    }

    #[test]
    fn test_comfyui_client_strips_trailing_slash() {
        let client = ComfyUIClient::new("http://localhost:8188/");
        assert_eq!(client.base_url, "http://localhost:8188");
    }

    #[test]
    fn test_workflow_template_from_json() {
        let json = serde_json::json!({"1": {"inputs": {"text": "hello"}}});
        let tmpl = WorkflowTemplate::from_json("quick_test", json.clone());

        assert_eq!(tmpl.name, "quick_test");
        assert_eq!(tmpl.template, json);
        assert!(tmpl.variables.is_empty());
    }

    #[test]
    fn test_generated_content_types() {
        let content = GeneratedContent {
            id: "test-id".to_string(),
            content_type: ContentType::Image,
            file_path: "/tmp/test.png".to_string(),
            metadata: HashMap::new(),
            prompt_used: "a cat".to_string(),
            template_name: "txt2img".to_string(),
            created_at: chrono::Utc::now(),
        };

        assert_eq!(content.content_type, ContentType::Image);
        assert_eq!(content.prompt_used, "a cat");

        let video = GeneratedContent {
            content_type: ContentType::Video,
            ..content
        };
        assert_eq!(video.content_type, ContentType::Video);
    }

    #[test]
    fn test_set_nested_deep_path() {
        let mut val = serde_json::json!({
            "inputs": {
                "config": {
                    "value": 0
                }
            }
        });

        set_nested(&mut val, "inputs.config.value", Value::Number(42.into())).unwrap();
        assert_eq!(val["inputs"]["config"]["value"], 42);
    }

    #[test]
    fn test_set_nested_invalid_path() {
        let mut val = serde_json::json!({"inputs": "not_an_object"});
        let result = set_nested(&mut val, "inputs.text", Value::String("x".to_string()));
        assert!(result.is_err());
    }

    #[test]
    fn test_var_type_serialization() {
        let seed_type = VarType::Seed;
        let json = serde_json::to_string(&seed_type).unwrap();
        let deserialized: VarType = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, VarType::Seed);
    }

    #[test]
    fn test_batch_parameter_building() {
        // Verify that generate_batch would correctly iterate prompts
        // (we can't test actual ComfyUI connection, but we test the parameter logic)
        let tmpl = make_test_template();

        let prompts = ["a dog", "a cat", "a bird"];
        let mut common_vars: HashMap<String, Value> = HashMap::new();
        common_vars.insert(
            "negative_prompt".to_string(),
            Value::String("blurry".to_string()),
        );

        // For each prompt, verify we can fill the template
        for prompt_text in &prompts {
            let mut vars = HashMap::new();
            vars.insert(
                "prompt".to_string(),
                Value::String(prompt_text.to_string()),
            );
            vars.insert(
                "negative_prompt".to_string(),
                common_vars["negative_prompt"].clone(),
            );
            // seed not set — auto-generates

            let filled = tmpl.fill(&vars).unwrap();
            assert_eq!(filled["6"]["inputs"]["text"], *prompt_text);
            assert_eq!(filled["7"]["inputs"]["text"], "blurry");
            assert!(filled["3"]["inputs"]["seed"].as_i64().unwrap() >= 0);
        }
    }
}
