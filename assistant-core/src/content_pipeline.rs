//! Content production pipeline — end-to-end from topic planning to multi-platform
//! publishing with self-evolution.

use crate::error::AssistantError;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;


// ---------------------------------------------------------------------------
// Platform & ContentFormat
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Platform {
    Twitter,
    LinkedIn,
    YouTube,
    Xiaohongshu,
    Instagram,
    Custom(String),
}

impl Platform {
    /// Max character length for the primary text field, if applicable.
    pub fn max_text_length(&self) -> Option<usize> {
        match self {
            Platform::Twitter => Some(280),
            Platform::LinkedIn => Some(3000),
            Platform::Instagram => Some(2200),
            Platform::Xiaohongshu => Some(1000),
            _ => None,
        }
    }

    fn key(&self) -> String {
        match self {
            Platform::Twitter => "twitter".to_string(),
            Platform::LinkedIn => "linkedin".to_string(),
            Platform::YouTube => "youtube".to_string(),
            Platform::Xiaohongshu => "xiaohongshu".to_string(),
            Platform::Instagram => "instagram".to_string(),
            Platform::Custom(s) => format!("custom:{}", s),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContentFormat {
    TextPost,
    ImagePost,
    Carousel,
    ShortVideo,
    LongVideo,
    Thread,
    Article,
}

impl ContentFormat {
    fn key(&self) -> &'static str {
        match self {
            ContentFormat::TextPost => "text_post",
            ContentFormat::ImagePost => "image_post",
            ContentFormat::Carousel => "carousel",
            ContentFormat::ShortVideo => "short_video",
            ContentFormat::LongVideo => "long_video",
            ContentFormat::Thread => "thread",
            ContentFormat::Article => "article",
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 1: Topic Planning
// ---------------------------------------------------------------------------

pub struct TopicPlanner;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicPlan {
    pub topics: Vec<PlannedTopic>,
    pub date: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedTopic {
    pub id: Uuid,
    pub title: String,
    pub angle: String,
    pub target_platforms: Vec<Platform>,
    pub content_formats: Vec<ContentFormat>,
    pub priority: f32,
    pub trending_score: Option<f32>,
    pub keywords: Vec<String>,
}

impl TopicPlanner {
    pub fn new() -> Self {
        Self
    }

    /// Ask an LLM to generate topic ideas based on persona + niche.
    ///
    /// The `llm` closure receives a prompt string and returns the raw LLM text
    /// (expected to be JSON).  This keeps the module free of any concrete
    /// provider dependency.
    pub fn plan_topics(
        &self,
        persona: &Persona,
        niche: &str,
        count: usize,
        llm: &dyn Fn(&str) -> Result<String, AssistantError>,
    ) -> Result<TopicPlan, AssistantError> {
        let prompt = format!(
            r#"You are a content strategist. Given the following persona and niche, generate {count} content topic ideas.

Persona:
{persona_prompt}

Niche: {niche}

Respond with a JSON array of objects, each with:
- "title": string
- "angle": string (unique perspective)
- "target_platforms": array of strings from ["Twitter","LinkedIn","YouTube","Xiaohongshu","Instagram"]
- "content_formats": array of strings from ["TextPost","ImagePost","Carousel","ShortVideo","LongVideo","Thread","Article"]
- "priority": number 0.0-1.0
- "trending_score": number or null
- "keywords": array of strings

Return ONLY the JSON array, no markdown fences."#,
            count = count,
            persona_prompt = persona.to_system_prompt(),
            niche = niche,
        );

        let raw = llm(&prompt)?;
        let topics = Self::parse_topics_json(&raw)?;

        Ok(TopicPlan {
            topics,
            date: Utc::now(),
        })
    }

    fn parse_topics_json(raw: &str) -> Result<Vec<PlannedTopic>, AssistantError> {
        let arr: Vec<serde_json::Value> = serde_json::from_str(raw)
            .map_err(|e| AssistantError::Reasoning(format!("failed to parse topic JSON: {e}")))?;

        let mut topics = Vec::with_capacity(arr.len());
        for v in &arr {
            let title = v["title"]
                .as_str()
                .unwrap_or("Untitled")
                .to_string();
            let angle = v["angle"]
                .as_str()
                .unwrap_or("")
                .to_string();

            let target_platforms: Vec<Platform> = v["target_platforms"]
                .as_array()
                .map(|a| a.iter().filter_map(|p| Self::parse_platform(p.as_str()?)).collect())
                .unwrap_or_else(|| vec![Platform::Twitter]);

            let content_formats: Vec<ContentFormat> = v["content_formats"]
                .as_array()
                .map(|a| a.iter().filter_map(|f| Self::parse_format(f.as_str()?)).collect())
                .unwrap_or_else(|| vec![ContentFormat::TextPost]);

            let priority = v["priority"].as_f64().unwrap_or(0.5) as f32;
            let trending_score = v["trending_score"].as_f64().map(|f| f as f32);
            let keywords: Vec<String> = v["keywords"]
                .as_array()
                .map(|a| a.iter().filter_map(|k| k.as_str().map(String::from)).collect())
                .unwrap_or_default();

            topics.push(PlannedTopic {
                id: Uuid::new_v4(),
                title,
                angle,
                target_platforms,
                content_formats,
                priority,
                trending_score,
                keywords,
            });
        }
        Ok(topics)
    }

    fn parse_platform(s: &str) -> Option<Platform> {
        match s {
            "Twitter" => Some(Platform::Twitter),
            "LinkedIn" => Some(Platform::LinkedIn),
            "YouTube" => Some(Platform::YouTube),
            "Xiaohongshu" => Some(Platform::Xiaohongshu),
            "Instagram" => Some(Platform::Instagram),
            other => Some(Platform::Custom(other.to_string())),
        }
    }

    fn parse_format(s: &str) -> Option<ContentFormat> {
        match s {
            "TextPost" => Some(ContentFormat::TextPost),
            "ImagePost" => Some(ContentFormat::ImagePost),
            "Carousel" => Some(ContentFormat::Carousel),
            "ShortVideo" => Some(ContentFormat::ShortVideo),
            "LongVideo" => Some(ContentFormat::LongVideo),
            "Thread" => Some(ContentFormat::Thread),
            "Article" => Some(ContentFormat::Article),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 2: Persona Management
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Persona {
    pub name: String,
    pub bio: String,
    pub voice_traits: Vec<String>,
    pub expertise: Vec<String>,
    pub target_audience: String,
    pub tone: String,
    pub language: String,
    pub banned_topics: Vec<String>,
    pub example_posts: Vec<String>,
}

impl Persona {
    /// Load a persona definition from a JSON file.
    pub fn from_file(path: &str) -> Result<Self, AssistantError> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| AssistantError::Config(format!("failed to read persona file: {e}")))?;
        serde_json::from_str(&data)
            .map_err(|e| AssistantError::Config(format!("failed to parse persona JSON: {e}")))
    }

    /// Convert persona into a full system prompt for LLM content generation.
    pub fn to_system_prompt(&self) -> String {
        let mut prompt = format!(
            "You are {name}, {bio}\n\n\
             Voice traits: {traits}\n\
             Expertise: {expertise}\n\
             Target audience: {audience}\n\
             Tone: {tone}\n\
             Language: {lang}\n",
            name = self.name,
            bio = self.bio,
            traits = self.voice_traits.join(", "),
            expertise = self.expertise.join(", "),
            audience = self.target_audience,
            tone = self.tone,
            lang = self.language,
        );

        if !self.banned_topics.is_empty() {
            prompt.push_str(&format!(
                "Never discuss: {}\n",
                self.banned_topics.join(", ")
            ));
        }

        if !self.example_posts.is_empty() {
            prompt.push_str("\nExample posts for style reference:\n");
            for (i, ex) in self.example_posts.iter().enumerate() {
                prompt.push_str(&format!("{}. {}\n", i + 1, ex));
            }
        }

        prompt
    }

    /// A shorter style guide (for content review prompts).
    pub fn to_style_guide(&self) -> String {
        format!(
            "Author: {name} | Tone: {tone} | Traits: {traits} | Audience: {audience}",
            name = self.name,
            tone = self.tone,
            traits = self.voice_traits.join(", "),
            audience = self.target_audience,
        )
    }
}

// ---------------------------------------------------------------------------
// Phase 3: Content Production
// ---------------------------------------------------------------------------

pub struct ContentProducer;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProducedContent {
    pub id: Uuid,
    pub topic_id: Uuid,
    pub platform: Platform,
    pub format: ContentFormat,
    pub text: String,
    pub media_prompts: Vec<String>,
    pub hashtags: Vec<String>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub quality_score: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentVariant {
    pub platform: Platform,
    pub text: String,
    pub media_prompts: Vec<String>,
    pub max_length: Option<usize>,
}

impl ContentProducer {
    pub fn new() -> Self {
        Self
    }

    /// Generate content for a planned topic using the given persona voice.
    pub fn produce(
        &self,
        topic: &PlannedTopic,
        persona: &Persona,
        llm: &dyn Fn(&str) -> Result<String, AssistantError>,
    ) -> Result<ProducedContent, AssistantError> {
        let platform = topic
            .target_platforms
            .first()
            .cloned()
            .unwrap_or(Platform::Twitter);
        let format = topic
            .content_formats
            .first()
            .cloned()
            .unwrap_or(ContentFormat::TextPost);

        let prompt = format!(
            r#"Write a {format:?} post for {platform:?} about the following topic.

Topic: {title}
Angle: {angle}
Keywords: {keywords}

{persona}

Respond with JSON:
{{
  "text": "the post text",
  "media_prompts": ["prompt for image/video generation if needed"],
  "hashtags": ["relevant", "hashtags"]
}}

Return ONLY valid JSON, no markdown fences."#,
            format = format,
            platform = platform,
            title = topic.title,
            angle = topic.angle,
            keywords = topic.keywords.join(", "),
            persona = persona.to_style_guide(),
        );

        let raw = llm(&prompt)?;
        let v: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| AssistantError::Reasoning(format!("failed to parse content JSON: {e}")))?;

        let text = v["text"].as_str().unwrap_or("").to_string();
        let media_prompts: Vec<String> = v["media_prompts"]
            .as_array()
            .map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect())
            .unwrap_or_default();
        let hashtags: Vec<String> = v["hashtags"]
            .as_array()
            .map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect())
            .unwrap_or_default();

        Ok(ProducedContent {
            id: Uuid::new_v4(),
            topic_id: topic.id,
            platform,
            format,
            text,
            media_prompts,
            hashtags,
            metadata: HashMap::new(),
            created_at: Utc::now(),
            quality_score: None,
        })
    }

    /// Adapt existing content for a different platform.
    pub fn adapt_for_platform(
        &self,
        content: &ProducedContent,
        target: Platform,
        llm: &dyn Fn(&str) -> Result<String, AssistantError>,
    ) -> Result<ContentVariant, AssistantError> {
        let max_length = target.max_text_length();
        let length_instruction = match max_length {
            Some(n) => format!("Maximum {} characters.", n),
            None => "No strict character limit.".to_string(),
        };

        let platform_guidance = match &target {
            Platform::Twitter => "Make it punchy, concise, and hook-driven. Use 1-2 hashtags max.",
            Platform::LinkedIn => "Professional tone, can be longer, use line breaks for readability.",
            Platform::YouTube => "Script format: strong hook in first 3 seconds, clear call to action.",
            Platform::Xiaohongshu => "Casual, relatable, emoji-friendly. Focus on visual appeal.",
            Platform::Instagram => "Visual-first, engaging caption, strategic hashtag use.",
            Platform::Custom(_) => "Adapt the tone appropriately for this platform.",
        };

        let prompt = format!(
            r#"Adapt the following content for {platform:?}.

Original text: {text}

Platform guidance: {guidance}
{length}

Respond with JSON:
{{
  "text": "adapted text",
  "media_prompts": ["updated media prompts if needed"]
}}

Return ONLY valid JSON."#,
            platform = target,
            text = content.text,
            guidance = platform_guidance,
            length = length_instruction,
        );

        let raw = llm(&prompt)?;
        let v: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| AssistantError::Reasoning(format!("failed to parse adaptation JSON: {e}")))?;

        let text = v["text"].as_str().unwrap_or("").to_string();
        let media_prompts: Vec<String> = v["media_prompts"]
            .as_array()
            .map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect())
            .unwrap_or_default();

        Ok(ContentVariant {
            platform: target,
            text,
            media_prompts,
            max_length,
        })
    }

    /// Self-review content for quality, persona consistency, and engagement potential.
    pub fn self_review(
        &self,
        content: &ProducedContent,
        persona: &Persona,
        llm: &dyn Fn(&str) -> Result<String, AssistantError>,
    ) -> Result<f32, AssistantError> {
        let prompt = format!(
            r#"Review this content for quality, persona consistency, and engagement potential.

Content: {text}
Platform: {platform:?}
{style_guide}

Rate from 0.0 to 1.0 on overall quality. Respond with ONLY a JSON object:
{{"score": 0.85, "reason": "brief explanation"}}
"#,
            text = content.text,
            platform = content.platform,
            style_guide = persona.to_style_guide(),
        );

        let raw = llm(&prompt)?;
        let v: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| AssistantError::Reasoning(format!("failed to parse review JSON: {e}")))?;

        let score = v["score"]
            .as_f64()
            .ok_or_else(|| AssistantError::Reasoning("missing 'score' in review response".to_string()))?
            as f32;

        Ok(score.clamp(0.0, 1.0))
    }
}

// ---------------------------------------------------------------------------
// Phase 4: Publishing Queue
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishQueue {
    queue: Vec<QueuedPost>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueuedPost {
    pub id: Uuid,
    pub content: ProducedContent,
    pub platform: Platform,
    pub scheduled_at: Option<DateTime<Utc>>,
    pub status: PostStatus,
    pub published_at: Option<DateTime<Utc>>,
    pub external_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PostStatus {
    Draft,
    Scheduled,
    Publishing,
    Published,
    Failed(String),
}

impl PublishQueue {
    pub fn new() -> Self {
        Self { queue: Vec::new() }
    }

    /// Add content to the queue and return the queued post ID.
    pub fn add(
        &mut self,
        content: ProducedContent,
        platform: Platform,
        scheduled_at: Option<DateTime<Utc>>,
    ) -> Uuid {
        let id = Uuid::new_v4();
        let status = if scheduled_at.is_some() {
            PostStatus::Scheduled
        } else {
            PostStatus::Draft
        };
        self.queue.push(QueuedPost {
            id,
            content,
            platform,
            scheduled_at,
            status,
            published_at: None,
            external_id: None,
        });
        id
    }

    /// Return the next post that is due for publishing (scheduled_at <= now).
    pub fn next_due(&self) -> Option<&QueuedPost> {
        let now = Utc::now();
        self.queue
            .iter()
            .filter(|p| p.status == PostStatus::Scheduled)
            .filter(|p| {
                p.scheduled_at
                    .map(|t| t <= now)
                    .unwrap_or(false)
            })
            .min_by_key(|p| p.scheduled_at)
    }

    /// Mark a queued post as published.
    pub fn mark_published(&mut self, id: Uuid, external_id: &str) {
        if let Some(post) = self.queue.iter_mut().find(|p| p.id == id) {
            post.status = PostStatus::Published;
            post.published_at = Some(Utc::now());
            post.external_id = Some(external_id.to_string());
        }
    }

    /// Mark a queued post as failed.
    pub fn mark_failed(&mut self, id: Uuid, error: &str) {
        if let Some(post) = self.queue.iter_mut().find(|p| p.id == id) {
            post.status = PostStatus::Failed(error.to_string());
        }
    }

    /// List all posts with a given status.
    pub fn list_by_status(&self, status: &PostStatus) -> Vec<&QueuedPost> {
        self.queue.iter().filter(|p| &p.status == status).collect()
    }

    /// Persist queue to a JSON file.
    pub fn save(&self, path: &str) -> Result<(), AssistantError> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| AssistantError::Config(format!("failed to serialize queue: {e}")))?;
        std::fs::write(path, json)
            .map_err(|e| AssistantError::Config(format!("failed to write queue file: {e}")))?;
        Ok(())
    }

    /// Load queue from a JSON file.
    pub fn load(path: &str) -> Result<Self, AssistantError> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| AssistantError::Config(format!("failed to read queue file: {e}")))?;
        serde_json::from_str(&data)
            .map_err(|e| AssistantError::Config(format!("failed to parse queue JSON: {e}")))
    }

    /// Number of posts in the queue.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Phase 5: Feedback & Evolution
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentEvolution {
    pub topic_weights: HashMap<String, f32>,
    pub format_weights: HashMap<String, f32>,
    pub platform_weights: HashMap<String, f32>,
    pub time_weights: HashMap<String, f32>,
    pub ema_alpha: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngagementMetrics {
    pub post_id: Uuid,
    pub platform: Platform,
    pub impressions: u64,
    pub likes: u64,
    pub comments: u64,
    pub shares: u64,
    pub clicks: u64,
    pub followers_gained: i64,
    pub collected_at: DateTime<Utc>,
}

impl EngagementMetrics {
    /// Calculate an engagement rate (0.0+).
    /// Uses (likes + comments*2 + shares*3 + clicks) / max(impressions, 1).
    pub fn engagement_rate(&self) -> f32 {
        let interactions =
            self.likes as f64 + self.comments as f64 * 2.0 + self.shares as f64 * 3.0 + self.clicks as f64;
        let impressions = (self.impressions as f64).max(1.0);
        (interactions / impressions) as f32
    }
}

impl ContentEvolution {
    pub fn new(ema_alpha: f32) -> Self {
        Self {
            topic_weights: HashMap::new(),
            format_weights: HashMap::new(),
            platform_weights: HashMap::new(),
            time_weights: HashMap::new(),
            ema_alpha: ema_alpha.clamp(0.0, 1.0),
        }
    }

    /// Record engagement performance and update all weight maps via EMA.
    pub fn record_performance(&mut self, metrics: &EngagementMetrics, content: &ProducedContent) {
        let rate = metrics.engagement_rate();

        // Update topic weights using each hashtag/keyword as a topic key
        for tag in &content.hashtags {
            self.ema_update(&mut self.topic_weights.clone(), tag, rate);
        }
        // Also index by the first keyword in metadata or format the topic_id
        let topic_key = content.topic_id.to_string();
        Self::ema_update_map(&mut self.topic_weights, &topic_key, rate, self.ema_alpha);
        for tag in &content.hashtags {
            Self::ema_update_map(&mut self.topic_weights, tag, rate, self.ema_alpha);
        }

        // Update format weights
        let format_key = content.format.key().to_string();
        Self::ema_update_map(&mut self.format_weights, &format_key, rate, self.ema_alpha);

        // Update platform weights
        let platform_key = metrics.platform.key();
        Self::ema_update_map(&mut self.platform_weights, &platform_key, rate, self.ema_alpha);

        // Update time weights (hour-of-day bucket)
        let hour_key = format!("hour_{:02}", content.created_at.format("%H"));
        Self::ema_update_map(&mut self.time_weights, &hour_key, rate, self.ema_alpha);
    }

    fn ema_update_map(map: &mut HashMap<String, f32>, key: &str, value: f32, alpha: f32) {
        let current = map.get(key).copied().unwrap_or(value);
        let updated = alpha * value + (1.0 - alpha) * current;
        map.insert(key.to_string(), updated);
    }

    #[allow(dead_code)]
    fn ema_update(&self, _map: &mut HashMap<String, f32>, _key: &str, _value: f32) {
        // Intentionally unused — see ema_update_map
    }

    /// Return the top-N topics by weight (descending).
    pub fn best_topics(&self, count: usize) -> Vec<(&str, f32)> {
        Self::sorted_top(&self.topic_weights, count)
    }

    /// Return the top-N formats by weight (descending).
    pub fn best_formats(&self, count: usize) -> Vec<(&str, f32)> {
        Self::sorted_top(&self.format_weights, count)
    }

    /// Return the top-N platforms by weight (descending).
    pub fn best_platforms(&self, count: usize) -> Vec<(&str, f32)> {
        Self::sorted_top(&self.platform_weights, count)
    }

    /// Return the top-N posting times by weight (descending).
    pub fn best_posting_times(&self, count: usize) -> Vec<(&str, f32)> {
        Self::sorted_top(&self.time_weights, count)
    }

    fn sorted_top(map: &HashMap<String, f32>, count: usize) -> Vec<(&str, f32)> {
        let mut entries: Vec<(&str, f32)> = map.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        entries.truncate(count);
        entries
    }

    /// Persist evolution state to JSON.
    pub fn save(&self, path: &str) -> Result<(), AssistantError> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| AssistantError::Config(format!("failed to serialize evolution: {e}")))?;
        std::fs::write(path, json)
            .map_err(|e| AssistantError::Config(format!("failed to write evolution file: {e}")))?;
        Ok(())
    }

    /// Load evolution state from JSON.
    pub fn load(path: &str) -> Result<Self, AssistantError> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| AssistantError::Config(format!("failed to read evolution file: {e}")))?;
        serde_json::from_str(&data)
            .map_err(|e| AssistantError::Config(format!("failed to parse evolution JSON: {e}")))
    }
}

// ---------------------------------------------------------------------------
// Phase 6: Full Pipeline Orchestrator
// ---------------------------------------------------------------------------

pub struct ContentPipeline {
    pub planner: TopicPlanner,
    pub producer: ContentProducer,
    pub queue: PublishQueue,
    pub evolution: ContentEvolution,
    pub persona: Persona,
}

impl ContentPipeline {
    pub fn new(persona: Persona) -> Self {
        Self {
            planner: TopicPlanner::new(),
            producer: ContentProducer::new(),
            queue: PublishQueue::new(),
            evolution: ContentEvolution::new(0.3),
            persona,
        }
    }

    /// Plan topics, produce content, self-review, and enqueue.
    pub fn plan_and_produce(
        &mut self,
        niche: &str,
        count: usize,
        llm: &dyn Fn(&str) -> Result<String, AssistantError>,
    ) -> Result<Vec<ProducedContent>, AssistantError> {
        let plan = self.planner.plan_topics(&self.persona, niche, count, llm)?;

        let mut produced = Vec::new();
        for topic in &plan.topics {
            let mut content = self.producer.produce(topic, &self.persona, llm)?;

            // Self-review and attach score
            match self.producer.self_review(&content, &self.persona, llm) {
                Ok(score) => content.quality_score = Some(score),
                Err(_) => {} // non-fatal
            }

            // Add to publish queue as draft
            let platform = content.platform.clone();
            self.queue.add(content.clone(), platform, None);
            produced.push(content);
        }

        Ok(produced)
    }

    /// Get the next post that should be published.
    pub fn get_next_post(&self) -> Option<&QueuedPost> {
        self.queue.next_due()
    }

    /// Record engagement feedback and update evolution weights.
    pub fn record_feedback(&mut self, _post_id: Uuid, metrics: EngagementMetrics) {
        // Find the content associated with this post in the queue
        if let Some(post) = self.queue.queue.iter().find(|p| p.content.id == metrics.post_id) {
            let content = post.content.clone();
            self.evolution.record_performance(&metrics, &content);
        }
    }

    /// Human-readable summary of evolution insights.
    pub fn evolution_report(&self) -> String {
        let mut report = String::from("=== Content Evolution Report ===\n\n");

        let topics = self.evolution.best_topics(5);
        if !topics.is_empty() {
            report.push_str("Top Topics:\n");
            for (name, w) in &topics {
                report.push_str(&format!("  {}: {:.4}\n", name, w));
            }
            report.push('\n');
        }

        let formats = self.evolution.best_formats(5);
        if !formats.is_empty() {
            report.push_str("Top Formats:\n");
            for (name, w) in &formats {
                report.push_str(&format!("  {}: {:.4}\n", name, w));
            }
            report.push('\n');
        }

        let platforms = self.evolution.best_platforms(5);
        if !platforms.is_empty() {
            report.push_str("Top Platforms:\n");
            for (name, w) in &platforms {
                report.push_str(&format!("  {}: {:.4}\n", name, w));
            }
            report.push('\n');
        }

        let times = self.evolution.best_posting_times(5);
        if !times.is_empty() {
            report.push_str("Top Posting Times:\n");
            for (name, w) in &times {
                report.push_str(&format!("  {}: {:.4}\n", name, w));
            }
        }

        report
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    fn test_persona() -> Persona {
        Persona {
            name: "TestBot".to_string(),
            bio: "A test persona for unit tests".to_string(),
            voice_traits: vec!["witty".to_string(), "technical".to_string()],
            expertise: vec!["Rust".to_string(), "AI".to_string()],
            target_audience: "developers".to_string(),
            tone: "casual-professional".to_string(),
            language: "en".to_string(),
            banned_topics: vec!["politics".to_string()],
            example_posts: vec!["Rust's borrow checker is secretly your best friend.".to_string()],
        }
    }

    /// Mock LLM that returns canned JSON based on keywords in the prompt.
    fn mock_llm(prompt: &str) -> Result<String, AssistantError> {
        if prompt.contains("content strategist") || prompt.contains("topic ideas") {
            Ok(r#"[
                {
                    "title": "Why Rust is eating the world",
                    "angle": "Performance vs safety tradeoff",
                    "target_platforms": ["Twitter", "LinkedIn"],
                    "content_formats": ["TextPost", "Thread"],
                    "priority": 0.9,
                    "trending_score": 0.7,
                    "keywords": ["rust", "performance", "safety"]
                },
                {
                    "title": "AI agents need better memory",
                    "angle": "Current RAG limitations",
                    "target_platforms": ["Twitter"],
                    "content_formats": ["TextPost"],
                    "priority": 0.8,
                    "trending_score": null,
                    "keywords": ["ai", "memory", "rag"]
                }
            ]"#
            .to_string())
        } else if prompt.contains("Adapt the following") {
            Ok(r#"{"text": "Adapted for the platform!", "media_prompts": []}"#.to_string())
        } else if prompt.contains("Review this content") {
            Ok(r#"{"score": 0.85, "reason": "Good quality content"}"#.to_string())
        } else {
            // Default: produce content
            Ok(r#"{"text": "This is a great post about the topic!", "media_prompts": ["A futuristic cityscape"], "hashtags": ["rust", "coding"]}"#.to_string())
        }
    }

    // -----------------------------------------------------------------------
    // 1. Persona from_file + to_system_prompt
    // -----------------------------------------------------------------------

    #[test]
    fn test_persona_from_file_and_system_prompt() {
        let persona = test_persona();

        // Write to temp file
        let dir = std::env::temp_dir();
        let path = dir.join("test_persona.json");
        let json = serde_json::to_string_pretty(&persona).unwrap();
        std::fs::write(&path, &json).unwrap();

        // Load from file
        let loaded = Persona::from_file(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.name, "TestBot");
        assert_eq!(loaded.voice_traits, vec!["witty", "technical"]);

        // System prompt
        let prompt = loaded.to_system_prompt();
        assert!(prompt.contains("TestBot"));
        assert!(prompt.contains("witty, technical"));
        assert!(prompt.contains("Never discuss: politics"));
        assert!(prompt.contains("borrow checker"));

        // Style guide
        let guide = loaded.to_style_guide();
        assert!(guide.contains("TestBot"));
        assert!(guide.contains("casual-professional"));

        let _ = std::fs::remove_file(&path);
    }

    // -----------------------------------------------------------------------
    // 2. TopicPlanner with mock LLM
    // -----------------------------------------------------------------------

    #[test]
    fn test_topic_planner_with_mock_llm() {
        let planner = TopicPlanner::new();
        let persona = test_persona();

        let plan = planner
            .plan_topics(&persona, "systems programming", 2, &mock_llm)
            .unwrap();

        assert_eq!(plan.topics.len(), 2);
        assert_eq!(plan.topics[0].title, "Why Rust is eating the world");
        assert_eq!(plan.topics[0].target_platforms, vec![Platform::Twitter, Platform::LinkedIn]);
        assert_eq!(plan.topics[0].content_formats, vec![ContentFormat::TextPost, ContentFormat::Thread]);
        assert!((plan.topics[0].priority - 0.9).abs() < 0.01);
        assert!(plan.topics[0].trending_score.is_some());
        assert!(plan.topics[1].trending_score.is_none());
        assert_eq!(plan.topics[0].keywords, vec!["rust", "performance", "safety"]);
    }

    // -----------------------------------------------------------------------
    // 3. ContentProducer with mock LLM
    // -----------------------------------------------------------------------

    #[test]
    fn test_content_producer_with_mock_llm() {
        let producer = ContentProducer::new();
        let persona = test_persona();

        let topic = PlannedTopic {
            id: Uuid::new_v4(),
            title: "Rust ownership".to_string(),
            angle: "Mental model".to_string(),
            target_platforms: vec![Platform::Twitter],
            content_formats: vec![ContentFormat::TextPost],
            priority: 0.8,
            trending_score: None,
            keywords: vec!["rust".to_string()],
        };

        let content = producer.produce(&topic, &persona, &mock_llm).unwrap();
        assert_eq!(content.topic_id, topic.id);
        assert_eq!(content.platform, Platform::Twitter);
        assert_eq!(content.format, ContentFormat::TextPost);
        assert!(!content.text.is_empty());
        assert!(!content.hashtags.is_empty());
    }

    // -----------------------------------------------------------------------
    // 4. Platform adaptation (Twitter length limit)
    // -----------------------------------------------------------------------

    #[test]
    fn test_platform_adaptation() {
        let producer = ContentProducer::new();
        let persona = test_persona();

        let topic = PlannedTopic {
            id: Uuid::new_v4(),
            title: "Test".to_string(),
            angle: "Test".to_string(),
            target_platforms: vec![Platform::LinkedIn],
            content_formats: vec![ContentFormat::TextPost],
            priority: 0.5,
            trending_score: None,
            keywords: vec![],
        };

        let content = producer.produce(&topic, &persona, &mock_llm).unwrap();

        let variant = producer
            .adapt_for_platform(&content, Platform::Twitter, &mock_llm)
            .unwrap();

        assert_eq!(variant.platform, Platform::Twitter);
        assert_eq!(variant.max_length, Some(280));
        assert!(!variant.text.is_empty());
    }

    // -----------------------------------------------------------------------
    // 5. Self-review scoring
    // -----------------------------------------------------------------------

    #[test]
    fn test_self_review_scoring() {
        let producer = ContentProducer::new();
        let persona = test_persona();

        let content = ProducedContent {
            id: Uuid::new_v4(),
            topic_id: Uuid::new_v4(),
            platform: Platform::Twitter,
            format: ContentFormat::TextPost,
            text: "Test content for review".to_string(),
            media_prompts: vec![],
            hashtags: vec!["test".to_string()],
            metadata: HashMap::new(),
            created_at: Utc::now(),
            quality_score: None,
        };

        let score = producer.self_review(&content, &persona, &mock_llm).unwrap();
        assert!((score - 0.85).abs() < 0.01);
        assert!(score >= 0.0 && score <= 1.0);
    }

    // -----------------------------------------------------------------------
    // 6. PublishQueue: add, next_due, mark_published, mark_failed
    // -----------------------------------------------------------------------

    #[test]
    fn test_publish_queue_operations() {
        let mut queue = PublishQueue::new();
        assert!(queue.is_empty());

        let content = ProducedContent {
            id: Uuid::new_v4(),
            topic_id: Uuid::new_v4(),
            platform: Platform::Twitter,
            format: ContentFormat::TextPost,
            text: "Hello world".to_string(),
            media_prompts: vec![],
            hashtags: vec![],
            metadata: HashMap::new(),
            created_at: Utc::now(),
            quality_score: Some(0.9),
        };

        // Add as draft (no scheduled time)
        let id1 = queue.add(content.clone(), Platform::Twitter, None);
        assert_eq!(queue.len(), 1);
        assert_eq!(queue.list_by_status(&PostStatus::Draft).len(), 1);

        // Add as scheduled (in the past so it's due)
        let past = Utc::now() - chrono::Duration::hours(1);
        let id2 = queue.add(content.clone(), Platform::LinkedIn, Some(past));
        assert_eq!(queue.len(), 2);
        assert_eq!(queue.list_by_status(&PostStatus::Scheduled).len(), 1);

        // next_due should return the scheduled post
        let due = queue.next_due().unwrap();
        assert_eq!(due.id, id2);

        // mark_published
        queue.mark_published(id2, "ext-123");
        assert_eq!(queue.list_by_status(&PostStatus::Published).len(), 1);
        let published = queue.list_by_status(&PostStatus::Published);
        assert_eq!(published[0].external_id.as_deref(), Some("ext-123"));
        assert!(published[0].published_at.is_some());

        // mark_failed
        queue.mark_failed(id1, "network error");
        let failed = queue.list_by_status(&PostStatus::Failed("network error".to_string()));
        assert_eq!(failed.len(), 1);

        // next_due should now return None (nothing scheduled)
        assert!(queue.next_due().is_none());
    }

    // -----------------------------------------------------------------------
    // 7. PublishQueue save/load roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_publish_queue_save_load_roundtrip() {
        let mut queue = PublishQueue::new();

        let content = ProducedContent {
            id: Uuid::new_v4(),
            topic_id: Uuid::new_v4(),
            platform: Platform::Twitter,
            format: ContentFormat::TextPost,
            text: "Roundtrip test".to_string(),
            media_prompts: vec!["cool image".to_string()],
            hashtags: vec!["test".to_string()],
            metadata: HashMap::new(),
            created_at: Utc::now(),
            quality_score: Some(0.75),
        };

        let id = queue.add(content, Platform::Twitter, None);

        let dir = std::env::temp_dir();
        let path = dir.join("test_queue.json");
        let path_str = path.to_str().unwrap();

        queue.save(path_str).unwrap();
        let loaded = PublishQueue::load(path_str).unwrap();

        assert_eq!(loaded.len(), 1);
        let post = &loaded.queue[0];
        assert_eq!(post.id, id);
        assert_eq!(post.content.text, "Roundtrip test");
        assert_eq!(post.status, PostStatus::Draft);

        let _ = std::fs::remove_file(&path);
    }

    // -----------------------------------------------------------------------
    // 8. ContentEvolution: record_performance updates weights (EMA math)
    // -----------------------------------------------------------------------

    #[test]
    fn test_evolution_ema_updates() {
        let mut evo = ContentEvolution::new(0.5);

        let content = ProducedContent {
            id: Uuid::new_v4(),
            topic_id: Uuid::new_v4(),
            platform: Platform::Twitter,
            format: ContentFormat::TextPost,
            text: "Test".to_string(),
            media_prompts: vec![],
            hashtags: vec!["rust".to_string()],
            metadata: HashMap::new(),
            created_at: Utc::now(),
            quality_score: None,
        };

        // First observation: engagement rate = (10 + 2*2 + 1*3 + 5) / 1000 = 0.022
        let metrics1 = EngagementMetrics {
            post_id: content.id,
            platform: Platform::Twitter,
            impressions: 1000,
            likes: 10,
            comments: 2,
            shares: 1,
            clicks: 5,
            followers_gained: 1,
            collected_at: Utc::now(),
        };

        evo.record_performance(&metrics1, &content);

        // First observation: EMA initializes to the value itself
        // alpha=0.5: new = 0.5 * rate + 0.5 * rate = rate (since no prior)
        let rate1 = metrics1.engagement_rate();
        let expected = (10.0 + 2.0 * 2.0 + 1.0 * 3.0 + 5.0) / 1000.0;
        assert!((rate1 - expected as f32).abs() < 0.001);

        // The topic weight for "rust" should be set
        assert!(evo.topic_weights.contains_key("rust"));
        let w1 = evo.topic_weights["rust"];
        assert!((w1 - rate1).abs() < 0.001);

        // Second observation with higher engagement
        let metrics2 = EngagementMetrics {
            post_id: content.id,
            platform: Platform::Twitter,
            impressions: 1000,
            likes: 50,
            comments: 10,
            shares: 5,
            clicks: 20,
            followers_gained: 5,
            collected_at: Utc::now(),
        };
        let rate2 = metrics2.engagement_rate();

        evo.record_performance(&metrics2, &content);

        // EMA: new_w = 0.5 * rate2 + 0.5 * w1
        let expected_w2 = 0.5 * rate2 + 0.5 * w1;
        let w2 = evo.topic_weights["rust"];
        assert!((w2 - expected_w2).abs() < 0.001, "EMA update: got {w2}, expected {expected_w2}");

        // Weight should have increased since rate2 > rate1
        assert!(w2 > w1);
    }

    // -----------------------------------------------------------------------
    // 9. ContentEvolution: best_topics returns correct order
    // -----------------------------------------------------------------------

    #[test]
    fn test_evolution_best_topics_order() {
        let mut evo = ContentEvolution::new(1.0); // alpha=1.0 means just use latest value

        evo.topic_weights.insert("rust".to_string(), 0.9);
        evo.topic_weights.insert("python".to_string(), 0.3);
        evo.topic_weights.insert("go".to_string(), 0.6);

        let best = evo.best_topics(2);
        assert_eq!(best.len(), 2);
        assert_eq!(best[0].0, "rust");
        assert!((best[0].1 - 0.9).abs() < 0.001);
        assert_eq!(best[1].0, "go");
        assert!((best[1].1 - 0.6).abs() < 0.001);

        // Formats/platforms work the same way
        evo.format_weights.insert("text_post".to_string(), 0.8);
        evo.format_weights.insert("thread".to_string(), 0.4);
        let best_fmt = evo.best_formats(1);
        assert_eq!(best_fmt[0].0, "text_post");
    }

    // -----------------------------------------------------------------------
    // 10. ContentEvolution save/load roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_evolution_save_load_roundtrip() {
        let mut evo = ContentEvolution::new(0.3);
        evo.topic_weights.insert("rust".to_string(), 0.8);
        evo.format_weights.insert("text_post".to_string(), 0.5);
        evo.platform_weights.insert("twitter".to_string(), 0.7);
        evo.time_weights.insert("hour_14".to_string(), 0.6);

        let dir = std::env::temp_dir();
        let path = dir.join("test_evolution.json");
        let path_str = path.to_str().unwrap();

        evo.save(path_str).unwrap();
        let loaded = ContentEvolution::load(path_str).unwrap();

        assert!((loaded.ema_alpha - 0.3).abs() < 0.001);
        assert!((loaded.topic_weights["rust"] - 0.8).abs() < 0.001);
        assert!((loaded.format_weights["text_post"] - 0.5).abs() < 0.001);
        assert!((loaded.platform_weights["twitter"] - 0.7).abs() < 0.001);
        assert!((loaded.time_weights["hour_14"] - 0.6).abs() < 0.001);

        let _ = std::fs::remove_file(&path);
    }

    // -----------------------------------------------------------------------
    // 11. Full pipeline: plan -> produce -> queue -> feedback -> evolution
    // -----------------------------------------------------------------------

    #[test]
    fn test_full_pipeline_end_to_end() {
        let persona = test_persona();
        let mut pipeline = ContentPipeline::new(persona);

        // Plan and produce
        let produced = pipeline
            .plan_and_produce("systems programming", 2, &mock_llm)
            .unwrap();

        assert_eq!(produced.len(), 2);
        assert!(produced[0].quality_score.is_some());

        // Queue should have 2 items
        assert_eq!(pipeline.queue.len(), 2);
        assert_eq!(pipeline.queue.list_by_status(&PostStatus::Draft).len(), 2);

        // Record feedback for first item
        let post_id = produced[0].id;
        let metrics = EngagementMetrics {
            post_id,
            platform: Platform::Twitter,
            impressions: 5000,
            likes: 100,
            comments: 20,
            shares: 10,
            clicks: 50,
            followers_gained: 10,
            collected_at: Utc::now(),
        };

        pipeline.record_feedback(post_id, metrics);

        // Evolution should now have some weights
        assert!(!pipeline.evolution.format_weights.is_empty());
        assert!(!pipeline.evolution.platform_weights.is_empty());

        // Report should be non-empty
        let report = pipeline.evolution_report();
        assert!(report.contains("Content Evolution Report"));
        assert!(report.contains("Top Formats:"));
    }

    // -----------------------------------------------------------------------
    // 12. ContentFormat and Platform serialization
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_and_platform_serialization() {
        // Platform roundtrip
        let platforms = vec![
            Platform::Twitter,
            Platform::LinkedIn,
            Platform::YouTube,
            Platform::Xiaohongshu,
            Platform::Instagram,
            Platform::Custom("tiktok".to_string()),
        ];

        let json = serde_json::to_string(&platforms).unwrap();
        let deserialized: Vec<Platform> = serde_json::from_str(&json).unwrap();
        assert_eq!(platforms, deserialized);

        // ContentFormat roundtrip
        let formats = vec![
            ContentFormat::TextPost,
            ContentFormat::ImagePost,
            ContentFormat::Carousel,
            ContentFormat::ShortVideo,
            ContentFormat::LongVideo,
            ContentFormat::Thread,
            ContentFormat::Article,
        ];

        let json = serde_json::to_string(&formats).unwrap();
        let deserialized: Vec<ContentFormat> = serde_json::from_str(&json).unwrap();
        assert_eq!(formats, deserialized);

        // PostStatus roundtrip (including Failed variant with message)
        let statuses = vec![
            PostStatus::Draft,
            PostStatus::Scheduled,
            PostStatus::Publishing,
            PostStatus::Published,
            PostStatus::Failed("timeout".to_string()),
        ];

        let json = serde_json::to_string(&statuses).unwrap();
        let deserialized: Vec<PostStatus> = serde_json::from_str(&json).unwrap();
        assert_eq!(statuses, deserialized);
    }

    // -----------------------------------------------------------------------
    // Extra: Platform max_text_length
    // -----------------------------------------------------------------------

    #[test]
    fn test_platform_max_length() {
        assert_eq!(Platform::Twitter.max_text_length(), Some(280));
        assert_eq!(Platform::LinkedIn.max_text_length(), Some(3000));
        assert_eq!(Platform::Instagram.max_text_length(), Some(2200));
        assert_eq!(Platform::YouTube.max_text_length(), None);
        assert_eq!(Platform::Custom("test".into()).max_text_length(), None);
    }

    // -----------------------------------------------------------------------
    // Extra: Engagement rate calculation
    // -----------------------------------------------------------------------

    #[test]
    fn test_engagement_rate_calculation() {
        let m = EngagementMetrics {
            post_id: Uuid::new_v4(),
            platform: Platform::Twitter,
            impressions: 1000,
            likes: 10,
            comments: 5,
            shares: 2,
            clicks: 3,
            followers_gained: 0,
            collected_at: Utc::now(),
        };

        // (10 + 5*2 + 2*3 + 3) / 1000 = (10 + 10 + 6 + 3) / 1000 = 29/1000 = 0.029
        let rate = m.engagement_rate();
        assert!((rate - 0.029).abs() < 0.001);

        // Zero impressions should not panic
        let m0 = EngagementMetrics {
            impressions: 0,
            ..m.clone()
        };
        let rate0 = m0.engagement_rate();
        assert!(rate0 > 0.0); // uses max(1, impressions)
    }

    // -----------------------------------------------------------------------
    // Extra: Persona from_file error handling
    // -----------------------------------------------------------------------

    #[test]
    fn test_persona_from_file_not_found() {
        let result = Persona::from_file("/nonexistent/path.json");
        assert!(result.is_err());
    }
}
