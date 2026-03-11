use crate::error::AssistantError;
use crate::providers::KnowledgeProvider;
use crate::types::Source;
use rusqlite::Connection;
use std::sync::Mutex;

/// A KnowledgeProvider backed by SQLite with basic text search.
pub struct SimpleKnowledgeStore {
    conn: Mutex<Connection>,
}

impl SimpleKnowledgeStore {
    pub fn new(db_path: &str) -> Result<Self, AssistantError> {
        let conn = if db_path == ":memory:" {
            Connection::open_in_memory()
        } else {
            Connection::open(db_path)
        }
        .map_err(|e| AssistantError::Knowledge(format!("Failed to open database: {}", e)))?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                content TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents(id)
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);",
        )
        .map_err(|e| AssistantError::Knowledge(format!("Failed to create tables: {}", e)))?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Split content into chunks by paragraphs (double newlines).
    fn chunk_content(content: &str) -> Vec<String> {
        content
            .split("\n\n")
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }
}

impl KnowledgeProvider for SimpleKnowledgeStore {
    fn ingest(&self, title: &str, content: &str) -> Result<(), AssistantError> {
        let conn = self.conn.lock().unwrap();
        let doc_id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();

        conn.execute(
            "INSERT INTO documents (id, title, content, created_at) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![doc_id, title, content, now],
        )
        .map_err(|e| AssistantError::Knowledge(format!("Failed to store document: {}", e)))?;

        let chunks = Self::chunk_content(content);
        for (i, chunk) in chunks.iter().enumerate() {
            let chunk_id = uuid::Uuid::new_v4().to_string();
            conn.execute(
                "INSERT INTO chunks (id, doc_id, content, chunk_index) VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![chunk_id, doc_id, chunk, i as i64],
            )
            .map_err(|e| AssistantError::Knowledge(format!("Failed to store chunk: {}", e)))?;
        }

        Ok(())
    }

    fn query(&self, text: &str, top_k: usize) -> Result<Vec<Source>, AssistantError> {
        let conn = self.conn.lock().unwrap();
        let pattern = format!("%{}%", text);

        let mut stmt = conn
            .prepare(
                "SELECT d.title, c.content
                 FROM chunks c
                 JOIN documents d ON d.id = c.doc_id
                 WHERE c.content LIKE ?1
                 ORDER BY d.created_at DESC, c.chunk_index ASC
                 LIMIT ?2",
            )
            .map_err(|e| AssistantError::Knowledge(format!("Failed to prepare query: {}", e)))?;

        let rows = stmt
            .query_map(rusqlite::params![pattern, top_k], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|e| AssistantError::Knowledge(format!("Failed to query: {}", e)))?;

        let mut results = Vec::new();
        for row in rows {
            let (title, content) = row
                .map_err(|e| AssistantError::Knowledge(format!("Failed to read row: {}", e)))?;
            results.push(Source {
                title,
                chunk_preview: content,
                relevance_score: 1.0, // Simple LIKE match, no real scoring
            });
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ingest_and_query() {
        let store = SimpleKnowledgeStore::new(":memory:").unwrap();

        store
            .ingest(
                "Rust Guide",
                "Rust is a systems programming language.\n\nIt focuses on safety and performance.\n\nThe borrow checker prevents data races.",
            )
            .unwrap();

        let results = store.query("safety", 5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Rust Guide");
        assert!(results[0].chunk_preview.contains("safety"));
    }

    #[test]
    fn test_query_multiple_docs() {
        let store = SimpleKnowledgeStore::new(":memory:").unwrap();

        store
            .ingest("Doc A", "Rust has great tooling.\n\nCargo is the package manager.")
            .unwrap();
        store
            .ingest("Doc B", "Python is popular for AI.\n\nRust is gaining traction in AI too.")
            .unwrap();

        let results = store.query("Rust", 10).unwrap();
        assert!(results.len() >= 2);
    }

    #[test]
    fn test_query_no_results() {
        let store = SimpleKnowledgeStore::new(":memory:").unwrap();
        store.ingest("Test", "Hello world").unwrap();

        let results = store.query("xyznonexistent", 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_chunking() {
        let chunks = SimpleKnowledgeStore::chunk_content("first paragraph\n\nsecond paragraph\n\nthird");
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "first paragraph");
        assert_eq!(chunks[1], "second paragraph");
        assert_eq!(chunks[2], "third");
    }

    #[test]
    fn test_query_respects_limit() {
        let store = SimpleKnowledgeStore::new(":memory:").unwrap();

        // Ingest a document with many chunks containing "test"
        let content = (0..10)
            .map(|i| format!("Test paragraph number {}", i))
            .collect::<Vec<_>>()
            .join("\n\n");
        store.ingest("Many Chunks", &content).unwrap();

        let results = store.query("Test", 3).unwrap();
        assert_eq!(results.len(), 3);
    }
}
