use crate::error::AssistantError;
use crate::providers::MemoryProvider;
use rusqlite::Connection;
use std::sync::Mutex;

/// A MemoryProvider backed by SQLite for simple persistent memory storage.
pub struct SimpleMemoryStore {
    conn: Mutex<Connection>,
}

impl SimpleMemoryStore {
    pub fn new(db_path: &str) -> Result<Self, AssistantError> {
        let conn = if db_path == ":memory:" {
            Connection::open_in_memory()
        } else {
            Connection::open(db_path)
        }
        .map_err(|e| AssistantError::Memory(format!("Failed to open database: {}", e)))?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);",
        )
        .map_err(|e| AssistantError::Memory(format!("Failed to create table: {}", e)))?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }
}

impl MemoryProvider for SimpleMemoryStore {
    fn store(&self, text: &str, source: &str) -> Result<(), AssistantError> {
        let conn = self.conn.lock().unwrap();
        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();

        conn.execute(
            "INSERT INTO memories (id, content, source, created_at) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![id, text, source, now],
        )
        .map_err(|e| AssistantError::Memory(format!("Failed to store memory: {}", e)))?;

        Ok(())
    }

    fn recall(&self, query: &str, limit: usize) -> Result<Vec<String>, AssistantError> {
        let conn = self.conn.lock().unwrap();
        let pattern = format!("%{}%", query);

        let mut stmt = conn
            .prepare(
                "SELECT content FROM memories WHERE content LIKE ?1 ORDER BY created_at DESC LIMIT ?2",
            )
            .map_err(|e| AssistantError::Memory(format!("Failed to prepare query: {}", e)))?;

        let rows = stmt
            .query_map(rusqlite::params![pattern, limit], |row| {
                row.get::<_, String>(0)
            })
            .map_err(|e| AssistantError::Memory(format!("Failed to query memories: {}", e)))?;

        let mut results = Vec::new();
        for row in rows {
            results.push(
                row.map_err(|e| AssistantError::Memory(format!("Failed to read row: {}", e)))?,
            );
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_recall() {
        let store = SimpleMemoryStore::new(":memory:").unwrap();

        store.store("The weather is sunny today", "chat").unwrap();
        store
            .store("I prefer dark mode for coding", "chat")
            .unwrap();
        store.store("Meeting at 3pm tomorrow", "chat").unwrap();

        let results = store.recall("weather", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].contains("sunny"));

        let results = store.recall("coding", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].contains("dark mode"));
    }

    #[test]
    fn test_recall_respects_limit() {
        let store = SimpleMemoryStore::new(":memory:").unwrap();

        for i in 0..10 {
            store
                .store(&format!("Memory item number {}", i), "test")
                .unwrap();
        }

        let results = store.recall("Memory", 3).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_recall_empty_query() {
        let store = SimpleMemoryStore::new(":memory:").unwrap();
        store.store("something", "test").unwrap();

        // Empty query with % pattern matches everything
        let results = store.recall("", 10).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_recall_no_matches() {
        let store = SimpleMemoryStore::new(":memory:").unwrap();
        store.store("hello world", "test").unwrap();

        let results = store.recall("xyznonexistent", 10).unwrap();
        assert!(results.is_empty());
    }
}
