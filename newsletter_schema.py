import sqlite3

DB_PATH = "newsletter_agent.db"


def initialize_database(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS newsletter_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brief TEXT NOT NULL,
            audience TEXT,
            tone TEXT,
            title TEXT,
            queries_json TEXT NOT NULL,
            sections_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            output_path TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS source_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            query_text TEXT NOT NULL,
            rank_index INTEGER NOT NULL,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            snippet TEXT,
            article_text TEXT,
            source_summary TEXT,
            relevance_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(run_id, url),
            FOREIGN KEY (run_id) REFERENCES newsletter_runs (id)
        )
        """
    )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    initialize_database()
    print("Newsletter database initialized.")
