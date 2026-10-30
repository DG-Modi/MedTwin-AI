import sqlite3
from app.db.session import init_db

def migrate():
    conn = sqlite3.connect("medtwin.db")
    cursor = conn.cursor()
    # Check user_profiles columns
    cursor.execute("PRAGMA table_info(user_profiles)")
    cols = [row[1] for row in cursor.fetchall()]
    if "clinical_tags" not in cols:
        cursor.execute("ALTER TABLE user_profiles ADD COLUMN clinical_tags TEXT DEFAULT '[]'")
        print("Added clinical_tags column to user_profiles")

    # Check chat_messages columns
    cursor.execute("PRAGMA table_info(chat_messages)")
    msg_cols = [row[1] for row in cursor.fetchall()]
    if "reasoning_steps" not in msg_cols:
        cursor.execute("ALTER TABLE chat_messages ADD COLUMN reasoning_steps TEXT DEFAULT '[]'")
        print("Added reasoning_steps column to chat_messages")
    if "latency_ms" not in msg_cols:
        cursor.execute("ALTER TABLE chat_messages ADD COLUMN latency_ms REAL DEFAULT 0.0")
        print("Added latency_ms column to chat_messages")

    conn.commit()
    conn.close()
    init_db()
    print("Database schema migration complete.")

if __name__ == "__main__":
    migrate()
