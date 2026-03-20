import sqlite3

from newsletter_schema import DB_PATH, initialize_database


def display_runs():
    initialize_database()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, title, created_at, output_path
        FROM newsletter_runs
        ORDER BY id DESC
        """
    )
    rows = cursor.fetchall()
    conn.close()

    print("\n--- NEWSLETTER RUNS ---")
    print(f"{'ID':<4} | {'Title':<32} | {'Created At':<20} | {'Output Path'}")
    print("-" * 96)
    for row in rows:
        print(f"{row[0]:<4} | {row[1][:32]:<32} | {row[2]:<20} | {row[3] or ''}")


if __name__ == "__main__":
    display_runs()
