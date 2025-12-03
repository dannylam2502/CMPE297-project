"""
User authentication module for the Fact-Checking System.
Extracted from server.py to maintain separation of concerns.
"""

import sqlite3
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash

USER_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


class AuthDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._get_connection() as conn:
            conn.execute(USER_TABLE_SQL)
            conn.commit()
            self._migrate_legacy_schema(conn)

    def _migrate_legacy_schema(self, conn: sqlite3.Connection) -> None:
        cursor = conn.execute("PRAGMA table_info(users)")
        columns = {row["name"] for row in cursor.fetchall()}
        if "username" in columns and "full_name" not in columns:
            conn.execute("ALTER TABLE users RENAME TO users_legacy")
            conn.execute(USER_TABLE_SQL)
            conn.execute("""
                INSERT INTO users (id, full_name, email, password_hash, created_at)
                SELECT id, username, email, password_hash, created_at FROM users_legacy
            """)
            conn.execute("DROP TABLE users_legacy")
            conn.commit()

    def register(self, full_name: str, email: str, password: str) -> tuple[bool, str]:
        password_hash = generate_password_hash(password)
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "INSERT INTO users (full_name, email, password_hash) VALUES (?, ?, ?)",
                    (full_name, email, password_hash)
                )
                conn.commit()
            return True, "User registered successfully"
        except sqlite3.IntegrityError:
            return False, "Email already registered"

    def authenticate(self, email: str, password: str) -> dict | None:
        with self._get_connection() as conn:
            user = conn.execute(
                "SELECT id, full_name, email, password_hash FROM users WHERE email = ?",
                (email,)
            ).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            return {"id": user["id"], "full_name": user["full_name"], "email": user["email"]}
        return None
