import io
import os
import sqlite3

import numpy as np

from core.config import IDENTITY_DB_PATH


def adapt_array(arr: np.ndarray):
    """convert numpy arr to binary data for sqlite"""
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(binary_data: bytes):
    out = io.BytesIO(binary_data)
    out.seek(0)
    return np.load(out)


sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("ARRAY", convert_array)


class DatabaseManager:
    def __init__(self, db_path=IDENTITY_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def _init_db(self):
        """Create schema & table if doesn't exist"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # id to name; id will be uuid so string
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL
                )
            """)

            # facial embeddings; many to one with relation to user
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    embedding ARRAY NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            """)

            # voice embeddings
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS voice_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    embedding ARRAY NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            """)

            # history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    transcript TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            """)

            cursor.execute(
                "CREATE INDEX IF NOT EXISTS id_face_user_id ON face_embeddings(user_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS id_voice_user_id ON voice_embeddings(user_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS id_chat_user_id ON chat_history(user_id)"
            )

            conn.commit()

    def delete_user(self, user_id: str):
        """deleting user"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()

    def create_user(self, user_id: str, name: str):
        """create user"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO users (id, name) VALUES (?, ?)
                ON CONFLICT(id) DO UPDATE SET name=excluded.name
            """,
                (user_id, name),
            )
            conn.commit()

    def update_user(self, user_id: str, new_name: str):
        """update user name"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE users SET name = ? WHERE id = ?", (new_name, user_id)
            )
            conn.commit()

    def get_user_name(self, user_id: str) -> str:
        """fetch ame for a given UUID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            return row[0] if row else "Unknown"

    def get_all_users(self) -> dict:
        """Returns a dictionary mapping user_id -> name"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name FROM users")
            return dict(cursor.fetchall())

    def save_face_embedding(self, user_id: str, embedding: np.ndarray):
        """Save new face embedding linked to UUID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO face_embeddings (user_id, embedding) VALUES (?, ?)",
                (user_id, embedding),
            )
            conn.commit()

    def save_voice_embedding(self, user_id: str, embedding: np.ndarray):
        """Save new voice embedding linked to UUID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO voice_embeddings (user_id, embedding) VALUES (?, ?)",
                (user_id, embedding),
            )
            conn.commit()

    def get_all_faces(self) -> dict:
        """return dict mapping: user_id to faces"""
        faces_dict = {}
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT user_id, embedding FROM face_embeddings")
            for uid, emb in cursor.fetchall():
                if uid not in faces_dict:
                    faces_dict[uid] = []
                faces_dict[uid].append(emb)
        return faces_dict

    def get_all_voices(self) -> dict:
        """dict of voice embeddings mapped by user_id"""
        voices_dict = {}
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT user_id, embedding FROM voice_embeddings")
            for uid, emb in cursor.fetchall():
                if uid not in voices_dict:
                    voices_dict[uid] = []
                voices_dict[uid].append(emb)
        return voices_dict

    def save_chat_history(self, user_id: str, transcript: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chat_history (user_id, transcript) VALUES (?, ?)",
                (user_id, transcript),
            )
            conn.commit()

    def get_chat_history(self, user_id: str, limit: int = 10) -> list:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT transcript, timestamp 
                FROM chat_history 
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
            """,
                (user_id, limit),
            )
            return cursor.fetchall()
    
    def get_recent_chat_history(self, limit: int = 50) -> list:
        """Returns most recent chat entries across all users, ordered oldest-first for display."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT ch.id, ch.user_id, u.name, ch.transcript, ch.timestamp
                FROM chat_history ch
                JOIN users u ON ch.user_id = u.id
                ORDER BY ch.id DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
        return [
            {"id": r[0], "user_id": r[1], "user_name": r[2], "transcript": r[3], "timestamp": str(r[4])}
            for r in reversed(rows)
        ]

    def get_user_ids_by_name(self, name: str):
        """Returns the UUID for a given name, or None if not found."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE name = ?", (name,))
            return [row[0] for row in cursor.fetchall()]
        
    def get_voice_embeddings_by_uid(self, user_id: str):
        """returns a list of voice embeddings for a given UUID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT embedding FROM voice_embeddings WHERE user_id = ?", (user_id,))
            return [emb for (emb,) in cursor.fetchall()]

    def clear_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users")
            conn.commit()
