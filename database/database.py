import io
import os
import sqlite3

from core.config import IDENTITY_DB_PATH
import numpy as np


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
        return sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)

    def _init_db(self):
        """Create schema & table if doesn't exist"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # name can be not unique; might have some bugs there
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL
                )
            """)

            # facial embeddings; many to one with relation to user
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    embedding ARRAY NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)

            # voice embeddings
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS voice_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    embedding ARRAY NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)

            # history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    transcript TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)

            conn.commit()

    def _get_or_create_user(self, name: str) -> int:
        """Return user id if exist; otherwise create"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE name = ?", (name,))
            row = cursor.fetchone()
            if row:
                return row[0]

            cursor.execute("INSERT INTO users (name) VALUES (?)", (name,))
            conn.commit()
            return cursor.lastrowid

    def save_face_embedding(self, name: str, embedding: np.ndarray):
        """Save new face embedding"""
        user_id = self._get_or_create_user(name)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO face_embeddings (user_id, embedding) VALUES (?, ?)",
                (user_id, embedding),
            )
            conn.commit()

    def save_voice_embedding(self, name: str, embedding: np.ndarray):
        """Save new voice embedding"""
        user_id = self._get_or_create_user(name)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO voice_embeddings (user_id, embedding) VALUES (?, ?)",
                (user_id, embedding),
            )
            conn.commit()

    def get_all_faces(self) -> dict:
        """return dict mapping: user to faces"""
        faces_dict = {}
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT u.name, f.embedding 
                FROM users u 
                JOIN face_embeddings f ON u.id = f.user_id
            """)
            for name, emb in cursor.fetchall():
                if name not in faces_dict:
                    faces_dict[name] = []
                faces_dict[name].append(emb)
        return faces_dict

    def get_all_voices(self) -> dict:
        """dict of voice embeddings"""
        voices_dict = {}
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT u.name, v.embedding 
                FROM users u 
                JOIN voice_embeddings v ON u.id = v.user_id
            """)
            for name, emb in cursor.fetchall():
                if name not in voices_dict:
                    voices_dict[name] = []
                voices_dict[name].append(emb)
        return voices_dict
    
    def save_chat_history(self, name: str, transcript: str):
        user_id = self._get_or_create_user(name)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chat_history (user_id, transcript) VALUES (?, ?)",
                (user_id, transcript)
            )
            conn.commit()

    def get_chat_history(self, name: str, limit: int = 10) -> list:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT transcript, timestamp 
                FROM chat_history h
                JOIN users u ON h.user_id = u.id
                WHERE u.name = ?
                ORDER BY h.id DESC
                LIMIT ?
            """, (name, limit))
            
            # return list of tuples: [("transcript stuff"), timestamp]
            return cursor.fetchall()
        
    
