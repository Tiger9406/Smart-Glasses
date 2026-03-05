import os
import numpy as np

from database.database import DatabaseManager
from core.config import IDENTITY_DB_PATH

def run_tests():
    print("--- Starting Database Tests ---")
    
    if os.path.exists(IDENTITY_DB_PATH):
        os.remove(IDENTITY_DB_PATH)
        
    db = DatabaseManager(db_path=IDENTITY_DB_PATH)
    print("Database initialized successfully")

    user_name = "Alice"
    
    dummy_face_emb_1 = np.random.rand(128).astype(np.float32)
    dummy_face_emb_2 = np.random.rand(128).astype(np.float32)
    dummy_voice_emb = np.random.rand(256).astype(np.float32)

    print("Saving embeddings")
    db.save_face_embedding(user_name, dummy_face_emb_1)
    db.save_face_embedding(user_name, dummy_face_emb_2)
    db.save_voice_embedding(user_name, dummy_voice_emb)
    
    all_faces = db.get_all_faces()
    all_voices = db.get_all_voices()

    assert user_name in all_faces, "User not found in face dictionary"
    assert len(all_faces[user_name]) == 2, "Should have exactly 2 face embeddings"
    
    assert np.array_equal(all_faces[user_name][0], dummy_face_emb_1), "Face embedding 1 corrupted!"
    assert np.array_equal(all_faces[user_name][1], dummy_face_emb_2), "Face embedding 2 corrupted!"
    assert np.array_equal(all_voices[user_name][0], dummy_voice_emb), "Voice embedding corrupted!"
    
    print("NumPy Array Adapter/Converter working; array conversion successful")

    print("Saving chat history...")
    db.save_chat_history(user_name, "Hello, system!")
    db.save_chat_history(user_name, "What is my name?")
    
    chat_history = db.get_chat_history(user_name, limit=5)
    
    assert len(chat_history) == 2, "Should have exactly 2 chat entries"
    assert chat_history[0][0] == "What is my name?", "Chat order or text mismatch!"
    assert chat_history[1][0] == "Hello, system!", "Chat text mismatch!"
    
    print("Chat history saved and retrieved in correct order (newest first).")

    # 6. Cleanup
    if os.path.exists(IDENTITY_DB_PATH):
        os.remove(IDENTITY_DB_PATH)
        os.remove(IDENTITY_DB_PATH+"-shm")
        os.remove(IDENTITY_DB_PATH+"-wal")
    print("Test database cleaned up")
    print("--- 🎉 All Tests Passed! 🎉 ---")

if __name__ == "__main__":
    run_tests()