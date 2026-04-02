"""
Test Steve's agentic DB query flow.

Seeds the database with Tiger + Shaun conversations across the last few days,
then has Tiger ask Steve: "what did Shaun talk about with me 2 days ago?"

Run from the project root:
    python -m tests.test_steve_db_query
"""

import multiprocessing as mp
import sqlite3
import time
import uuid
from datetime import datetime, timedelta

from database.database import DatabaseManager
from workers.api_worker import APIWorker


def seed_db():
    db = DatabaseManager()

    # Upsert Tiger
    tiger_ids = db.get_user_ids_by_name("Tiger")
    tiger_id = tiger_ids[0] if tiger_ids else str(uuid.uuid4())
    db.create_user(tiger_id, "Tiger")

    # Upsert Shaun
    shaun_ids = db.get_user_ids_by_name("Shaun")
    shaun_id = shaun_ids[0] if shaun_ids else str(uuid.uuid4())
    db.create_user(shaun_id, "Shaun")

    two_days_ago = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = [
        # --- 2 days ago ---
        (tiger_id, "Tiger: Hey Shaun, what do you think about the new smart glasses prototype?", two_days_ago),
        (shaun_id, "Shaun: I think the face recognition is coming along really well.", two_days_ago),
        (tiger_id, "Tiger: Yeah, I want to add database queries so Steve can look up conversations.", two_days_ago),
        (shaun_id, "Shaun: That's a great idea. We should also look at improving voice identification accuracy.", two_days_ago),
        (shaun_id, "Shaun: And let's add a better UI for the control panel dashboard.", two_days_ago),
        (tiger_id, "Tiger: Agreed. I'll start on the Steve tool calling this week.", two_days_ago),
        # --- yesterday ---
        (tiger_id, "Tiger: Good morning Shaun.", yesterday),
        (shaun_id, "Shaun: Morning! Did you sort out that memory leak we found yesterday?", yesterday),
        (tiger_id, "Tiger: Still working on it, almost there.", yesterday),
        (shaun_id, "Shaun: No rush. Let me know if you need help debugging.", yesterday),
        # --- today ---
        (tiger_id, "Tiger: Shaun, the new Steve tool-calling features are working.", today),
        (shaun_id, "Shaun: Awesome! Can't wait to test them out.", today),
    ]

    conn = sqlite3.connect(db.db_path)
    conn.execute("PRAGMA journal_mode = WAL;")
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT INTO chat_history (user_id, transcript, timestamp) VALUES (?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()

    print(f"Seeded {len(rows)} conversation rows.")
    print(f"  Tiger  ID: {tiger_id}")
    print(f"  Shaun  ID: {shaun_id}")
    print(f"  2 days ago timestamp: {two_days_ago}")


def run_steve_query():
    in_q = mp.Queue()
    out_q = mp.Queue()

    worker = APIWorker(in_q, out_q)
    worker.start()

    # Simulate the coordinator's prompt format — Tiger asking Steve about Shaun
    prompt = (
        "Transcript:\n"
        "Tiger: Hey Steve, what did Shaun talk about with me 2 days ago?\n\n"
        "Task: Parse the intent for the latest message from Tiger."
    )

    try:
        print(f"\nSending prompt to Steve:\n  {prompt.splitlines()[1]}")
        in_q.put({
            "cmd": "PARSE_INTENT",
            "text": prompt,
            "timestamp": time.time(),
            "voice_embedding": None,
        })

        event = out_q.get(timeout=40)

        print("\n--- Steve's Response ---")
        if event.get("cmd") == "SPEAK":
            print(f"[STEVE]: {event['args'].get('message', '')}")
        else:
            print(f"Unexpected event cmd '{event.get('cmd')}': {event}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        worker.shutdown()
        worker.join(timeout=5)
        if worker.is_alive():
            worker.terminate()


if __name__ == "__main__":
    print("=" * 50)
    print("Step 1: Seeding database")
    print("=" * 50)
    seed_db()

    print("\n" + "=" * 50)
    print("Step 2: Tiger asks Steve about Shaun (2 days ago)")
    print("=" * 50)
    run_steve_query()
