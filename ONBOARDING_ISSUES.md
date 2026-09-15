# Issue backlog for restarting the project

Ready-to-file GitHub issues, ordered so new members can start on day one without stepping on the people doing the hard rework. Each one is scoped to a single verified problem with acceptance criteria you can actually check.

Suggested labels to create first: `good-first-issue`, `bug`, `infra`, `core`, `docs`, `needs-design`.

**Sizing:** 🟢 ≤2 h · 🟡 half day–1 day · 🔴 multi-day

---

## Wave 0 — do these before onboarding anyone

These are repo hygiene. A new member who clones today gets a broken environment and a 733 MB PDF.

---

### 0.1 — Review and merge `steve_agent` 🟡
**Labels:** `core` · **Assign:** Shaun

`steve_agent` is pushed (`c9edef2`) and is `main` + 1 commit, 0 behind, so it merges cleanly. It is the only unmerged work in the repo and it is a complete feature: agentic DB tool-calling that lets Steve answer questions about past conversations.

It adds:
- `query_conversations` + `get_known_people` tool schemas
- a 5-iteration tool loop in `APIWorker._run_intent_with_tools`
- `DatabaseManager.search_conversations()` with relative-date parsing ("2 days ago")
- `tests/test_steve_db_query.py`

**Do:** open a PR, review it, merge. Two things to check in review: the loop calls `self.client._call_tools_api` (a private method) from `APIWorker`, and `tests/test_steve_db_query.py` seeds the **real** database rather than a temp one — same defect as issue 1.2.

**Acceptance:** merged to `main`; the test seeds a throwaway DB.

---

### 0.2 — Purge large media from the working tree and block it recurring 🟢
**Labels:** `infra`

A 733 MB PDF (`3746058.pdf`), a 12 MB `.MOV`, and several `.m4a` files sit untracked at the repo root. One `git add -A` bloats the repository permanently.

**Do:**
- Move or delete the media files.
- Extend `.gitignore` with `*.pdf`, `*.mov`, `*.MOV`, `*.m4a`, `*.mp4` (keep the intentional `api/simulator_resources/*.mp4` fixtures via a negation rule).
- Add a pre-commit hook rejecting any staged file over 50 MB.

**Acceptance:** `git status` is clean of media; a test commit of a 60 MB file is rejected by the hook.

---

### 0.3 — Fix the clean-clone install 🟢
**Labels:** `infra`, `good-first-issue`

`main.py` imports `imageio_ffmpeg`, which is not in `requirements.txt`, so a fresh clone crashes on shutdown. README says Python 3.11; the local interpreter is 3.10.

**Do:** add the missing dependency, pin a Python version in `.python-version`, and verify from a fresh venv on a machine that has never run this project.

**Acceptance:** fresh clone → venv → `pip install -r requirements.txt` → `python main.py` + `python -m api.simulator` runs end to end and shuts down cleanly with Ctrl-C.

---

### 0.4 — ~~Branch cleanup~~ ✅ done 2026-09-15
**Labels:** `infra`

Nine remote branches deleted (seven merged into `main`, plus the deprecated `GeminiAgent` and `openai_api`) and twelve stale local branches. `main` and `steve_agent` remain. The ESP32 ingest client, already deleted from GitHub by someone earlier, is preserved as the local tag `archive/ingest_from_hardware`.

**Remaining:** turn on **auto-delete branches on merge** in repo settings (Settings → General → Pull Requests) so this does not accumulate again. Optionally `git push origin archive/ingest_from_hardware` to back the tag up remotely.

---

### 0.5 — Commit the planning docs, delete the scratch scripts 🟢
**Labels:** `docs`

`ARCHITECTURE.md`, `ROADMAP.md`, `ONBOARDING_ISSUES.md`, and `tests/benchmark_rtf.py` are untracked. They are the only current description of the system and they exist on one machine. Meanwhile five superseded scratch scripts and ~745 MB of media sit alongside them at the repo root.

**Do:** follow the triage table in `ARCHITECTURE.md` §7 ("Working-tree files that were never committed") — commit four docs + the benchmark + `start.sh`, delete five scratch scripts, keep three personal docs local, remove the media and the face thumbnails.

**Acceptance:** `git status` shows a clean working tree; `README.md` links to `ARCHITECTURE.md` in its first section.

---

## Wave 1 — good first issues

Self-contained, verified bugs. Each one teaches a different part of the system.

---

### 1.1 — Every clean shutdown deletes the entire database 🟢
**Labels:** `bug`, `good-first-issue` · **Teaches:** lifecycle, config

`config.START_WITH_SAMPLE_DATA` defaults to `True`, so `main.py:96` calls `db.clear_db()` on exit, which runs `DELETE FROM users` and cascades to `face_embeddings`, `voice_embeddings`, and `chat_history`. Every person the system ever learned is wiped when you press Ctrl-C.

**Do:** separate "seed demo data on start" from "wipe everything on exit." Sample data should load into a throwaway DB path, or seeding should be idempotent and never trigger a teardown delete. Real identities must survive a restart.

**Acceptance:**
- Run, enroll someone via the simulator, Ctrl-C, restart → that person is still in the DB and still recognized.
- A test asserts `clear_db()` is not reachable from the normal shutdown path.

**Files:** `main.py:94-96`, `core/config.py:12`, `database/database.py:242`

---

### 1.2 — The test suite deletes the production database 🟢
**Labels:** `bug`, `good-first-issue` · **Teaches:** test isolation

`tests/test_database.py:13` does `os.remove(IDENTITY_DB_PATH)` against the real configured path. Running the tests destroys your working data.

**Do:** point the test at a temp path (`tmp_path` fixture once pytest lands, or `tempfile` for now). While you are in there, convert the `print` statements to real `assert`s — this file is currently a script, not a test.

**Acceptance:** `./database/identities.db` is byte-identical before and after the suite runs.

---

### 1.3 — Dashboard delete/rename never reaches the running workers 🟡
**Labels:** `bug`, `good-first-issue` · **Teaches:** the command-queue pattern

`AudioWorker.known_speakers` and `InspireFaceProcessor.known_faces` are loaded once in `setup()` and never refreshed. Delete a user in the dashboard and the running workers keep recognizing them; rename someone and the workers keep the old name until restart.

**Do:** send an invalidation command down `audio_command_queue` / `vision_command_queue` when the dashboard mutates a user. Both workers already have a `_handle_commands()` loop — add a `RELOAD_IDENTITIES` (or targeted `FORGET_USER`) case. Note that `frontend/server.py` runs in the main process and does not currently hold references to those queues; wiring them through is part of the task.

**Acceptance:** delete a user in the dashboard while the system is running → that face/voice is no longer matched, no restart.

**Files:** `frontend/server.py:181-242`, `workers/audio.py:65,290`, `workers/vision_utils/inspireface_processor.py:22`

---

### 1.4 — `memory_result` events are silently dropped 🟢
**Labels:** `bug`, `good-first-issue` · **Teaches:** the event contract

`APIWorker` emits `{"type": "memory_result", ...}` for the `ANALYZE_MEMORY` command, but `Coordinator.event_handlers` has no entry for it, so it lands in `_handle_unknown_event` and prints "got other event."

**Do:** either wire a real handler that persists extracted facts, or delete the `ANALYZE_MEMORY` path entirely. Do not leave it half-connected.

**Acceptance:** no event type emitted anywhere in the codebase lacks a Coordinator handler. Add a test that asserts this by introspecting `event_handlers`.

**Files:** `workers/api_worker.py:94-117`, `workers/coordinator.py:62-68`

---

### 1.5 — Remove the dead Gemini path (or restore it behind a flag) 🟢
**Labels:** `good-first-issue`, `needs-design`

`api/gemini_client.py` (213 lines), `tests/test_gemini_tool.py`, and `tests/test_gemini_memory.py` are live files, but every import of `GeminiClient` in the workers is commented out. New readers cannot tell which client is real.

**Do:** pick one. Either delete the Gemini files, or introduce a `LLM_PROVIDER` setting with a shared client interface so both are selectable and both are tested. Commented-out imports are not an acceptable third option.

**Acceptance:** `grep -rn "GeminiClient" workers/` returns either nothing or only live code.

---

### 1.6 — Delete unreachable config 🟢
**Labels:** `good-first-issue`, `docs`

- `config.VOICE_TRAP_LIMIT = 15.0` is never read; `Coordinator` uses its own `self.max_delay = 20.0`.
- `config.SAMPLE_FACE_EMBEDDING_PATHS` is `{}`, so the `TEST_REGISTER_IDENTITY` branch at `coordinator.py:56` can never execute.
- `database/face_thumbnails/` is referenced by no code.

**Do:** delete each, or make it real. If `VOICE_TRAP_LIMIT` was meant to drive `max_delay`, wire it up.

**Acceptance:** every name in `core/config.py` has at least one reader outside of `config.py`, enforced by a test.

---

## Wave 2 — core work

Needs someone who has read `ARCHITECTURE.md` §3 and §4 in full.

---

### 2.1 — Decide the fate of `is_speaking`: implement or delete 🔴
**Labels:** `core`, `bug`, `needs-design` · **This is the critical path**

The active-speaker fusion path — the mechanism that makes this product novel — has never executed. `coordinator.py:496` reads `face.get("is_speaking", False)`; no producer anywhere writes that key. Every voice-to-face binding in the system's history came from the single-person fallback at `coordinator.py:510`.

Two honest options:
- **Implement it.** VisionWorker computes an active-speaker signal (mouth-region motion between frames, or lip-landmark variance if InspireFace exposes landmarks) and emits it on each face object.
- **Delete it.** Remove the branch and be explicit that fusion is single-person-only today.

There is no acceptable third state where the code reads a field nobody writes.

**Acceptance:** either every `vision_result` face object carries `is_speaking` and a test proves the active-speaker branch fires on multi-person input, or the branch is gone. Fusion path must be logged per binding so the split is measurable.

**Depends on:** 2.2 (typed events would have made this unrepresentable).

---

### 2.2 — Typed events instead of loose dicts 🟡
**Labels:** `core`, `infra`

`.get("is_speaking", False)` is the pattern that hid issue 2.1 for six months: a silent default turns a contract violation into normal-looking behavior.

**Do:** add `core/events.py` with frozen dataclasses — `FaceObservation`, `SpeechEvent`, `IntentEvent`, `VisionResult` — and construct them at the producer. A producer that omits a field then raises at construction time.

**Acceptance:** no `.get(` with a default on any cross-process event payload in `workers/`.

---

### 2.3 — Consolidate five config modules into one `Settings` dataclass 🟡
**Labels:** `core`, `infra`

Seven constants are defined in two files each (`USER_NAME`, `VLM_ACTIVE`, `BUFFER_DURATION`, `TEMPERATURE`, `MAXOUTPUTTOKENS`, `TOOLS_JSON_PATH`, `PROMPT_JSON_PATH`), and every threshold that actually drives behavior is *also* hardcoded in a worker:

| Config says | Code actually uses |
|---|---|
| `SIMILARITY_THRESHOLD = 0.30` | hardcoded `0.30` at `coordinator.py:355` |
| `CONFIDENCE_THRESHOLD_MATCHING = 0.5` | hardcoded `0.5` at `vision.py:38` |
| *(nothing)* | fusion weights `0.7 / 0.3` at `coordinator.py:454-455`, `0.3 / 0.8` at `coordinator.py:505,512` |

Two people cannot tune this system concurrently: editing config silently does nothing, and the numbers that matter are not discoverable.

**Do:** merge `config.py`, `config_audio.py`, `config_vision.py`, `config_openai.py`, `config_gemini.py` into one frozen `Settings`. Move every magic number in. Then add a test that fails on any float literal outside `{0.0, 1.0}` under `workers/` — that mechanically prevents the drift returning.

**Acceptance:** changing a threshold in one place demonstrably changes behavior; the literal-check test passes.

---

### 2.4 — Gate the per-utterance LLM call 🟡
**Labels:** `core`

Every single transcribed sentence fires a call to a 120B model (`nemotron-3-super-120b-a12b`) through `APIWorker`, which processes serially, off an unbounded queue. This — not the ML — is the throughput and cost bottleneck. STT runs at RTF 0.085; the LLM round trip does not.

**Do:** skip the call when it cannot produce an actionable intent (no wake word, no name-like token, everyone in view already identified). Bound `gemini_command_queue` and define an explicit drop policy. Count calls per conversation-hour.

**Acceptance:** LLM calls per conversation-hour drops measurably on the standard simulator clip with no loss of successful enrollments.

---

### 2.5 — Speaker matching is biased toward false accepts 🟡
**Labels:** `core`, `needs-design`

`identify_speaker` (`audio.py:128`) accepts at cosine ≥ 0.30 and *also* applies a sticky last-speaker check at the same threshold. Once a sentence is mislabeled, the sticky bias makes the next one more likely to be mislabeled too, and `register_identity` folds the bad embedding into the running average.

Per `ROADMAP.md` §9, **false identification is the product-killing metric** — saying nothing is neutral, saying the wrong name in a client meeting is not.

**Do:** measure before tuning. Produce a speaker-accuracy split (correct / wrong / unknown) on labeled clips, then re-tune with an asymmetric cost. Consider requiring a higher margin before folding an embedding into a stored average.

**Acceptance:** committed baseline for the three-way split; false-identification rate reported next to accuracy in every future measurement.

**Blocked by:** needs the labeled corpus from `ROADMAP.md` §4.

---

### 2.6 — UDP loss is unmeasurable 🟢
**Labels:** `core`, `good-first-issue`

`udp_receiver.py:30-43` drops the oldest item when a queue is full, silently and without counters. The current frame/chunk drop rate is unknown.

**Do:** add counters for received, dropped-on-full, and malformed packets per stream, and surface them on the dashboard.

**Acceptance:** drop rate visible at :8765 and included in the scorecard schema from `ROADMAP.md` §8.

---

### 2.7 — Convert the tests to pytest with assertions 🟡
**Labels:** `infra`

Everything in `tests/` is a `print`-based script run by hand. Nothing fails. Nothing runs in CI.

**Do:** adopt pytest with markers `unit` / `models` / `e2e`. Start with the Coordinator — `_handle_speech`, `_try_visual_voice_association`, and `_resolve_unknown_face` are pure functions over dicts with no models and no network, and `setup()` is already separate from `run()`, so handlers can be driven in-process. That is the cheapest large win in the repo.

**Acceptance:** `pytest -m unit` runs in under 60 s on Linux with no models, no network, no data — and catches issues 1.1, 1.2, and 2.1.

**Unblocks:** the `pr.yml` CI workflow in `ROADMAP.md` §5.

---

## Filing these

```bash
gh label create good-first-issue --color 7057ff 2>/dev/null
gh label create infra --color 0e8a16 2>/dev/null
gh label create core --color d93f0b 2>/dev/null
gh label create needs-design --color fbca04 2>/dev/null

gh issue create \
  --title "Every clean shutdown deletes the entire database" \
  --label "bug,good-first-issue" \
  --body-file - <<'EOF'
<paste the issue body from section 1.1>
EOF
```

### Suggested assignment for a team of 4

| Person | Wave 1 | Wave 2 |
|---|---|---|
| Returning member A | 0.1, 0.4, 0.5 | **2.1** (critical path, do not split) |
| Returning member B | 0.2, 0.3 | 2.2, 2.3 |
| New member 1 | 1.1, 1.2, 1.4 | 2.6 |
| New member 2 | 1.3, 1.5, 1.6 | 2.7 |

2.4 and 2.5 come after 2.7 produces a baseline. Measuring after you change things gives a number with nothing to compare against.
