# Engineering Roadmap — 3 Month Sprint

Team-facing plan. Supersedes `RESEARCH.md` (UIST poster track, not pursued).
Goal: a working product with benchmarks that demonstrate business impact, on a test suite that makes future work cheap.

---

## 1. Goals

| | Goal | Success means | Owner |
|---|---|---|---|
| **G1** | **Trustworthy** | Every claimed feature actually executes. No data loss. CI red = broken. | A |
| **G2** | **Measured** | One command emits a scorecard. Baseline committed. Regressions fail CI. | A |
| **G3** | **Decisive** | Identity decisions are explicit and logged, not threshold accidents. Ambiguity → ask. | B |

**Targets.** Only three can be absolute before a baseline exists. The rest are deltas measured at M2.

| Metric | Target |
|---|---|
| False-identification rate | **< 1%** |
| Session data survival | **100%** |
| PR CI wall time | **< 5 min** |
| Fusion accuracy, non-fallback path | baseline + 25pp |
| Speaker accuracy @ 100 identities | ≥ baseline @ 4 |
| Cost per conversation-hour | ≤ 50% of baseline |
| RTF | within 15% of baseline |

A target with no baseline is a wish. Don't invent the delta numbers before M2.

---

## 2. Confirmed defects

**Tier 1 — wrong today, silently**

| # | Defect | Location |
|---|---|---|
| 1 | **Fusion primary path is dead code.** Nothing ever sets `is_speaking`; active-speaker detection can never fire. All bindings fall through to the single-person fallback. | `coordinator.py:496` reads it, `vision.py:154` never writes it |
| 2 | **Every shutdown wipes the DB.** `START_WITH_SAMPLE_DATA=True` → `clear_db()` → `DELETE FROM users`, cascading to faces, voices, chat. | `main.py:88`, `config.py:12` |
| 3 | **Speaker matching biased toward false accepts.** Threshold 0.30 plus a sticky last-speaker check at the same threshold; mislabels compound. | `audio.py:130`, `config_audio.py` |
| 4 | **Worker caches never invalidate.** Dashboard delete/rename never reaches `known_speakers` / `known_faces`. | `audio.py:66`, `inspireface_processor.py:22` |
| 5 | **Test suite deletes the production DB.** | `tests/test_database.py:13` |

**Tier 2 — architecture**

- **LLM call per utterance, serial worker, unbounded queue.** Every sentence triggers a 120B-model call; `APIWorker` processes serially; `gemini_command_queue` has no bound. This — not the ML — is the throughput and cost bottleneck. STT runs at RTF 0.085 (12× real-time).
- **Face matching is a linear scan** over every stored embedding, at 10 FPS. Fine at 4 identities, not at 200.
- **One speaker per VAD segment.** 160 ms of silence ends a sentence; overlapping speech merges two people into one embedding, then pollutes the enrolled average.
- **UDP has no loss accounting.** Drops oldest silently, no counters — drop rate is currently unmeasurable.
- **Apple Silicon only** (MLX).

**Tier 3 — hygiene**

- No CI. Tests are `print` scripts, not assertions.
- `requirements.txt` omits `mlx`, `imageio-ffmpeg`, `torch`. Clean clone is broken. Local Python is 3.10, README says 3.11.
- 733 MB PDF untracked in the tree — one `git add -A` from a wrecked repo.
- 14 unmerged branches; ~5 months dormant.
- `VLM_ACTIVE=False` everywhere — VLM path is dead in shipped config.

---

## 3. Tooling decisions

| Question | Verdict | Why |
|---|---|---|
| CI/CD platform | **GitHub Actions** | Repo is public → free ARM macOS runners. Anything else means paying for Apple Silicon. |
| Dynatrace / APM | **No** — use OpenTelemetry | Enterprise APM for distributed fleets; we have 4 processes on one host. OTel instruments once, exports anywhere, no lock-in. |
| Kafka | **No** | Durable distributed streaming. We have single-host IPC. Operating it would cost more than the pipeline. |
| Redis | **Not yet — build the seam** | Real case exists (splitting ML workers onto a bigger host), but that's a post-profiling call. Decide at M3, in writing, either way. |

**We need benchmarking, not monitoring.** Benchmarking is offline, reproducible, comparative. Monitoring is live and alerting. The existing log-queue → WebSocket dashboard already covers live introspection for a team this size.

The queue defect is the missing bound and drop policy, not the transport. Add an `EventBus` interface plus event recording to disk — that buys deterministic replay (which the harness needs anyway) and makes Redis a swap-the-implementation change later.

---

## 4. Test strategy

Three different things get called "tests." Only one needs labels.

| Tier | Needs | Catches | Where |
|---|---|---|---|
| **Logic** | Nothing — zero data | All 5 Tier-1 defects | ubuntu, `-m unit`, <60s |
| **Regression** | Data, no labels | Unintended behavior change | macOS ARM, `-m models` |
| **Accuracy** | Labels | How good the ML is | macOS ARM + nightly |

**The coordinator is a pure function over dicts.** `_handle_speech`, `_try_visual_voice_association`, `_resolve_unknown_face` take event dicts and emit event dicts — no models, no network. `setup()` is already separate from `run()`, so handlers can be driven in-process. That's ~60% of the harness value at zero data cost.

### Root-cause fix, before any tests

Replace loose dicts with frozen dataclasses in `core/events.py` (`FaceObservation`, `SpeechEvent`). `.get("is_speaking", False)` is the anti-pattern — a silent default turns a contract violation into normal-looking behavior. With a dataclass, a producer that omits the field raises at construction. Defect #1 becomes unrepresentable rather than merely detectable. ~2 hours.

### Where labels come from

| Source | Gives | Effort |
|---|---|---|
| **Filenames** — `solo_voices/{Matt,Shaun,Tiger}.wav` (39s), single-person mp4s | Speaker + face ground truth, free | 0 |
| **Synthetic conversations** — segment solo WAVs, recombine with known boundaries | Unlimited labeled multi-speaker audio; sweep turn-gap, overlap, SNR | 2 days |
| **Golden files** — run pipeline, correct output, commit as expected | WER; every future change is a reviewable diff | <1 hr |
| **12 scripted recordings** — 30–60s, 3–4 people | Fusion / enrollment truth; doubles as demo footage | half day |
| **Public sets** — AMI, VoxCeleb1-O, LibriSpeech | Component numbers comparable to published work | half day |

**Synthetic conversations are the highest-leverage item in the sprint.** `VadSegmenter` in `tests/benchmark_rtf.py` is already most of the segmentation half. The difficulty sweep turns "84% accurate" into "holds to 88% until turn gaps drop below 300 ms, then collapses" — a better engineering result and a better business slide.

We are building a **regression suite, not a research corpus**: coverage of failure modes, not statistical power. ~20 scenarios, not 10,000 samples.

### Layout

```
tests/
  conftest.py          # tmp-db + coordinator fixtures
  unit/                # no models, no network, no data
  pipeline/            # real models, macOS ARM
  e2e/                 # full loop over UDP
  corpus/generate.py   # synthetic conversation builder
  scorecard/run.py     # `make scorecard`
```

One suite, three pytest markers (`unit`, `models`, `e2e`). CI selects by marker; local default excludes the slow ones.

---

## 5. CI/CD

| Workflow | Runner | Trigger | Budget | Does |
|---|---|---|---|---|
| `pr.yml` | ubuntu-latest | every PR, **required** | <5 min | ruff, clean `uv sync`, `-m unit`, reject files >50 MB |
| `ml.yml` | macos-15 (ARM) | every PR, **required** | <30 min | real models, cached weights, `-m models` |
| `nightly.yml` | macos-15 | cron + manual | <45 min | full scorecard, diff vs `baseline.json`, open issue on regression |

All five Tier-1 defects are catchable in `pr.yml` — no models required. Cache model weights (`~/.cache/huggingface`, `~/.inspireface`) keyed on `uv.lock`; first run pulls ~600 MB, cached runs start in seconds.

**Branch protection on `main`:** both required checks, one approval, no direct pushes, auto-delete merged branches.

**CD is not deployment** — this runs on a laptop/edge device. Scope: `uv.lock` committed, tagged releases, and a `make demo` that works from a cold clone on someone else's machine. Treat that as a tested path, not documentation.

---

## 6. Standards

**If there are two ways to do something, delete one.**

| Concern | The one way |
|---|---|
| Python | 3.11, pinned in `.python-version` |
| Dependencies | `uv` + committed `uv.lock`. No bare `pip install`. |
| Lint + format | `ruff` only |
| Tests | `pytest`, assertions not `print` |
| Config | One frozen `Settings` dataclass |
| Events | `core/events.py` dataclasses |

### Config consolidation — do this first

Seven constants are defined in two files each (`USER_NAME`, `VLM_ACTIVE`, `BUFFER_DURATION`, `TEMPERATURE`, `MAXOUTPUTTOKENS`, `TOOLS_JSON_PATH`, `PROMPT_JSON_PATH`), and every tunable threshold is *also* hardcoded in a worker:

- `SIMILARITY_THRESHOLD = 0.30` vs. hardcoded `sim > 0.30` at `coordinator.py:355`
- `CONFIDENCE_THRESHOLD_MATCHING = 0.5` vs. hardcoded `0.5` at `vision.py:38`
- Fusion weights `0.7 / 0.3 / 0.3` at `coordinator.py:454-455, 505` exist in no config at all

Two developers cannot tune the same system this way: changing config silently does nothing, and the numbers that *do* drive behavior aren't discoverable. Merge the five `config_*.py` into one frozen dataclass, then add a test that fails on any float literal outside `{0.0, 1.0}` in `workers/`. That mechanically prevents the drift returning.

### Definition of Done — `.github/pull_request_template.md`

```
- [ ] Tests added, and they fail without this change
- [ ] No new magic numbers — thresholds go in Settings
- [ ] Scorecard impact stated: which metric moves, which way
- [ ] Dead code deleted, not commented out
- [ ] Works from a clean clone
```

Line 3 is what makes goals real. A PR that can't name the metric it moves has an opinion, not a goal.

### Docs — three files, no more

- `CONTRIBUTING.md` — setup, conventions, DoD. Human-written.
- `METRICS.md` — current scorecard. **Generated, never hand-edited.**
- `CLAUDE.md` — architecture + constraints, so a new dev or agent is productive immediately.

---

## 7. Plan — 3 workstreams, 6 milestones

| | Workstream | Owns |
|---|---|---|
| **A** | Harness, CI, measurement | `tests/`, `.github/`, OTel instrumentation |
| **B** | Correctness & agent decisions | `coordinator.py`, `vision.py`, tool configs |
| **C** | Pipeline & scale | `audio.py`, `api_worker.py`, `shared_mem.py`, `database/` |

**Staffing.** 2 → A+C merged, B standalone. 3 → as above. 4 → split A into harness vs. CI/release.

| Milestone | Work | Exit criteria |
|---|---|---|
| **M1** wk 1–2 | `core/events.py`; config consolidation; `pr.yml` + branch protection; repo hygiene; decide `is_speaking` — implement or delete, no third state | Clean clone passes CI. No shutdown data loss. Zero files >50 MB. |
| **M2** wk 3–4 | Unit + corpus layers; golden files; OTel spans on every queue hand-off; event recording → replay | **`baseline.json` committed.** One command emits WER, speaker split, fusion by path, drop rate, p50/p95. |
| **M3** wk 5–6 | `ml.yml` + `nightly.yml`; profile under sustained load; gate the LLM call | Bottleneck ranking written with numbers. Cost/hour measured. **Redis decision recorded.** |
| **M4** wk 7–8 | Coordinator stops deciding from thresholds — assembles evidence packet, agent returns decision + confidence + **logged reason** | Every enrollment has a rationale. Fusion accuracy no longer depends on the fallback, provable from the path split. |
| **M5** wk 9–10 | `ask_clarification` tool; vector index; scale to 100+ identities; re-tune thresholds against false-ID asymmetry | False-ID @ 100 identities beats M2 baseline @ 4. Ambiguity resolves by asking. |
| **M6** wk 11–12 | Re-run scorecard; consent model; embeddings encrypted at rest; deletion reaching worker caches; cold-clone demo | Before/after table fit to show a customer. Biometric handling survives a procurement question. |

**Sequencing rules**

- **B is the critical path** and hardest to parallelize — one file, heavy rework. Strongest person, don't split.
- **A must lead.** Every claim in M4–M6 is gated on the M2 baseline.
- **C is most parallelizable** — queue fixes, indexing, profiling are independent. A 4th person adds most here.
- **Baseline before rework.** Measuring after you change things gives a number with nothing to compare against.
- **Write the failing test before the fix.** Defects #1 and #2 have known expected behavior, so those tests are specifications you can write today. Red CI on day one proves the harness detects things.

---

## 8. Metrics contract

`make scorecard` → `scorecard.json` + `METRICS.md`. Same keys every run, so any two runs diff cleanly:

```
git_sha, timestamp, hardware, dataset
accuracy.wer
accuracy.speaker  { correct, wrong, unknown }
accuracy.fusion   { active_speaker, single_person_fallback, failed }
performance       { rtf, p50_ms, p95_ms, drop_rate }
cost              { llm_calls_per_hour, usd_per_hour }
```

**Three rules that keep the numbers honest**

1. **Baseline is committed.** `baseline.json` changes only via a PR that explains why.
2. **CI enforces the ratchet.** Nightly fails on degradation past tolerance. Numbers nobody checks become numbers nobody trusts.
3. **Splits, never single numbers.** Speaker is always correct/wrong/unknown; fusion is always broken out by path. Today 100% of bindings come from the fallback — a blended number would look fine and hide it forever.

**Failure modes to watch**

- *Baseline rot* — dataset stops representing reality. Add labeled clips each milestone; record the dataset in every scorecard.
- *Metric gaming* — thresholds tuned to flatter the numbers. False-ID is always reported next to accuracy; it punishes overconfidence.
- *CI erosion* — slow or flaky CI gets bypassed. Hard 5-minute budget; a flaky test is fixed or deleted within a day, never muted.

**Cadence.** Per PR: DoD + CI. Nightly: auto-scorecard, auto-issue. Weekly 15 min: *did the number move the way we said it would?* Per milestone 30 min: exit criteria met, yes/no — not "mostly."

---

## 9. Business framing

The product claim is **ambient identity memory**: you never have to admit you forgot someone's name.

| Use case | Fit | Metric that proves it |
|---|---|---|
| **Accessibility** — prosopagnosia, low vision, memory impairment | Strongest. Zero-enrollment is a requirement, not a convenience. | Time-to-correct-ID; recall on next encounter |
| **Client-facing / sales** | Direct revenue attribution | **False-ID rate — must be near zero** |
| **Care facilities** | High turnover, hundreds of recurring faces | Enrollment throughput; accuracy at 100+ |
| **Conferences** | Dense, high-churn, naturally full of introductions | Enrollment success per conversation-hour |

**The headline metric is false-identification rate, not accuracy.** Saying nothing is neutral; saying the wrong name in a client meeting is product-killing. Tune every threshold against that asymmetry.

**Hard gate before any pilot:** we enroll biometric face and voice templates of third parties who never opted in, streamed unauthenticated over UDP. BIPA, CUBI, and GDPR Art. 9 all treat that as live liability. Needs a consent model, encryption at rest, and a retention policy — M6, not an afterthought.

---

## Week 1

1. `core/events.py` typed events *(2 h)*
2. Merge five config modules into one `Settings` dataclass; move every hardcoded threshold in *(1 day)*
3. `pyproject.toml` + `uv.lock` + `.python-version`; delete `requirements.txt` *(4 h)*
4. `pr.yml` + branch protection; purge the 733 MB PDF; >50 MB pre-commit hook *(4 h)*
5. Triage 14 branches — cherry-pick what's live, delete the rest *(2 h)*
6. `CONTRIBUTING.md` + PR template *(2 h)*

One person-week. Makes everything after it parallelizable, which is the point — G1 and G3 can't run concurrently until the config seam exists.
