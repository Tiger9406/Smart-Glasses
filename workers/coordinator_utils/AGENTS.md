# coordinator_utils

Event handling and voice/face association for the coordinator. `Coordinator.run` only pulls events and calls `handle_event(state)`.

## handlers.py

Speech, intent, and identity registration. `CoordinatorState` holds the db (`IdentityDirectory`), vision and audio caches, conversation history, pending voice registration, and the llm, vision, and audio command queues.

`vision_result` caches faces. `speech` binds the speaker, then enqueues `PARSE_INTENT`. `intent` runs `REGISTER_IDENTITY`, `SPEAK`, or `VISION_CONTEXT`. `vlm_result` and `api_error` log.

### Speech

An unknown voice (`config.DEFAULT_ID` plus an embedding) is cached. A pending registration inside the window (`max_delay`) binds that user and enqueues `REGISTER_VOICE`. Otherwise visual association may bind a known face and register the voice. Known users get chat history saved. Every utterance is appended and `PARSE_INTENT` is enqueued with recent history and the names currently in view.

### Identity

`REGISTER_IDENTITY` looks up the name first.

Existing name must match a face in the closest frame or a self-intro voice. On a match, register the unknown face if the face was not known, register voice on self-intro, or attempt voice registration when `speaker_name` is `config.USER_NAME`. If neither face nor voice matches and the speaker is `config.USER_NAME`, set pending voice registration on the first existing id.

New identity creates a user, registers an unknown face, registers voice on an Unknown self-intro (`config.DEFAULT_NAME`), or attempts voice registration when `config.USER_NAME` names someone. Attempt uses a cached unknown voice within `max_delay` after the intro, or sets pending voice registration for the next one.

## association.py

Timing and geometry helpers. No queues or db.

- `pending_registration_hit`: voice time is in `[0, max_delay)` after the pending registration.
- `closest_frame`: vision frame nearest the timestamp.
- `face_matches`: that uid is in the frame.
- `voice_matches`: self-intro embedding vs the mean stored voice, cosine similarity above 0.30.
- `associate_voice`: within 1s, a known face speaking in at least 30% of frames, or the only known person in at least 80%.
- `resolve_unknown_face`: unknown face with the highest attention score, `0.3 * area_scaled - 0.7 * distance_scaled`.
- `resolve_unknown_voice`: first cached unknown voice in `[0, max_delay]` after the target time.
- `trim_before`: drop cache entries older than the duration.
