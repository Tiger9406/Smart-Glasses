# vision_utils

Helpers for `VisionWorker` in `workers/vision.py` (`IngestionWorker`, its own process). The worker owns the processor, `active_identities`, `frame_buffer`, and the input, output, and command queues. This folder supplies the functions; it does not hold that state.

`VisionWorker.__init__` only keeps `command_queue`. `run()` calls `setup()`, which builds `InspireFaceProcessor`, so the InspireFace session is created in the child process, not worker `__init__`. Shutdown calls `session.release()`.

## InspireFace session

`inspireface_processor.py`: launch Pikachu or Megatron (`config_vision`), then `InspireFaceSession` in track-by-detection with recognition and emotion. Known embeddings load from `DatabaseManager` into `known_faces`. `detect_faces`, `extract_embedding`, `compare_to_person` (`feature_comparison`), `identify_embedding`, and `register_identity` (append in memory and `save_face_embedding`).

## Tracks and matching

`tracks.py`: `FaceTrack`, `ActiveIdentity`, `update_tracks`. New tracks start as `DEFAULT_ID` / `DEFAULT_NAME`. Recognize when still unknown or when `recheck_interval` has passed (worker: 2s). A score above `confidence_threshold` (worker: 0.5) stores the identity; otherwise the track stays unknown. Drop tracks unseen longer than `lost_after` (worker: 1s). `emb` is set only on the frame that recomputed it.

`matching.py` `best_identity` returns the highest score above the match threshold (`CONFIDENCE_THRESHOLD_MATCHING`), else `DEFAULT_ID`. That threshold is separate from the worker's track threshold.

## Decode, recognize, publish

`pipeline.py`, called from the worker loop after commands:

- `decode_frame` JPEG-decodes queue bytes (`cv2.imdecode`).
- `recognize_frame` detects, embeds, identifies, then `update_tracks`.
- `publish_faces` puts `{"type": "vision_result", "faces": ...}` with `block=False`. A full queue is skipped.

Each kept frame is also `(timestamp, raw_bytes)` on `frame_buffer` (`FPS * BUFFER_DURATION`).

## Commands

`apply_vision_commands` drains `vision_command_queue`.

- `GET_VIDEO_CONTEXT`: requires `VLM_ACTIVE`. `sampled_frames` keeps every third buffer frame; `ask_vlm` runs on the worker asyncio loop and publishes `vlm_result`. Inactive VLM returns from the drain. Empty buffer returns.
- `REGISTER_FACE`: `register_tracked_face` stores `emb` for `user_id` and renames that live `track_id`.

## Offline annotated video

If `config.SAVE_ANNOTATED_VID`, the worker archives `(time, bytes, faces)`. After the loop, `render_annotated_video` writes an mp4 (`config.VIDEO_OUTPUT_PATH`) with boxes and `name (ID: track_id)`, repeating frames to fill gaps. That render is offline, after the stream.

## Tools and samples

`get_embedding.py` is a one-shot script: read `database/sample_data/{name}_face.jpg`, extract one embedding, save `face_{name}.npy`. It is not on the live loop.

`facial_processing_demos` is sample/demo code, not the pipeline.
