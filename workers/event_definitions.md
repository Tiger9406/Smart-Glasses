## Event Schema Documentation

The `Coordinator` processes an `event` dictionary received from the `results_queue`. Every event contains a `"type"` key that determines its structure.

### 1. Vision Result (`vision_result`)

Sent by the Vision worker to report detected faces and tracking data.

* **Source:** Vision Worker
* **Purpose:** Updates the temporal face cache and triggers identity registration

| Key | Type | Description |
| --- | --- | --- |
| `type` | `string` | Always `"vision_result"`. |
| `faces` | `list[dict]` | A list of detected face objects |
| `timestamp` | `float` | Timestamp when the face was processed |

**Face Object Structure:**

```python
{
    "track_id": int,          # uniqye tracking id for the face
    "bbox": (x1, y1, x2, y2), # bounding box for the face
    "name": string,           # identified name or default unknown name
    "score": float,          # confidence score with the associated name
    "emb": np.array | None    # embedding; only an np array when re-calced embedding
}

```

---

### 2. Speech/Audio Event (`speech`)

Sent by the Audio worker when human speech parsed

* **Source:** Audio/STT Worker
* **Purpose:** Provides text for intent parsing and potential identity verification via voice

| Key | Type | Description |
| --- | --- | --- |
| `type` | `string` | Always `"speech"`. |
| `text` | `string` | The transcribed text |
| `id` | `string` | The session ID (Shaun knows what this is) |
| `speech_start_time` | `float` | Time when sentence began |
| `final` | `bool` | pretty much always true; true if end of sentence |
| `embedding` | `np.array` | embedding for speaker voice |

---

### 3. VLM Result (`vlm_result`)

Sent when the Vision-Language Model completes a complex visual reasoning task

* **Source:** Gemini/VLM Worker
* **Purpose:** Provides high-level semantic descriptions of the video feed.

| Key | Type | Description |
| --- | --- | --- |
| `type` | `string` | Always `"vlm_result"` |
| `request_id` | `int` | Maps the response back to a specific `Coordinator` request |
| `text` | `string` | The natural language description/answer from the VLM. |
| `timestamp` | `float` | Time the inference was completed |

---

### 4. Intent Event (`intent`)

Generated after the LLM parses a user's speech into a specific actionable command; individual 
`cmd` and `args` defined in `core.config_tools`

* **Source:** LLM/Logic Worker
* **Purpose:** Triggers specific state changes like registering a new user.

| Key | Type | Description |
| --- | --- | --- |
| `type` | `string` | Always `"intent"` |
| `cmd` | `string` | The command type (e.g., `REGISTER_FACE`, `CHAT`) |
| `args` | `dict` | Key-value pairs containing arguments for the command.|

---

### 5. Internal Commands (Outgoing)

The Coordinator sends these to the `commands_queue` to control sub-workers.

* **`GET_VIDEO_CONTEXT`**: Requests the VLM to analyze the current video buffer.
* **`REGISTER_FACE`**: Tells the Vision worker to bind a `track_id` to a specific `name` and save the embedding.
