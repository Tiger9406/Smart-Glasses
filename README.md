# Smart Glasses Multimodel Backend

Real-time, asycnh ML-based backend designed for smart glasses

---

## Architecture Overview

We leverage python's multiprocessing to prevent ML inference from blocking asynchronous network communication

1. **Ingestion:** Socket ingestion thread defined in `api.udp_receiver.py`
2. **Shared Memory:** Data pushed into thread-safe multiprocessing queues, in `shared_mem.py`
3. **Parallel processing:** We have dedicated workers, each its own subprocess, pulling from the queues to run heavy inferences such as inspireface and MLX parakeet in parallel.
4. **Coordination:** Central `coordinator.py` aggregating vision and audio workers' processed eventrs, manages overall server state, and decides commands/next action such as registering an identity or summarizing video frames.

## Tech Stack

- **Networking:** `UDP`
- **Concurrency:** `multiprocessing`, `asyncio`
- **Vision** `OpenCV`, `InspireFace`
- **Audio:** MLX `Parakeet` via `ONNX`
- **Context:** `GEMINI` API
- **Database:** `SQLite` (FAISS & vector db to be explored)

## Setup

Using Python 3.11; other requirements minimal as of rn; reference requirements.txt

I'm gonna use venv instead of conda; feel free to use either though

Pip install requirements.txt

First time running may download parakeet & inspireface models

We have full speech-to-text pipeline along with facial recognition pipeline

Coordinator as an agent

### 1. Clone Repo
```bash
git clone https://github.com/Tiger9406/Smart-Glasses.git
cd Smart-Glasses

```

### 2. Create Virtual Env
* **Windows:**
```bash
python -m venv .venv

```
* **macOS / Linux:**
```bash
python3 -m venv .venv

```

### 3. Use Virtual Env
* **Windows (Command Prompt):**
```bash
.venv\Scripts\activate

```

* **macOS / Linux:**
```bash
source .venv/bin/activate

```

### 4. Install Dependencies

With the environment active, install the required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt

```

### 5. Make .env

Copy .env.example and rename it to .env
Enter your gemini API key for LLM functions

### 6. Run!!

``` bash
python main.py
```

Open separate terminal, activate `.venv` again (refer above) run:

``` bash
python -m api.simulator
```

Data types & stream parameters defined in core/config.py
