# API Module

This directory handles the networking of the smartglasses backend. It's responsible for accepting real time connections and parsing incoming byte stream, and routing data to the queues.

## Components

### `simulator.py`
Simulates physical smartglasses hardware
- Reads local video (`.mp4`) and audio (`.wav`) files
- Streams the byte data over WebSockets to the local `/stream` endpoint to test the backend pipeline locally without needing physical smart glasses for now