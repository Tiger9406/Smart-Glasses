# RWE ESP32 Code

This repository contains the software code for RWE Smart Glasses. It enables real-time video and audio streaming to host computer on the same network via UDP

**[Link to Software Code Repository](https://github.com/Tiger9406/Smart-Glasses)**

---

## Features

* **Real-Time Video Streaming:** Captures JPEG frames using the `esp_camera` library and streams them to the server
* **Live Audio Transmission:** Reads audio data from an I2S microphone (e.g., INMP441) at 16kHz, filters DC offset, and streams it alongside the video
* **FreeRTOS Multithreading:** Utilizes ESP32's dual-core architecture to pin camera capture, audio processing, and network transmission tasks to separate cores for optimized performance

---

## Directory Structure

```text
└── ./
    ├── include/          
    ├── src/
    ├── udp_server.py       # server to receive and output streamed data
    └── websocket_server.py

```


---

## Setup

### 1. Python Server (Host Computer)

The host computer receives the stream, displays the video using OpenCV, and plays the audio via PyAudio.

**Dependencies:**

Don't forget to make an venv too

```bash
pip install opencv-python numpy pyaudio websockets

```

**Running the Server:**
Depending on your ESP32 configuration, run one of the provided servers:

```bash
python udp_server.py
```

*Note: Press `q` while focused on the video window to quit the server*

### 2. ESP32 Client

The ESP32 acts as the client, capturing media and pushing it to the server

1. **Rename Secrets:** Rename `include/secrets_example.h` to `include/secrets.h`
2. **Configure Network:** Open `secrets.h` and update your Wi-Fi credentials and your host computer's IP address:
```cpp
const char* ssid = "Your Wifi Name";
const char* password = "Your Password";
const char* ws_server_uri = "ws:<Your ip>:8000/stream";
const char* udp_server_ip = "<Your IP, ex 192.12.10.2>";
const int port = 0; //your port here

```

3. **Flash:** Compile and upload the code to your ESP32 board using PlatformIO
