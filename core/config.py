import os


from dotenv import load_dotenv

load_dotenv()

HOST = "0.0.0.0"
PORT = 8000
SERVER_URL = f"ws://localhost:{PORT}/stream"

HEADER_VISION = b"\x01"
HEADER_AUDIO = b"\x02"

RESOLUTION = (1280, 720)
FPS = 15
FRAME_DELAY = 1.0 / FPS
TARGET_VIDEO = "./api/Friends_Clip.mp4"
TARGET_VIDEO = "./api/Friends_Clip.mp4"

# model for face rec
DEFAULT_ISF_MODEL = os.getenv("DEFAULT_ISF_MODEL", "Megatron")
MEGATRON_MODEL_PATH = os.getenv("MEGATRON_MODEL_PATH", "")
PIKACHU_MODEL_PATH = os.getenv("PIKACHU_MODEL_PATH", "")
ANNOTATED_OUTPUT_PATH = "./api/annotated_video.mp4"


def get_model_path(model_type):
    if model_type == "Megatron":
        return MEGATRON_MODEL_PATH
    return PIKACHU_MODEL_PATH


DEFAULT_NAME = "Unknown"

CONFIDENCE_THRESHOLD_DETECTION = 0.5
CONFIDENCE_THRESHOLD_MATCHING = 0.5

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2
CHUNK_SIZE = 1024
TARGET_AUDIO = "./api/MattandShaun.wav"

# model for audio
PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"

# chunking
AUDIO_CHUNK_SIZE_MS = 400  # 800 old
AUDIO_SAMPLE_RATE_HZ = 16000
# chunking
AUDIO_CHUNK_SIZE_MS = 400  # 800 old
AUDIO_SAMPLE_RATE_HZ = 16000

# Streaming context, defaults used in parakeet readme
CONTEXT_LEFT = 64  # 256 default both
CONTEXT_RIGHT = 64

# variables for checking if a person stopped talking
SPEECH_CHUNK_SIZE = 1  # each chunk is 0.8 seconds so 3 chunks means they stop speaking for 2.4 seconds to signify a sentence break
LOUDNESS_THRESHOLD = 0.01  # how quiet it needs to be to signify stop talking, can tune this when we get mic based on backround noise

# voice diartization
SIMILARITY_THRESHOLD = 0.55
