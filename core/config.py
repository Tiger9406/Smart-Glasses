import os

from dotenv import load_dotenv

load_dotenv()

HOST = "0.0.0.0"
PORT = 8000

HEADER_VISION = b"\x01"
HEADER_AUDIO = b"\x02"

FPS = 30

# model for face rec
DEFAULT_ISF_MODEL = os.getenv("DEFAULT_ISF_MODEL", "Megatron")
MEGATRON_MODEL_PATH = os.getenv("MEGATRON_MODEL_PATH", "")
PIKACHU_MODEL_PATH = os.getenv("PIKACHU_MODEL_PATH", "")
ANNOTATED_OUTPUT_PATH = "./api/simulator_resources/annotated_video.mp4"

USER_NAME = "Tiger"
DEFAULT_NAME = "Unknown"
CONFIDENCE_THRESHOLD_DETECTION = 0.5
CONFIDENCE_THRESHOLD_MATCHING = 0.5


def get_model_path(model_type):
    if model_type == "Megatron":
        return MEGATRON_MODEL_PATH
    return PIKACHU_MODEL_PATH


TEST_REGISTER_IDENTITY = False
START_WITH_SAMPLE_DATA = True
SAMPLE_FACE_EMBEDDING_PATHS = {}
SAMPLE_VOICE_EMBEDDING_PATHS = {
    "Tiger": ["database/sample_data/voice_Tiger1.npy"],
}
SAVE_ANNOTATED_VID = False

BUFFER_DURATION = 5
USE_LLM = False
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_LINK = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
TEMPERATURE = 0.4
MAXOUTPUTTOKENS = 1000
VLM_ACTIVE = False
TOOLS_JSON_PATH = "core/config_tools.json"
PROMPT_JSON_PATH = "core/config_prompts.json"

# model for audio
PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"

DEBUG_AUDIO = False
# chunking
AUDIO_CHUNK_SIZE_MS = 160 
AUDIO_SAMPLE_RATE_HZ = 16000

# Streaming context, defaults used in parakeet readme
CONTEXT_LEFT = 64  # 256 default both
CONTEXT_RIGHT = 64

# variables for checking if a person stopped talking
SILENT_CHUNK_THRESHOLD = 1  #how many chunks of audio_chunk_size silent to consider end of sentence

# voice diartization
SIMILARITY_THRESHOLD = 0.55

IDENTITY_DB_PATH = "./database/identities.db"
