import os

from dotenv import load_dotenv

load_dotenv()

FPS = 10
DEFAULT_ISF_MODEL = os.getenv("DEFAULT_ISF_MODEL", "Megatron")
MEGATRON_MODEL_PATH = os.getenv("MEGATRON_MODEL_PATH", "")
PIKACHU_MODEL_PATH = os.getenv("PIKACHU_MODEL_PATH", "")
ANNOTATED_OUTPUT_PATH = "./api/simulator_resources/annotated_video.mp4"

CONFIDENCE_THRESHOLD_DETECTION = 0.7
CONFIDENCE_THRESHOLD_MATCHING = 0.5

BUFFER_DURATION = 5
VLM_ACTIVE = False

def get_model_path(model_type):
    if model_type == "Megatron":
        return MEGATRON_MODEL_PATH
    return PIKACHU_MODEL_PATH