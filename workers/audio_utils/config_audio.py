PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
REDIMNET_PATH = "workers/audio_utils/redimnet_b2.onnx"
VAD_PATH = "workers/audio_utils/silero_vad.onnx"
DEBUG_AUDIO = False

AUDIO_CHUNK_SIZE_MS = 160 
AUDIO_SAMPLE_RATE_HZ = 16000

CONTEXT_LEFT = 64  # 256 default both
CONTEXT_RIGHT = 64

# variables for checking if a person stopped talking
SILENT_CHUNK_THRESHOLD = 1  #how many chunks of audio_chunk_size silent to consider end of sentence

# voice diartization
SIMILARITY_THRESHOLD = 0.30