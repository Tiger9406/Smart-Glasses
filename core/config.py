HOST = "0.0.0.0"
PORT = 8000

HEADER_VISION = b"\x01"
HEADER_AUDIO = b"\x02"

USER_NAME = "Tiger"
DEFAULT_NAME = "Unknown"

TEST_REGISTER_IDENTITY = False
START_WITH_SAMPLE_DATA = True
SAMPLE_FACE_EMBEDDING_PATHS = {}
SAMPLE_VOICE_EMBEDDING_PATHS = {
    "Tiger": ["database/sample_data/voice_Tiger1.npy"],
}
SAVE_ANNOTATED_VID = False

IDENTITY_DB_PATH = "./database/identities.db"
