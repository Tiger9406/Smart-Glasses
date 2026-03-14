import os

from dotenv import load_dotenv

load_dotenv()

BUFFER_DURATION = 5
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_LINK = "https://api.ai.it.ufl.edu/v1/chat/completions"
TEMPERATURE = 0.4
MAXOUTPUTTOKENS = 1000
VLM_ACTIVE = False
TOOLS_JSON_PATH = "core/config_tools_openai.json"
PROMPT_JSON_PATH = "core/config_prompts.json"
MODEL = "nemotron-3-super-120b-a12b"
VISION_MODEL = "gemma-3-27b-it"

USER_NAME = "Tiger"