import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_LINK = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
TEMPERATURE = 0.4
MAXOUTPUTTOKENS = 1000
TOOLS_JSON_PATH = "api/gemini_config/config_tools.json"
PROMPT_JSON_PATH = "api/gemini_config/config_prompts.json"