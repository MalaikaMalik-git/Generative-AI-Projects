import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6").strip()
MAX_TOKENS = 700
TEMPERATURE = 0.4

if not API_KEY:
    raise ValueError("ANTHROPIC_API_KEY is missing. Add it to your .env file.")