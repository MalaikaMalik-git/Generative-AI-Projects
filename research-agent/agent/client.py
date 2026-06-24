"""
client.py
Singleton OpenAI client — import this everywhere instead of
creating a new OpenAI() in every file.
"""
from openai import OpenAI
from agent.config import OPENAI_API_KEY, OPENAI_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

__all__ = ["client", "OPENAI_MODEL"]