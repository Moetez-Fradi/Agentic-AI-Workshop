import os

from dotenv import load_dotenv

load_dotenv()

# Default backend - tried first.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Fallback backend - used automatically if a Gemini call fails.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

MAX_TURNS = int(os.environ.get("MAX_TURNS", "10"))

SYSTEM_PROMPT = (
    "You are a helpful local terminal coding agent. Perform tasks step by "
    "step using tools. Always test your written code using run_cmd. Keep "
    "your reasoning brief."
)

SAMPLE_PROMPT = (
    "Write a Python script called calculator.py with add, subtract, multiply, "
    "and divide functions (divide should handle division by zero gracefully). "
    "Add a few example calls at the bottom that print their results, then run "
    "the script with run_cmd to confirm everything works."
)
