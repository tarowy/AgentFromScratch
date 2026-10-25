from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _require(name):
    # Fail fast with the variable's name only, never its value.
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set; add it to {PROJECT_ROOT / '.env'}")
    return value


base_url = _require("OPENAI_BASE_URL")
api_key = _require("OPENAI_API_KEY")
model = _require("CODEX_MODEL")

client = OpenAI(
    api_key= api_key,
    base_url= base_url,
)
