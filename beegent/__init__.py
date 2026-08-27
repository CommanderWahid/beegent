"""Beegent discovery pipeline - see CLAUDE.md for architecture."""

from dotenv import load_dotenv

load_dotenv()  # a .env at repo root; never overrides anything already set in the real
# environment, so `LLM_BACKEND=x uv run ...` still wins
