"""Beegent discovery pipeline - see CLAUDE.md for architecture.

`.env` is loaded HERE, not in config.py, and the placement is deliberate: connectors read
their own credentials straight from os.environ, so loading must happen before any module in
the package reads a variable - whatever order they happen to be imported in. Putting it in
the package __init__ makes that guaranteed rather than an accident of config.py being
imported early.
"""

from dotenv import load_dotenv

load_dotenv()  # a .env at repo root; never overrides anything already set in the real
# environment, so `LLM_BACKEND=x uv run ...` still wins
