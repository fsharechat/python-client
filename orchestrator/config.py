import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# Use Haiku for fast, cheap intent classification — no extended thinking needed
CLAUDE_MODEL = os.getenv("ORCHESTRATOR_MODEL", "claude-haiku-4-5-20251001")

PYTHON_QA_AGENT_URL = os.getenv("PYTHON_QA_AGENT_URL", "http://localhost:8000")
JAVA_ORDER_AGENT_URL = os.getenv("JAVA_ORDER_AGENT_URL", "http://localhost:8080")

ORCHESTRATOR_PORT = int(os.getenv("ORCHESTRATOR_PORT", "8001"))
REQUEST_TIMEOUT = 30  # seconds
