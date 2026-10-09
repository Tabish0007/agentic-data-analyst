import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    data_dir: str = os.getenv("DATA_DIR", "data")
    max_steps: int = int(os.getenv("MAX_STEPS", "12"))
    big_model: str = os.getenv("AGENT_BIG_MODEL", "gpt-4o")
    small_model: str = os.getenv("AGENT_SMALL_MODEL", "gpt-4o-mini")
    base_url: str | None = os.getenv("OPENAI_BASE_URL", None)
    api_key: str | None = os.getenv("OPENAI_API_KEY", None)
    verifier_base_url: str | None = os.getenv("VERIFIER_BASE_URL", None)
    verifier_model: str | None = os.getenv("VERIFIER_MODEL", None)
    router_path: str = os.getenv("ROUTER_PATH", "models/router.joblib")
    memory_path: str = os.getenv("MEMORY_PATH", "memory/failures.jsonl")
    temperature: float = 0.0

CFG = Config()
