# config/config.py
import os

class Settings:
    VLLM_BASE_URL = os.getenv("VLLM_BASE_URL")
    VLLM_API_KEY = os.getenv("VLLM_API_KEY")
    VLLM_MODEL = os.getenv("VLLM_MODEL")

    LLM_TIMEOUT_S = int(os.getenv("LLM_TIMEOUT_S", "180"))

settings = Settings()
