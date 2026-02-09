import os
from dotenv import load_dotenv

load_dotenv()
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"

class Config:
    OLLAMA_MODEL = "llama3.1"
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    MAX_CONCURRENT_CREWS = 1  # Reduced to 1 to prevent system freeze with large local models
    
    # Direct params for CrewAI
    OLLAMA_TEMP = 0.1
    OLLAMA_MAX_TOKENS = 300