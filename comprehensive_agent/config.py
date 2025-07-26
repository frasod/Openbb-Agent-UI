import os
from openai import OpenAI
import httpx

class Settings:
    # LLM Provider: "openai", "openrouter", or "ollama"
    llm_provider = os.getenv("LLM_PROVIDER", "openai")
    
    # OpenAI Settings
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # OpenRouter Settings
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
    openrouter_base_url = "https://openrouter.ai/api/v1"
    
    # Ollama Settings
    ollama_model = os.getenv("OLLAMA_MODEL", "gemma2:9b")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Server and Timeout Settings
    server_port = int(os.getenv("SERVER_PORT", 7777))
    connection_timeout = float(os.getenv("CONNECTION_TIMEOUT", 10.0))
    read_timeout = float(os.getenv("READ_TIMEOUT", 30.0))

# Create a single settings instance
settings = Settings()

# --- Centralized LLM Client ---
# Provides a single, configured client for the entire application.
# This avoids re-initializing the client in multiple places.
openai_client = None

if settings.llm_provider == "openai":
    if not settings.openai_api_key:
        raise ValueError("LLM_PROVIDER is 'openai', but OPENAI_API_KEY is not set.")
    openai_client = OpenAI(api_key=settings.openai_api_key)

elif settings.llm_provider == "openrouter":
    if not settings.openrouter_api_key:
        raise ValueError("LLM_PROVIDER is 'openrouter', but OPENROUTER_API_KEY is not set.")
    openai_client = OpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url
    )

elif settings.llm_provider == "ollama":
    # For Ollama, we'll use httpx for direct API calls since it doesn't use OpenAI format
    pass  # Ollama client will be handled separately in model_engine.py

else:
    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}. Must be 'openai', 'openrouter', or 'ollama'.")

def get_current_model():
    """Return the current model name based on the provider."""
    if settings.llm_provider == "openai":
        return settings.openai_model
    elif settings.llm_provider == "openrouter":
        return settings.openrouter_model
    elif settings.llm_provider == "ollama":
        return settings.ollama_model
    return "unknown" 