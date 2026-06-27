import os
from pathlib import Path
from pydantic import BaseModel, Field


class Settings(BaseModel):
    app_name: str = "AI Tutor Backend"
    default_domain: str = "Data Structures and Algorithms"
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    )

    # LLM Provider Selection
    # "auto"   → picks OpenAI if key exists, else Groq if key exists, else GGUF, else Ollama
    # "openai" → forces OpenAI
    # "groq"   → forces Groq (fastest, great free tier) ← DEFAULT
    # "gguf"   → forces local GGUF
    # "ollama" → forces Ollama
    llm_provider: str = Field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "groq").lower()
    )

    # OpenAI settings
    openai_api_key: str | None = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY") or None
    )
    openai_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )
    openai_base_url: str | None = Field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL") or None
    )

    # Groq settings (fast inference, free tier available) ← DEFAULT PROVIDER
    groq_api_key: str | None = Field(
        default_factory=lambda: os.getenv("GROQ_API_KEY") or None
    )
    groq_model: str = Field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    )

    # Local GGUF settings
    llm_model: str = Field(
        default_factory=lambda: os.getenv(
            "LLM_MODEL",
            "gemma-4-E2B-it-Q4_K_M.gguf"
        )
    )
    llm_temperature: float = Field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.7"))
    )
    llm_max_tokens: int = Field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "2048"))
    )

    # Ollama settings (fallback)
    ollama_base_url: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )


settings = Settings()
