"""Unified LLM provider supporting OpenAI, Groq, local GGUF, and Ollama."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage


class LLMResponse:
    """Simple wrapper to provide .content attribute."""
    def __init__(self, text: str):
        self.content = text


class LLMProvider:
    """
    Unified LLM provider with four backends:
    
    - **OpenAI** (cloud): Best quality, requires API key + costs money
    - **Groq** (cloud): Extremely fast inference, free tier available, great value
    - **GGUF** (local): Free, private, requires llama-cpp-python + model file
    - **Ollama** (local server): Free, private, requires Ollama running
    
    Selection order (when provider="auto"):
    1. OpenAI if OPENAI_API_KEY is set
    2. Groq if GROQ_API_KEY is set (fastest, great free tier)
    3. GGUF if model file exists
    4. Ollama as last resort
    """
    
    def __init__(
        self,
        provider: str = "auto",
        model_path: str = "gemma4:e2b",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        openai_api_key: str | None = None,
        openai_model: str = "gpt-4o-mini",
        openai_base_url: str | None = None,
        groq_api_key: str | None = None,
        groq_model: str = "llama-3.1-8b-instant",
        ollama_base_url: str = "http://localhost:11434",
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        verbose: bool = False,
    ):
        self.provider = provider.lower()
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.openai_base_url = openai_base_url
        self.groq_api_key = groq_api_key
        self.groq_model = groq_model
        self.ollama_base_url = ollama_base_url
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self._llm = None
        self._mode = "unknown"
        self._init_error = None
    
    def _init_llm(self):
        """Lazy-init the LLM backend."""
        if self._llm is not None:
            return
        if self._init_error is not None:
            raise RuntimeError(self._init_error)
        
        try:
            if self.provider == "openai":
                self._init_openai()
            elif self.provider == "groq":
                self._init_groq()
            elif self.provider == "gguf":
                self._init_gguf()
            elif self.provider == "ollama":
                self._init_ollama()
            elif self.provider == "auto":
                self._auto_select()
            else:
                raise ValueError(f"Unknown LLM provider: {self.provider}. Use: auto, openai, groq, gguf, ollama")
        except Exception as e:
            self._init_error = str(e)
            raise
    
    def _auto_select(self):
        """Automatically pick the best available provider."""
        errors = []
        
        # 1. Try OpenAI if key is available
        if self.openai_api_key:
            try:
                self._init_openai()
                return
            except Exception as e:
                errors.append(f"OpenAI: {e}")
        
        # 2. Try Groq if key is available (fast, great free tier)
        if self.groq_api_key:
            try:
                self._init_groq()
                return
            except Exception as e:
                errors.append(f"Groq: {e}")
        
        # 3. Try GGUF if model file exists
        gguf_path = Path(self.model_path)
        if not gguf_path.is_absolute():
            base = Path(__file__).resolve().parents[3]
            gguf_path = base / self.model_path
        if gguf_path.exists():
            try:
                self._init_gguf()
                return
            except Exception as e:
                errors.append(f"GGUF: {e}")
        
        # 4. Try Ollama as last resort
        try:
            self._init_ollama()
            return
        except Exception as e:
            errors.append(f"Ollama: {e}")
        
        raise RuntimeError(
            "No LLM backend available. Tried:\n" + "\n".join(errors) + "\n\n"
            "To fix:\n"
            "• Set OPENAI_API_KEY for OpenAI (best quality)\n"
            "• Set GROQ_API_KEY for Groq (fastest, free tier available) — get one at https://console.groq.com/keys\n"
            "• Or place a .gguf model file and set LLM_MODEL\n"
            "• Or start Ollama locally"
        )
    
    def _init_openai(self):
        """Use OpenAI API."""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise RuntimeError(
                "langchain-openai is not installed. "
                "Install it with: pip install langchain-openai"
            )
        
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Get one at https://platform.openai.com/api-keys"
            )
        
        kwargs = {
            "model": self.openai_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_key": self.openai_api_key,
        }
        if self.openai_base_url:
            kwargs["base_url"] = self.openai_base_url
        
        self._llm = ChatOpenAI(**kwargs)
        self._mode = "openai"
    
    def _init_groq(self):
        """Use Groq API for fast inference."""
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise RuntimeError(
                "langchain-groq is not installed. "
                "Install it with: pip install langchain-groq"
            )
        
        if not self.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Get a free key at https://console.groq.com/keys"
            )
        
        self._llm = ChatGroq(
            model=self.groq_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.groq_api_key,
        )
        self._mode = "groq"
    
    def _init_gguf(self):
        """Load GGUF directly via llama-cpp-python."""
        LlamaCpp = None
        import_errors = []
        
        for import_path in [
            ("langchain_community.llms", "LlamaCpp"),
            ("langchain_community.llms.llamacpp", "LlamaCpp"),
        ]:
            try:
                module = __import__(import_path[0], fromlist=[import_path[1]])
                LlamaCpp = getattr(module, import_path[1])
                break
            except ImportError as e:
                import_errors.append(f"{import_path[0]}.{import_path[1]}: {e}")
        
        if LlamaCpp is None:
            try:
                import llama_cpp
                llama_version = llama_cpp.__version__
            except ImportError:
                raise RuntimeError(
                    "llama-cpp-python is not installed.\n"
                    "Install it with: pip install llama-cpp-python\n"
                    "For GPU: CMAKE_ARGS=-DLLAMA_CUDA=on pip install llama-cpp-python"
                )
            raise RuntimeError(
                f"llama-cpp-python {llama_version} is installed, but langchain-community is missing.\n"
                f"Install it with: pip install langchain-community"
            )
        
        gguf_path = Path(self.model_path)
        if not gguf_path.is_absolute():
            base = Path(__file__).resolve().parents[3]
            gguf_path = base / self.model_path
        
        if not gguf_path.exists():
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")
        
        self._llm = LlamaCpp(
            model_path=str(gguf_path),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=self.verbose,
            n_batch=512,
            f16_kv=True,
        )
        self._mode = "gguf"
    
    def _init_ollama(self):
        """Use Ollama via ChatOllama."""
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise RuntimeError(
                "langchain-ollama is not installed. "
                "Install it with: pip install langchain-ollama"
            )
        
        self._llm = ChatOllama(
            model=self.model_path,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            base_url=self.ollama_base_url,
            timeout=60,
        )
        self._mode = "ollama"
    
    async def ainvoke(self, messages: list[BaseMessage]) -> LLMResponse:
        """Async invoke. Returns a response with .content."""
        self._init_llm()
        
        if self._mode in ("openai", "groq", "ollama"):
            # These are ChatModels that handle messages natively
            response = await self._llm.ainvoke(messages)
            return LLMResponse(response.content)
        
        # GGUF mode: convert messages to prompt string, then invoke
        prompt = self._messages_to_prompt(messages)
        import asyncio
        loop = asyncio.get_event_loop()
        
        def _run_llm():
            for method_name in ("invoke", "predict", "__call__"):
                try:
                    method = getattr(self._llm, method_name)
                    return method(prompt)
                except (AttributeError, TypeError):
                    continue
            raise RuntimeError(
                "LlamaCpp does not have invoke(), predict(), or __call__ methods. "
                "Your langchain-community version may be incompatible."
            )
        
        text = await loop.run_in_executor(None, _run_llm)
        return LLMResponse(text)
    
    def invoke(self, messages: list[BaseMessage]) -> LLMResponse:
        """Sync invoke. Returns a response with .content."""
        self._init_llm()
        
        if self._mode in ("openai", "groq", "ollama"):
            response = self._llm.invoke(messages)
            return LLMResponse(response.content)
        
        prompt = self._messages_to_prompt(messages)
        for method_name in ("invoke", "predict", "__call__"):
            try:
                method = getattr(self._llm, method_name)
                return LLMResponse(method(prompt))
            except (AttributeError, TypeError):
                continue
        raise RuntimeError("LlamaCpp has no compatible invoke method.")
    
    def _messages_to_prompt(self, messages: list[BaseMessage]) -> str:
        """Convert LangChain messages to a single prompt string for GGUF."""
        system_parts = []
        user_parts = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_parts.append(msg.content)
            elif isinstance(msg, HumanMessage):
                user_parts.append(msg.content)
            else:
                user_parts.append(str(msg.content))
        
        system = "\n\n".join(system_parts)
        user = "\n\n".join(user_parts)
        
        if system and user:
            return f"{system}\n\n### User:\n{user}\n\n### Assistant:\n"
        elif user:
            return f"### User:\n{user}\n\n### Assistant:\n"
        elif system:
            return f"{system}\n\n### Assistant:\n"
        return ""
    
    @property
    def mode(self) -> str:
        """Returns 'openai', 'groq', 'gguf', 'ollama', or 'unknown'."""
        return self._mode
    
    @property
    def init_error(self) -> str | None:
        """Returns the initialization error message, if any."""
        return self._init_error
