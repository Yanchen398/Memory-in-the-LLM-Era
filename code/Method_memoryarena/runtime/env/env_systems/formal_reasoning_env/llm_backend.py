"""LLM backend support for the formal reasoning environment."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from google import genai
    from google.genai.types import GenerateContentConfig
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


QWEN35_CHAT_EXTRA_BODY = {
    "top_k": 20,
    "chat_template_kwargs": {"enable_thinking": False},
}


def _is_qwen_model(model_name: str) -> bool:
    return "qwen" in (model_name or "").lower()


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """Generate a chat completion."""
        raise NotImplementedError("Subclasses should implement this method")


class OpenAIBackend(LLMBackend):
    """OpenAI backend with support for custom base URLs."""
    
    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        max_tokens: int = 2000,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is required. Install with: pip install openai")
        
        self.client = OpenAI(
            api_key=api_key if api_key else os.getenv("OPENAI_API_KEY"),
            base_url=base_url if base_url else os.getenv("OPENAI_BASE_URL")
        )
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        if _is_qwen_model(self.model_name):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                temperature=0.7,
                top_p=0.8,
                presence_penalty=1.5,
                extra_body=QWEN35_CHAT_EXTRA_BODY,
                **kwargs
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_completion_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                **kwargs
            )
        return response.choices[0].message.content.strip()


class AnthropicBackend(LLMBackend):
    """Anthropic Claude backend."""
    
    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        api_key: Optional[str] = None
    ):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package is required. Install with: pip install anthropic")
        
        self.client = anthropic.Anthropic(
            api_key=api_key if api_key else os.getenv("ANTHROPIC_API_KEY")
        )
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        # Convert OpenAI-style messages to Anthropic format
        system_message = messages[0]['content'] if messages[0]['role'] == "system" else "You are a helpful assistant."
        user_messages = messages[1:] if messages[0]['role'] == "system" else messages
        
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            system=system_message,
            messages=user_messages,
            temperature=temperature if temperature is not None else self.temperature,
            **kwargs
        )
        return response.content[0].text.strip()


class OpenRouterBackend(LLMBackend):
    """OpenRouter backend for accessing multiple models."""
    
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        api_key: Optional[str] = None
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package is required. Install with: pip install openai")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key if api_key else os.environ["OPENROUTER_API_KEY"]
        )
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_completion_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            **kwargs
        )
        return response.choices[0].message.content.strip()


class GoogleGeminiBackend(LLMBackend):
    """Google Gemini backend."""
    
    def __init__(
        self,
        model_name: str = "gemini-1.5-pro",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        api_key: Optional[str] = None
    ):
        if not GOOGLE_AVAILABLE:
            raise ImportError("google-genai package is required. Install with: pip install google-genai")
        
        import google.generativeai
        google.generativeai.configure(api_key=api_key if api_key else os.getenv("GOOGLE_API_KEY"))
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = genai.Client()
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        # Convert messages to Gemini format
        prompt = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in messages])
        
        # Extract system instruction if present
        system_instruction = None
        if messages and messages[0]['role'] == 'system':
            system_instruction = [messages[0]['content']]
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature if temperature is not None else self.temperature,
                max_output_tokens=max_tokens if max_tokens is not None else self.max_tokens
            )
        )
        return response.text.strip()


def create_backend(
    backend_name: str,
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 2000,
    **kwargs
) -> LLMBackend:
    """
    Factory function to create LLM backend instances.
    
    Args:
        backend_name: Name of the backend ('openai', 'anthropic', 'openrouter', 'gemini')
        model_name: Model name to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        **kwargs: Additional backend-specific arguments
    
    Returns:
        LLMBackend instance
    """
    backend_name = backend_name.lower()
    
    if backend_name == "openai":
        return OpenAIBackend(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    elif backend_name == "anthropic":
        return AnthropicBackend(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif backend_name == "openrouter":
        return OpenRouterBackend(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif backend_name == "gemini" or backend_name == "google":
        return GoogleGeminiBackend(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            # **kwargs
        )
    else:
        raise ValueError(f"Unknown backend: {backend_name}. Supported: openai, anthropic, openrouter, gemini")
