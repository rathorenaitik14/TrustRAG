import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

try:
    from google import genai
except ImportError:
    genai = None

class LLMClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generates a response given a system prompt and a user prompt."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def provider(self) -> str:
        pass

class GeminiClient(LLMClient):
    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        if genai is None:
            raise ImportError("google-genai package is not installed.")
            
        # Get from args or environment variables
        self._api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY")
        self._model_name = model_name or os.getenv("LLM_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-flash-lite-latest"
        
        if not self._api_key:
            raise ValueError(
                "Gemini API key is missing. Please set LLM_API_KEY or GEMINI_API_KEY in your environment."
            )
            
        self.client = genai.Client(api_key=self._api_key)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "Google Gemini"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import time
        # Candidate model order: requested model first, then verified fallbacks
        candidate_models = [self._model_name]
        for fb in ["gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-flash-latest"]:
            if fb not in candidate_models:
                candidate_models.append(fb)

        last_error = None
        for model in candidate_models:
            for attempt in range(2):
                try:
                    response = self.client.models.generate_content(
                        model=model,
                        contents=user_prompt,
                        config=genai.types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=0.0
                        )
                    )
                    if response and response.text:
                        self._model_name = model
                        return response.text
                except Exception as e:
                    last_error = e
                    err_lower = str(e).lower()
                    # If model is unavailable / high demand (503) or not found (404), switch model immediately
                    if any(p in err_lower for p in ["503", "unavailable", "404", "not found", "not_found", "high demand"]):
                        break
                    time.sleep(1.0)

        raise RuntimeError(f"LLM API request failed across all candidate models: {last_error}")

def get_llm_client() -> LLMClient:
    """Factory function to get the configured LLM client."""
    # For now, we default to Gemini if the key is available
    if os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY"):
        return GeminiClient()
    raise ValueError("No LLM provider configured. Set LLM_API_KEY in your environment.")
