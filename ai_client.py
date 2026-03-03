"""
═══════════════════════════════════════════════════════════════════════════════
  AI Client — Thin wrapper around Google Generative AI SDK
  ─────────────────────────────────────────────────────────────────────────
  Handles API key configuration, model selection, and error handling.
  Uses free tier via Google AI Studio.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import google.generativeai as genai
from dotenv import load_dotenv

# Load .env file
load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# CLIENT
# ══════════════════════════════════════════════════════════════════════════════

MODEL_NAME = "gemini-2.5-flash"


def init_ai() -> None:
    """Configure the SDK with the API key from environment."""
    api_key = os.getenv("AI_API_KEY", "")
    if not api_key:
        raise AIError(
            "❌ API key not found. Please set AI_API_KEY in your .env file."
        )
    genai.configure(api_key=api_key)


def call_ai(
    prompt: str,
    system_instruction: str = "",
    temperature: float = 0.2,
    max_output_tokens: int = 4096,
) -> str:
    """
    Send a prompt to the AI model and return the text response.

    Args:
        prompt:             User-facing prompt text.
        system_instruction: System-level instructions (persona, constraints).
        temperature:        0.0 = deterministic, 1.0 = creative.
        max_output_tokens:  Max response length.

    Returns:
        The model's text response.

    Raises:
        AIError: On API failures (invalid key, rate limit, network).
    """
    try:
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=system_instruction or None,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        response = model.generate_content(prompt)

        # Handle blocked / empty responses
        if not response.candidates:
            raise AIError(
                "AI returned no candidates. The prompt may have been "
                "blocked by safety filters. Try rephrasing your strategy."
            )

        text = response.text
        if not text or not text.strip():
            raise AIError("AI returned an empty response.")

        return text.strip()

    except AIError:
        raise  # re-raise our own errors
    except Exception as e:
        error_msg = str(e).lower()
        if "api_key" in error_msg or "401" in error_msg or "invalid" in error_msg:
            raise AIError(
                "❌ AI service authentication failed. Please check the API key."
            )
        elif "429" in error_msg or "quota" in error_msg or "rate" in error_msg:
            raise AIError(
                "⏳ Rate limit reached. Please wait a moment and try again."
            )
        elif "timeout" in error_msg or "network" in error_msg:
            raise AIError(
                "🌐 Network error connecting to AI service. "
                "Check your internet connection and try again."
            )
        else:
            raise AIError(f"AI API error: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# EXCEPTIONS
# ══════════════════════════════════════════════════════════════════════════════

class AIError(Exception):
    """Raised when an AI API call fails."""
    pass
