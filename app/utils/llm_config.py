"""
Central LLM configuration for VoyageAI.

This file provides reusable Groq LLM instances for all agents.
Every agent should import get_llm() from here instead of creating
its own ChatGroq object.
"""

import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq


# Load environment variables from .env file
load_dotenv()


DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
DEFAULT_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.3"))
DEFAULT_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "1500"))


def validate_groq_api_key() -> None:
    """
    Validates that GROQ_API_KEY is available in environment variables.
    Raises a clear error if missing.
    """

    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Please add it to your .env file as: GROQ_API_KEY=your_api_key_here"
        )


@lru_cache(maxsize=8)
def get_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> ChatGroq:
    """
    Returns a cached ChatGroq LLM instance.

    Args:
        model: Groq model name. Uses GROQ_MODEL from .env if not provided.
        temperature: Controls creativity. Lower = more deterministic.
        max_tokens: Maximum response tokens.

    Returns:
        ChatGroq: Configured LangChain Groq chat model.
    """

    validate_groq_api_key()

    return ChatGroq(
        model=model or DEFAULT_MODEL,
        temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS,
        timeout=None,
        max_retries=2,
    )


def get_strict_llm() -> ChatGroq:
    """
    Returns a low-temperature LLM for structured or factual tasks.

    Useful for:
    - destination selection
    - budget checking
    - query parsing
    """

    return get_llm(
        model=DEFAULT_MODEL,
        temperature=0.0,
        max_tokens=DEFAULT_MAX_TOKENS,
    )


def get_creative_llm() -> ChatGroq:
    """
    Returns a slightly more creative LLM.

    Useful for:
    - itinerary generation
    - travel descriptions
    - user-facing recommendations
    """

    return get_llm(
        model=DEFAULT_MODEL,
        temperature=0.7,
        max_tokens=DEFAULT_MAX_TOKENS,
    )