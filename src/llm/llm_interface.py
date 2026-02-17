#!/usr/bin/env python3
"""
Unified LLM Query Interface for Causal Discovery Meta-Knowledge Testing
========================================================================

Supports 7 LLMs:
1. GPT 5.2 (OpenAI)
2. Claude Opus 4.6 (Anthropic)
3. Gemini 2.5 Pro (Google AI)
4. Gemini 3 Pro Preview (Google AI)
5. Llama 3.3 70B (Together AI)
6. Qwen 3 32B (Together AI)
7. DeepSeek R1 0528 (Together AI)

Usage:
    from llm_interface import LLMQueryInterface

    llm = LLMQueryInterface('claude')
    response = llm.query(prompt)
"""

import os
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import json


@dataclass
class LLMResponse:
    """Container for LLM response with metadata."""
    content: str
    model: str
    timestamp: float
    success: bool
    error: Optional[str] = None


class LLMQueryInterface:
    """
    Unified interface for querying multiple LLMs.

    Handles:
    - API authentication
    - Rate limiting and retries
    - Response parsing
    - Error handling
    """

    SUPPORTED_MODELS = {
        'gpt5': 'gpt-5.2',
        'deepseek': 'deepseek-ai/DeepSeek-R1',
        'claude': 'claude-opus-4-6',
        'gemini': 'gemini-2.5-pro',
        'gemini3': 'gemini-3-pro-preview',
        'llama': 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
        'qwen': 'Qwen/Qwen3-32B'
    }

    def __init__(self, model_name: str, max_retries: int = 3, retry_delay: int = 5):
        """
        Initialize LLM query interface.

        Args:
            model_name: Name of the model ('gpt5', 'deepseek', 'claude', 'gemini', 'gemini3', 'llama', 'qwen')
            max_retries: Maximum number of retry attempts (default: 3)
            retry_delay: Delay in seconds between retries (default: 5)
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model_name} not supported. Choose from: {list(self.SUPPORTED_MODELS.keys())}")

        self.model_name = model_name
        self.model_id = self.SUPPORTED_MODELS[model_name]
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize API client
        self.client = self._initialize_client()

    def _initialize_client(self):
        """Initialize the appropriate API client based on model name."""
        if self.model_name == 'gpt5':
            return self._init_openai()
        elif self.model_name == 'deepseek':
            return self._init_together()
        elif self.model_name == 'claude':
            return self._init_anthropic()
        elif self.model_name in ['gemini', 'gemini3']:
            return self._init_google()
        elif self.model_name in ['llama', 'qwen']:
            return self._init_together()

    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        return OpenAI(api_key=api_key)

    def _init_anthropic(self):
        """Initialize Anthropic client."""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        return Anthropic(api_key=api_key)

    def _init_google(self):
        """Initialize Google AI client."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Google AI package not installed. Run: pip install google-generativeai")

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        genai.configure(api_key=api_key)
        return genai

    def _init_together(self):
        """Initialize Together AI client."""
        try:
            from together import Together
        except ImportError:
            raise ImportError("Together package not installed. Run: pip install together")

        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY environment variable not set")

        return Together(api_key=api_key)

    def query(self, prompt: str, temperature: float = 0.1) -> LLMResponse:
        """
        Query the LLM with automatic retry logic.

        Args:
            prompt: The prompt to send to the LLM
            temperature: Sampling temperature (default: 0.1 for consistency)

        Returns:
            LLMResponse object with content and metadata
        """
        for attempt in range(self.max_retries):
            try:
                timestamp = time.time()

                # Route to appropriate query method
                if self.model_name == 'gpt5':
                    content = self._query_openai(prompt, temperature)
                elif self.model_name == 'deepseek':
                    content = self._query_deepseek(prompt, temperature)
                elif self.model_name == 'claude':
                    content = self._query_claude(prompt, temperature)
                elif self.model_name in ['gemini', 'gemini3']:
                    content = self._query_gemini(prompt, temperature)
                elif self.model_name == 'llama':
                    content = self._query_llama(prompt, temperature)
                elif self.model_name == 'qwen':
                    content = self._query_qwen(prompt, temperature)

                return LLMResponse(
                    content=content,
                    model=self.model_name,
                    timestamp=timestamp,
                    success=True
                )

            except Exception as e:
                error_msg = f"Attempt {attempt + 1}/{self.max_retries} failed: {str(e)}"
                print(error_msg)

                if attempt < self.max_retries - 1:
                    print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    return LLMResponse(
                        content="",
                        model=self.model_name,
                        timestamp=time.time(),
                        success=False,
                        error=error_msg
                    )

    def _query_openai(self, prompt: str, temperature: float) -> str:
        """Query GPT 5.2 via OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1024
        )
        return response.choices[0].message.content

    def _query_deepseek(self, prompt: str, temperature: float) -> str:
        """Query DeepSeek R1 via Together AI."""
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1024
        )
        return response.choices[0].message.content

    def _query_claude(self, prompt: str, temperature: float) -> str:
        """Query Claude Opus 4.6 via Anthropic API."""
        message = self.client.messages.create(
            model=self.model_id,
            max_tokens=1024,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    def _query_gemini(self, prompt: str, temperature: float) -> str:
        """Query Gemini 2.5 Pro via Google AI."""
        model = self.client.GenerativeModel(self.model_id)

        generation_config = {
            'temperature': temperature,
            'max_output_tokens': 1024,
        }

        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response.text

    def _query_llama(self, prompt: str, temperature: float) -> str:
        """Query Llama 3.3 70B via Together AI."""
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1024
        )
        return response.choices[0].message.content

    def _query_qwen(self, prompt: str, temperature: float) -> str:
        """Query Qwen 3 32B via Together AI."""
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1024
        )
        return response.choices[0].message.content


def test_llm_connection(model_name: str) -> bool:
    """
    Test connection to a specific LLM.

    Args:
        model_name: Name of the model to test

    Returns:
        True if connection successful, False otherwise
    """
    try:
        llm = LLMQueryInterface(model_name)
        response = llm.query("Reply with 'OK' if you can see this message.")
        return response.success
    except Exception as e:
        print(f"Connection test failed for {model_name}: {e}")
        return False


if __name__ == "__main__":

    models = ['gpt5', 'deepseek', 'claude', 'gemini', 'gemini3', 'llama', 'qwen']

    for model in models:
        print(f"\nTesting {model}...")
        try:
            success = test_llm_connection(model)
            status = "SUCCESS" if success else "FAILED"
            print(f"{model}: {status}")
        except Exception as e:
            print(f"{model}: FAILED - {e}")
