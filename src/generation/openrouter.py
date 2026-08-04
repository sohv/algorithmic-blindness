# one client for every model under test.

import logging
import os
import re
import time
from dataclasses import dataclass

import requests
from openai import OpenAI

from src.generation.models import resolve

LOGGER = logging.getLogger(__name__)

BASE_URL = "https://openrouter.ai/api/v1"
MODELS_ENDPOINT = f"{BASE_URL}/models"

# pinned low: the study measures movement across prompt wordings, so sampling noise must not add to it
TEMPERATURE = 0.1
SEED = 42
MAX_TOKENS = 4096

# reasoning models wrap their working in these; the answer is what follows
THINK_TAG = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)


@dataclass
class LLMResponse:
    content: str
    model: str
    timestamp: float
    success: bool
    error: str | None = None
    # the sampling controls actually sent, since a model that does not honour one gets it withheld
    temperature: float | None = None
    seed: int | None = None
    # which upstream backend served the call, since a multi-homed model can be served by several
    provider: str | None = None
    usage: dict | None = None


def api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    assert key, "OPENROUTER_API_KEY is not set. copy .env.example to .env and fill it in"
    return key


def fetch_catalog() -> dict[str, dict]:
    """The live model list, keyed by openrouter id. Used to validate the registry before a run."""
    response = requests.get(MODELS_ENDPOINT, timeout=30)
    response.raise_for_status()
    return {entry["id"]: entry for entry in response.json()["data"]}


class OpenRouterClient:
    """Queries one model. Retries on transient failure and records the failure when it gives up."""

    def __init__(self, model_key: str, max_retries: int = 3, retry_delay: int = 5, seed: int = SEED):
        spec = resolve(model_key)
        self.model_key = model_key
        self.model_id = spec.openrouter_id
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = OpenAI(base_url=BASE_URL, api_key=api_key())
        # openrouter drops an unsupported control silently, so withhold it rather than claim it
        self.temperature = TEMPERATURE if spec.supports_temperature else None
        self.seed = seed if spec.supports_seed else None
        if self.temperature is None:
            LOGGER.warning(f"{model_key} does not honour temperature; its responses are not sampling-pinned")

    def query(self, prompt: str) -> LLMResponse:
        """Send one prompt. Returns a failed response rather than raising, so one bad call never
        discards the results of a long run."""
        error = None
        for attempt in range(1, self.max_retries + 1):
            timestamp = time.time()
            try:
                controls = {}
                if self.temperature is not None:
                    controls["temperature"] = self.temperature
                if self.seed is not None:
                    controls["seed"] = self.seed
                completion = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=MAX_TOKENS,
                    **controls,
                )
            except Exception as exception:  # noqa: BLE001 - the per-call boundary, per CLAUDE.md
                error = f"attempt {attempt}/{self.max_retries} failed: {exception}"
                LOGGER.error(f"{self.model_key}: {error}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                continue

            content = completion.choices[0].message.content or ""
            # a model that spent its whole budget thinking returns empty; that is a failure, not a refusal
            stripped = THINK_TAG.sub("", content).strip()
            if not stripped:
                error = f"attempt {attempt}/{self.max_retries}: empty response after stripping reasoning"
                LOGGER.error(f"{self.model_key}: {error}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                continue

            return LLMResponse(
                content=stripped,
                model=self.model_key,
                timestamp=timestamp,
                success=True,
                temperature=self.temperature,
                seed=self.seed,
                provider=getattr(completion, "provider", None),
                usage=completion.usage.model_dump() if completion.usage else None,
            )

        return LLMResponse(content="", model=self.model_key, timestamp=time.time(), success=False, error=error)


def test_connection(model_key: str) -> bool:
    """One trivial call, to confirm the key and the model slug work before a paid run."""
    try:
        return OpenRouterClient(model_key, max_retries=1).query("Reply with OK.").success
    except Exception:
        LOGGER.exception(f"connection test failed for {model_key}")
        return False
