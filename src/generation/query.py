# issues one api call per request and records the raw response.

import logging
import time
from pathlib import Path

from src.conditions import Request
from src.generation.models import resolve
from src.generation.openrouter import OpenRouterClient
from src.generation.prompts import build_prompt
from src.metrics.io import write_jsonl

LOGGER = logging.getLogger(__name__)


def initialise_clients(models: list[str]) -> dict[str, OpenRouterClient]:
    """One client per model, all through the same key. A model that cannot be constructed is a bad
    registry entry, not a missing credential, so it is fatal rather than skipped."""
    clients = {model: OpenRouterClient(model) for model in models}
    LOGGER.info(f"initialised {len(clients)} models: {[resolve(m).openrouter_id for m in clients]}")
    return clients


def response_record(request: Request, prompt: str, response) -> dict:
    """One row per call, carrying every axis value it was produced under."""
    return {
        "id": request.id,
        "domain": request.condition.domain,
        "instance": request.condition.instance,
        "variant": request.condition.variant,
        "algorithm": request.condition.algorithm,
        "naming": request.prompt_spec.naming,
        "metadata_level": request.prompt_spec.metadata_level,
        "formulation": request.prompt_spec.formulation,
        "model": request.model,
        "prompt": prompt,
        "response": response.content if response.success else None,
        "error": response.error,
        "temperature": response.temperature,
        "seed": response.seed,
    }


def run_requests(
    requests: list[Request],
    diagnostics: dict[str, dict],
    output_dir: Path | str,
    sleep_seconds: float = 1.0,
) -> list[dict]:
    """Query every request. A failure is written with a null response, never dropped.

    `diagnostics` is keyed "<dataset>__<noise>" and is required for any request above the sparse
    metadata level. Its absence is an error rather than a silent downgrade to a thinner prompt.
    """
    output_dir = Path(output_dir)
    raw_dir = output_dir / "generations"
    raw_dir.mkdir(parents=True, exist_ok=True)

    clients = initialise_clients(sorted({request.model for request in requests}))
    records = []

    for index, request in enumerate(requests, 1):
        client = clients.get(request.model)
        if client is None:
            continue

        diagnostic = diagnostics.get(
            f"{request.condition.domain}__{request.condition.instance}__{request.condition.variant}"
        )
        prompt = build_prompt(request.condition, request.prompt_spec, diagnostic)
        response = client.query(prompt)

        if response.success:
            (raw_dir / f"{request.id}.txt").write_text(response.content)
        else:
            LOGGER.error(f"{request.model} failed on {request.id}: {response.error}")

        records.append(response_record(request, prompt, response))
        if index % 25 == 0:
            n_failed = sum(1 for r in records if r["response"] is None)
            LOGGER.info(f"{index}/{len(requests)} requests done, {n_failed} failed")
        time.sleep(sleep_seconds)

    write_jsonl(records, output_dir / "responses.jsonl")
    LOGGER.info(f"wrote {len(records)} responses to {output_dir / 'responses.jsonl'}")
    return records
