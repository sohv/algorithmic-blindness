# the models under test, addressed through openrouter.

import logging
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)

# tier is a claim about relative standing, so it is checked by separation rather than a fixed price
MIN_TIER_SEPARATION_USD_PER_MTOK = 1.0

TIERS = ("frontier", "mid")


@dataclass(frozen=True)
class ModelSpec:
    """One model under test. `key` is what appears in configs, filenames and the paper's tables."""

    key: str
    openrouter_id: str
    tier: str
    label: str
    # openrouter drops an unsupported sampling control silently, so what each model honours is declared
    supports_temperature: bool = True
    supports_seed: bool = True

    def __post_init__(self):
        assert self.tier in TIERS, f"{self.key}: unknown tier {self.tier}"


# pairs with haiku: same lab, same controls, so the tier gap is not confounded with lineage
FRONTIER_MODELS = [
    ModelSpec("opus", "anthropic/claude-opus-5", "frontier", "Claude Opus 5", supports_seed=False),
]

# six labs, so a shared failure across them cannot be explained by shared training
MID_TIER_MODELS = [
    ModelSpec("gemini", "google/gemini-3.6-flash", "mid", "Gemini 3.6 Flash"),
    ModelSpec("gpt", "openai/gpt-5.6-terra", "mid", "GPT-5.6 Terra", supports_temperature=False),
    ModelSpec("grok", "x-ai/grok-4.5", "mid", "Grok 4.5"),
    ModelSpec("qwen", "qwen/qwen3.8-max", "mid", "Qwen3.8 Max"),
    ModelSpec("haiku", "anthropic/claude-haiku-4.5", "mid", "Claude Haiku 4.5", supports_seed=False),
    ModelSpec("deepseek", "deepseek/deepseek-v4-pro", "mid", "DeepSeek V4 Pro"),
]

MODELS = FRONTIER_MODELS + MID_TIER_MODELS
MODELS_BY_KEY = {model.key: model for model in MODELS}
ALL_MODELS = [model.key for model in MODELS]


def resolve(key: str) -> ModelSpec:
    assert key in MODELS_BY_KEY, f"unknown model {key}. known: {ALL_MODELS}"
    return MODELS_BY_KEY[key]


def openrouter_id(key: str) -> str:
    return resolve(key).openrouter_id


def tier_of(key: str) -> str:
    return resolve(key).tier


def tier_separation(priced: dict[str, dict]) -> list[str]:
    """Every frontier model must cost more per output token than every mid-tier one.

    An absolute threshold dates badly: a lab can reprice its flagship below last year's cutoff
    without becoming a smaller model. Separation asks the question the tier label actually makes,
    which is whether the two groups are still ordered the way the design claims.
    """
    frontier = {k: v["output_usd_per_mtok"] for k, v in priced.items() if v["declared_tier"] == "frontier"}
    mid = {k: v["output_usd_per_mtok"] for k, v in priced.items() if v["declared_tier"] == "mid"}
    if not frontier or not mid:
        return []
    ceiling = max(mid.values())
    return [
        f"{key}: declared frontier at ${price:.2f}/Mtok out, not clear of the mid-tier top of ${ceiling:.2f}"
        for key, price in frontier.items()
        if price < ceiling + MIN_TIER_SEPARATION_USD_PER_MTOK
    ]


def check_against_catalog(catalog: dict[str, dict]) -> dict:
    """Compare the registry against openrouter's live model list.

    `catalog` maps an openrouter id to its entry, as returned by `GET /api/v1/models`. Reports
    missing ids and any model whose declared tier disagrees with its actual price, so a stale slug or
    a repricing is caught before a run rather than after.
    """
    missing, unsupported, priced = [], [], {}

    for model in MODELS:
        entry = catalog.get(model.openrouter_id)
        if entry is None:
            missing.append(model.openrouter_id)
            continue

        pricing = entry.get("pricing", {})
        # openrouter quotes usd per token as a string; the paper reports per million
        input_price = float(pricing.get("prompt", 0)) * 1e6
        output_price = float(pricing.get("completion", 0)) * 1e6
        supported = entry.get("supported_parameters") or []
        priced[model.key] = {
            "openrouter_id": model.openrouter_id,
            "label": model.label,
            "declared_tier": model.tier,
            "input_usd_per_mtok": input_price,
            "output_usd_per_mtok": output_price,
            "context_length": entry.get("context_length"),
            "supports_temperature": "temperature" in supported,
            "supports_seed": "seed" in supported,
        }
        for control in ("temperature", "seed"):
            declared = getattr(model, f"supports_{control}")
            if declared != (control in supported):
                unsupported.append(
                    f"{model.key}: declares supports_{control}={declared}, catalog says {control in supported}"
                )

    mismatched = tier_separation(priced)

    return {
        "n_models": len(MODELS),
        "n_found": len(priced),
        "missing_ids": missing,
        "tier_mismatches": mismatched,
        "control_mismatches": unsupported,
        "by_model": priced,
        "ok": not missing and not mismatched and not unsupported,
    }
