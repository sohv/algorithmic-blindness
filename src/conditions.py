# the experiment grid: every axis of the study is defined here, once.

import itertools
from dataclasses import dataclass, field

from src.domains import (
    ALL_DOMAINS,
    BENCHMARK_DATASETS,
    CAUSAL,
    domain,
    instance_size,
    metrics_of,
    valid_variants,
)

# the anonymized arm replaces only the name, so the name is the single varying factor
NAMINGS = ("real", "anonymized")

# how much the model is told; a failure surviving to `full` cannot be blamed on a starved prompt
METADATA_LEVELS = ("sparse", "diagnostic", "full")

FORMULATIONS = (1, 2, 3)

PREDICTOR_KINDS = ("llm", "baseline", "human")

# every instance across every domain, in a stable order, so the anonymised ids are deterministic
ALL_INSTANCES = tuple(instance for name in ALL_DOMAINS for instance in domain(name).instances)


@dataclass(frozen=True, order=True)
class Condition:
    """One thing to be predicted. Ground truth is computed per condition, once."""

    domain: str
    instance: str
    variant: str
    algorithm: str

    @property
    def key(self) -> str:
        return f"{self.domain}__{self.instance}__{self.variant}__{self.algorithm}"

    @property
    def metrics(self) -> tuple[str, ...]:
        return metrics_of(self.domain)

    @property
    def is_causal(self) -> bool:
        return self.domain == CAUSAL

    @property
    def is_synthetic(self) -> bool:
        """Whether the instance was generated for this study, so no memorable statistics exist.

        Every non-causal instance is generated, which is what makes the benchmark-versus-synthetic
        contrast a causal-domain question rather than a global one.
        """
        return self.instance not in BENCHMARK_DATASETS

    @property
    def size(self) -> int:
        return instance_size(self.instance)


@dataclass(frozen=True, order=True)
class PromptSpec:
    """How a condition is asked about."""

    naming: str
    metadata_level: str
    formulation: int

    @property
    def key(self) -> str:
        return f"{self.naming}__{self.metadata_level}__f{self.formulation}"


@dataclass(frozen=True, order=True)
class Request:
    """One condition, asked one way, of one model. Exactly one api call."""

    condition: Condition
    prompt_spec: PromptSpec
    model: str

    @property
    def id(self) -> str:
        return f"{self.condition.key}__{self.prompt_spec.key}__{self.model}"


def anonymous_name(instance: str) -> str:
    """A neutral, lexically empty stand-in, assigned by position so it is deterministic.

    Carries no hint of the real instance and never collides with a real benchmark name.
    """
    return f"Instance-{ALL_INSTANCES.index(instance) + 1:02d}"


def expand_conditions(
    domains: list[str] | None = None,
    instances: list[str] | None = None,
    algorithms: list[str] | None = None,
    variants: list[str] | None = None,
) -> list[Condition]:
    """Every (domain, instance, variant, algorithm) tuple, skipping variants an instance cannot take.

    Filters compose: naming an algorithm restricts to its domain, naming an instance restricts to
    the domain that owns it. An empty result means the filters do not intersect, which is an error
    the caller should see rather than an empty run.
    """
    selected_domains = domains or list(ALL_DOMAINS)
    conditions = []

    for domain_name in selected_domains:
        spec = domain(domain_name)
        domain_algorithms = [a for a in spec.algorithms if algorithms is None or a in algorithms]
        domain_instances = [i for i in spec.instances if instances is None or i in instances]

        for instance in domain_instances:
            allowed = valid_variants(instance)
            selected = [v for v in allowed if variants is None or v in variants]
            conditions.extend(
                Condition(domain_name, instance, variant, algorithm)
                for variant in selected
                for algorithm in sorted(domain_algorithms)
            )

    return sorted(conditions)


def expand_prompt_specs(
    namings: list[str] | None = None,
    metadata_levels: list[str] | None = None,
    formulations: list[int] | None = None,
) -> list[PromptSpec]:
    return sorted(
        PromptSpec(naming, level, formulation)
        for naming, level, formulation in itertools.product(
            namings or list(NAMINGS), metadata_levels or list(METADATA_LEVELS), formulations or list(FORMULATIONS)
        )
    )


def expand_requests(conditions: list[Condition], prompt_specs: list[PromptSpec], models: list[str]) -> list[Request]:
    return [
        Request(condition, spec, model)
        for condition, spec, model in itertools.product(conditions, prompt_specs, sorted(models))
    ]


@dataclass
class Grid:
    """The axes of one experiment, so a run's scope is one object rather than eight flags."""

    domains: list[str] = field(default_factory=lambda: list(ALL_DOMAINS))
    instances: list[str] = field(default_factory=list)
    algorithms: list[str] = field(default_factory=list)
    variants: list[str] = field(default_factory=list)
    namings: list[str] = field(default_factory=lambda: list(NAMINGS))
    metadata_levels: list[str] = field(default_factory=lambda: list(METADATA_LEVELS))
    formulations: list[int] = field(default_factory=lambda: list(FORMULATIONS))
    models: list[str] = field(default_factory=list)

    def conditions(self) -> list[Condition]:
        return expand_conditions(
            self.domains or None, self.instances or None, self.algorithms or None, self.variants or None
        )

    def prompt_specs(self) -> list[PromptSpec]:
        return expand_prompt_specs(self.namings, self.metadata_levels, self.formulations)

    def requests(self) -> list[Request]:
        return expand_requests(self.conditions(), self.prompt_specs(), self.models)

    def describe(self) -> dict:
        conditions = self.conditions()
        specs = self.prompt_specs()
        # metric count varies by domain, so scored rows are counted per condition rather than assumed
        n_metric_rows = sum(len(condition.metrics) for condition in conditions)
        by_domain = {name: sum(1 for c in conditions if c.domain == name) for name in ALL_DOMAINS}
        return {
            "n_conditions": len(conditions),
            "n_conditions_by_domain": {k: v for k, v in by_domain.items() if v},
            "n_prompt_specs": len(specs),
            "n_models": len(self.models),
            "n_requests": len(conditions) * len(specs) * len(self.models),
            "n_scored_rows": n_metric_rows * len(specs) * len(self.models),
            "domains": self.domains,
            "instances": self.instances or "all",
            "algorithms": self.algorithms or "all",
            "variants": self.variants or "all",
            "namings": self.namings,
            "metadata_levels": self.metadata_levels,
            "formulations": self.formulations,
            "models": self.models,
        }
