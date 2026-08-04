# the three algorithmic domains under test, and what performance means in each.

import logging
import math
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)

CAUSAL = "causal_discovery"
SHORTEST_PATH = "shortest_path"
SORTING = "sorting"


@dataclass(frozen=True)
class Domain:
    """One algorithmic family: what runs, what is measured, and over what instances."""

    name: str
    algorithms: tuple[str, ...]
    metrics: tuple[str, ...]
    instances: tuple[str, ...]
    variants: tuple[str, ...]
    description: str
    # metrics where a smaller value is better, so the bias sign can be oriented consistently
    lower_is_better: tuple[str, ...] = ()


# three paradigms, one algorithm each; see docs/project_design_decisions.md for why FCI is out
BENCHMARK_DATASETS = ("asia", "alarm", "sachs", "survey", "child", "cancer", "hepar2", "earthquake", "insurance")
SYNTHETIC_DATASETS = ("synthetic_12", "synthetic_30", "synthetic_50", "synthetic_60")

NATIVE_VARIANT = "native"
NOISE_VARIANTS = ("gaussian", "uniform", "laplace", "exponential")

CAUSAL_DOMAIN = Domain(
    name=CAUSAL,
    algorithms=("pc", "lingam", "notears"),
    metrics=("precision", "recall", "f1", "shd"),
    instances=BENCHMARK_DATASETS + SYNTHETIC_DATASETS,
    variants=(NATIVE_VARIANT,) + NOISE_VARIANTS,
    description="recovering a causal DAG from observational data",
    lower_is_better=("shd",),
)

# one problem, two complexity classes: O(E log V) against O(VE)
GRAPH_SIZES = {"graph_200": 200, "graph_1000": 1000, "graph_5000": 5000}
GRAPH_DENSITIES = {"sparse": 4.0, "dense": 20.0}  # mean out-degree

SHORTEST_PATH_DOMAIN = Domain(
    name=SHORTEST_PATH,
    algorithms=("dijkstra", "bellman_ford"),
    metrics=("relaxations", "successful_relaxations", "nodes_settled"),
    instances=tuple(GRAPH_SIZES),
    variants=tuple(GRAPH_DENSITIES),
    description="single-source shortest paths on a weighted directed graph",
    lower_is_better=("relaxations",),
)

# the domain whose answers are standard textbook results, so a miss cannot be blamed on difficulty
ARRAY_SIZES = {"array_1k": 1000, "array_10k": 10000, "array_100k": 100000}
ARRAY_DISTRIBUTIONS = ("random", "sorted", "reverse", "few_unique")

SORTING_DOMAIN = Domain(
    name=SORTING,
    algorithms=("quicksort", "mergesort"),
    metrics=("comparisons", "moves", "max_depth"),
    instances=tuple(ARRAY_SIZES),
    variants=ARRAY_DISTRIBUTIONS,
    description="sorting an array of integers in place or via merging",
    lower_is_better=("comparisons", "moves", "max_depth"),
)

DOMAINS = {d.name: d for d in (CAUSAL_DOMAIN, SHORTEST_PATH_DOMAIN, SORTING_DOMAIN)}
ALL_DOMAINS = tuple(DOMAINS)


def domain(name: str) -> Domain:
    assert name in DOMAINS, f"unknown domain {name}. known: {ALL_DOMAINS}"
    return DOMAINS[name]


def domain_of_algorithm(algorithm: str) -> str:
    for name, spec in DOMAINS.items():
        if algorithm in spec.algorithms:
            return name
    raise ValueError(f"unknown algorithm {algorithm}")


def domain_of_instance(instance: str) -> str:
    for name, spec in DOMAINS.items():
        if instance in spec.instances:
            return name
    raise ValueError(f"unknown instance {instance}")


def metrics_of(domain_name: str) -> tuple[str, ...]:
    return domain(domain_name).metrics


def lower_is_better(domain_name: str, metric: str) -> bool:
    return metric in domain(domain_name).lower_is_better


def valid_variants(instance: str) -> tuple[str, ...]:
    """A benchmark network has one distribution; everything else is generated, so all variants apply."""
    spec = domain(domain_of_instance(instance))
    if spec.name == CAUSAL:
        return (NATIVE_VARIANT,) if instance in BENCHMARK_DATASETS else NOISE_VARIANTS
    return spec.variants


# instance sizes, and the scales that make a width comparable across instances of different size
BENCHMARK_NODES = {
    "asia": 8,
    "cancer": 5,
    "earthquake": 5,
    "survey": 6,
    "sachs": 11,
    "child": 20,
    "insurance": 27,
    "alarm": 37,
    "hepar2": 70,
}
SYNTHETIC_NODES = {"synthetic_12": 12, "synthetic_30": 30, "synthetic_50": 50, "synthetic_60": 60}

SAMPLE_SIZES = {name: 10000 for name in ("asia", "alarm")}
SAMPLE_SIZES |= {name: 5000 for name in ("sachs", "survey", "child", "cancer", "hepar2", "earthquake", "insurance")}
SAMPLE_SIZES |= {name: 1000 for name in SYNTHETIC_NODES}


def n_nodes(instance: str) -> int:
    """Nodes for a causal network or a graph; array length for a sorting instance."""
    if instance in SYNTHETIC_NODES:
        return SYNTHETIC_NODES[instance]
    if instance in BENCHMARK_NODES:
        return BENCHMARK_NODES[instance]
    if instance in GRAPH_SIZES:
        return GRAPH_SIZES[instance]
    if instance in ARRAY_SIZES:
        return ARRAY_SIZES[instance]
    raise ValueError(f"unknown instance {instance}")


def instance_size(instance: str) -> int:
    return n_nodes(instance)


def n_samples(instance: str) -> int:
    assert instance in SAMPLE_SIZES, f"{instance} has no sample count; it is not a causal dataset"
    return SAMPLE_SIZES[instance]


def max_edges(instance: str) -> int:
    """Upper bound on edges in a DAG over these nodes, the natural scale for SHD."""
    d = n_nodes(instance)
    return d * (d - 1) // 2


def expected_edges(instance: str, variant: str) -> float:
    """Edge count of a generated graph, from its size and mean out-degree."""
    return GRAPH_SIZES[instance] * GRAPH_DENSITIES[variant]


def metric_scale(domain_name: str, metric: str, instance: str, variant: str = "") -> float:
    """Divisor putting a metric on a comparable footing across instances of very different size.

    Chosen as the quantity the metric is asymptotically proportional to, so a normalised value is
    roughly "operations per unit of the natural bound" and reads the same at every instance size.
    """
    if domain_name == CAUSAL:
        return float(max_edges(instance)) if metric == "shd" else 1.0

    if domain_name == SORTING:
        n = ARRAY_SIZES[instance]
        if metric == "max_depth":
            # recursion depth is logarithmic when balanced and linear in the degenerate case
            return float(n)
        return float(n * math.log2(n))

    if domain_name == SHORTEST_PATH:
        nodes = GRAPH_SIZES[instance]
        if metric == "nodes_settled":
            return float(nodes)
        # both counts scale with E; normalising both by it keeps the two algorithms' ratio readable
        return float(expected_edges(instance, variant)) if variant else float(nodes)

    raise ValueError(f"unknown domain {domain_name}")


def describe() -> dict:
    """Shape of the whole experiment, for the run's grid.json."""
    return {
        name: {
            "algorithms": list(spec.algorithms),
            "metrics": list(spec.metrics),
            "instances": list(spec.instances),
            "variants": list(spec.variants),
        }
        for name, spec in DOMAINS.items()
    }
