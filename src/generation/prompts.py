# builds a prompt from the naming, metadata-level and formulation axes.

import logging
from dataclasses import dataclass

from src.conditions import Condition, PromptSpec, anonymous_name
from src.domains import CAUSAL, SHORTEST_PATH, SORTING, max_edges, metrics_of, n_nodes, n_samples

LOGGER = logging.getLogger(__name__)

TARGET_STATEMENT = (
    "The algorithm is run {n_runs} times on independent instances drawn the same way. "
    "Your interval must contain the MEAN of a metric across those runs. "
    "It is not the spread of individual runs, and not a confidence interval on your own belief."
)

FORMAT_INSTRUCTION = """Respond with exactly these {n} lines and nothing else. No preamble, no
reasoning, no units, no commentary.

{format_block}

{bounds}"""


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    summary: str
    assumptions: tuple[str, ...]
    implementation: str


ALGORITHM_SPECS = {
    # causal discovery
    "pc": AlgorithmSpec(
        name="PC",
        summary="Constraint-based. Starts from a complete graph and removes edges whose endpoints are conditionally independent, then orients what it can.",
        assumptions=(
            "causal sufficiency: no unmeasured common causes",
            "faithfulness: no conditional independence that the graph does not imply",
            "correctly specified conditional independence test",
        ),
        implementation="causal-learn PC with default settings: Fisher-z conditional independence test, alpha = 0.05, stable skeleton search. Output is a CPDAG, so edges it cannot orient stay undirected.",
    ),
    "lingam": AlgorithmSpec(
        name="LiNGAM",
        summary="Functional causal model. Identifies a full causal order by exploiting non-Gaussianity of the disturbance terms via independent component analysis.",
        assumptions=(
            "linear structural equations",
            "NON-GAUSSIAN disturbances: this is the identifying assumption, and under Gaussian noise the model is not identifiable at all",
            "causal sufficiency",
            "acyclicity",
        ),
        implementation="lingam DirectLiNGAM with default settings. Returns a fully directed weighted adjacency matrix; any non-zero weight is read as an edge.",
    ),
    "notears": AlgorithmSpec(
        name="NOTEARS",
        summary="Continuous optimisation. Replaces the combinatorial acyclicity constraint with a smooth one and solves a regularised least-squares problem.",
        assumptions=(
            "linear structural equations",
            "acyclicity, enforced by the trace-exponential constraint",
            "no explicit noise-distribution requirement, but least squares is best matched to Gaussian noise",
        ),
        implementation="notears-pytorch notears_linear with lambda1 = 0.1. Returns a weighted adjacency matrix; any non-zero weight is read as an edge.",
    ),
    # shortest path
    "dijkstra": AlgorithmSpec(
        name="Dijkstra",
        summary="Single-source shortest paths with a binary min-heap. Settles nodes in order of distance; each node's outgoing edges are relaxed once, when it is settled.",
        assumptions=(
            "all edge weights are non-negative",
            "a node is settled at most once; stale heap entries are skipped rather than decreased in place",
        ),
        implementation="Standard binary-heap implementation. Complexity O((V + E) log V). Each reachable node is settled exactly once and each of its outgoing edges is relaxed exactly once.",
    ),
    "bellman_ford": AlgorithmSpec(
        name="Bellman-Ford",
        summary="Single-source shortest paths by repeated relaxation. Sweeps every edge on every pass, up to V-1 passes, and stops early when a pass changes nothing.",
        assumptions=(
            "negative weights are permitted, unlike Dijkstra",
            "worst case is V-1 passes, but the early exit usually stops far sooner on a random graph",
        ),
        implementation="Standard implementation with the early exit when a full pass makes no change. Complexity O(V*E) worst case. Every edge of every reached node is relaxed once per pass.",
    ),
    # sorting
    "quicksort": AlgorithmSpec(
        name="Quicksort",
        summary="Divide and conquer. Partitions around a pivot, then recurses on both sides.",
        assumptions=(
            "Lomuto partition scheme with the LAST element as pivot, with no median-of-three and no randomisation",
            "that pivot choice makes already-sorted and reverse-sorted input the quadratic worst case, not the best case",
        ),
        implementation="Textbook Lomuto partition on the last element, recursing on both partitions. Expected comparisons on distinct random input are 2(n+1)H_n - 4n, about 1.386 n log2(n). On sorted or reverse-sorted input every partition is maximally unbalanced.",
    ),
    "mergesort": AlgorithmSpec(
        name="Mergesort",
        summary="Divide and conquer. Splits in half, sorts each half, then merges.",
        assumptions=(
            "top-down recursive split with a standard two-pointer merge",
            "no run detection and no adaptation to existing order, so the input distribution barely changes the count",
        ),
        implementation="Textbook top-down mergesort. Comparisons lie between n*log2(n) - n + 1 and n*ceil(log2 n) - 2^ceil(log2 n) + 1 for every input; every element is written once per level.",
    ),
}

# what each metric means and its valid range, spelled out rather than assumed
METRIC_DEFINITIONS = {
    "precision": "correctly recovered edges as a fraction of predicted edges. In [0, 1].",
    "recall": "correctly recovered edges as a fraction of true edges. In [0, 1].",
    "f1": "harmonic mean of precision and recall. In [0, 1].",
    "shd": "Structural Hamming Distance: edge insertions, deletions and reversals needed to turn the predicted graph into the true DAG. A non-negative count, not bounded by 1. Lower is better.",
    "comparisons": "number of element-to-element comparisons performed. A non-negative count, typically of order n*log2(n).",
    "moves": "number of element writes performed. A swap counts as two writes. A non-negative count.",
    "max_depth": "maximum recursion depth reached. A non-negative count, of order log2(n) when balanced and of order n when degenerate.",
    "relaxations": "number of edge relaxation attempts, counting every edge examined. A non-negative count.",
    "successful_relaxations": "number of relaxations that actually lowered a node's distance. A non-negative count, at most the number of relaxations.",
    "nodes_settled": "number of nodes whose final distance was determined. A non-negative count, at most the node count.",
}


def dataset_label(condition: Condition, naming: str) -> str:
    """The name as it appears in the prompt. `anonymized` keeps the slot and empties the cue."""
    assert naming in ("real", "anonymized"), f"unknown naming {naming}"
    if naming == "anonymized":
        return anonymous_name(condition.instance)
    return condition.instance.replace("_", "-").title()


def sparse_block(condition: Condition, naming: str, diagnostics: dict | None) -> str:
    """The original submission's information set: name, size and type. One shape per domain, since
    "8 variables and 10000 samples" and "10000 integers" are not the same fact."""
    label = dataset_label(condition, naming)
    diagnostics = diagnostics or {}

    if condition.domain == CAUSAL:
        data_type = diagnostics.get("data_type", "continuous" if condition.is_synthetic else "discrete")
        return "\n".join(
            [
                f"Dataset: {label}",
                f"- Variables: {n_nodes(condition.instance)}",
                f"- Samples: {n_samples(condition.instance)}",
                f"- Data type: {data_type}",
            ]
        )

    if condition.domain == SORTING:
        return "\n".join(
            [
                f"Input: {label}",
                f"- Elements: {n_nodes(condition.instance)} integers",
                f"- Arrangement: {condition.variant}",
            ]
        )

    if condition.domain == SHORTEST_PATH:
        return "\n".join(
            [
                f"Graph: {label}",
                f"- Nodes: {n_nodes(condition.instance)}",
                f"- Density: {condition.variant}",
                "- Directed, with positive real edge weights",
            ]
        )

    raise ValueError(f"no sparse block for domain {condition.domain}")


def diagnostic_block(condition: Condition, diagnostics: dict) -> str:
    """Properties computable from the instance before any algorithm is run.

    Each domain's most predictive property lives here: non-Gaussianity for LiNGAM, sortedness for
    quicksort, edge count for both shortest-path algorithms. Without them the task really is
    unanswerable, which is the criticism this level exists to remove.
    """
    if condition.domain == CAUSAL:
        return "\n".join(
            [
                "Data diagnostics:",
                f"- Samples per variable: {diagnostics['samples_per_variable']:.1f}",
                f"- Mean absolute excess kurtosis: {diagnostics['mean_abs_excess_kurtosis']:.3f}",
                f"- Mean absolute skew: {diagnostics['mean_abs_skew']:.3f}",
                f"- Disturbance distribution: {diagnostics['non_gaussianity_verdict']}",
                f"- Mean absolute pairwise correlation: {diagnostics['mean_abs_correlation']:.3f}",
                f"- Maximum absolute pairwise correlation: {diagnostics['max_abs_correlation']:.3f}",
                f"- Correlation matrix condition number: {diagnostics['condition_number']:.1f}",
            ]
        )

    if condition.domain == SORTING:
        return "\n".join(
            [
                "Input diagnostics:",
                f"- Elements: {diagnostics['n_elements']}",
                f"- Distinct values: {diagnostics['n_unique_values']}",
                f"- Duplicate rate: {diagnostics['duplicate_rate']:.3f}",
                f"- Sortedness (share of adjacent pairs already in order): {diagnostics['sortedness']:.3f}",
                f"- Value range: {diagnostics['value_range']}",
            ]
        )

    if condition.domain == SHORTEST_PATH:
        return "\n".join(
            [
                "Graph diagnostics:",
                f"- Nodes: {diagnostics['n_nodes']}",
                f"- Edges: {diagnostics['n_edges']}",
                f"- Mean out-degree: {diagnostics['mean_out_degree']:.2f}",
                f"- Maximum out-degree: {diagnostics['max_out_degree']}",
                f"- Nodes with no outgoing edge: {diagnostics['n_isolated_nodes']}",
                f"- Mean edge weight: {diagnostics['mean_edge_weight']:.1f}",
                f"- Minimum edge weight: {diagnostics['min_edge_weight']:.1f}",
                f"- Most edges on any shortest path: {diagnostics['max_shortest_path_hops']}",
            ]
        )

    raise ValueError(f"no diagnostic block for domain {condition.domain}")


def structure_block(condition: Condition, diagnostics: dict) -> str:
    """Ground-truth structure, known here only because the answer is known. This level is an upper
    bound on what any predictor could have, not a realistic practitioner setting."""
    if condition.domain == CAUSAL:
        return "\n".join(
            [
                "True graph structure:",
                f"- Edges: {diagnostics['n_edges']}",
                f"- Edge density: {diagnostics['edge_density']:.3f} of the {max_edges(condition.instance)} possible",
                f"- Mean degree: {diagnostics['mean_degree']:.2f}",
                f"- Maximum in-degree: {diagnostics['max_in_degree']:.0f}",
                f"- Maximum out-degree: {diagnostics['max_out_degree']:.0f}",
            ]
        )

    if condition.domain == SORTING:
        return "\n".join(
            [
                "Additional structure:",
                f"- Mean inversions per sampled element: {diagnostics['mean_inversions_per_element']:.1f}",
                "- The array is regenerated from the same distribution for each run, so the counts vary only through that redraw.",
            ]
        )

    return "\n".join(
        [
            "Additional structure:",
            f"- Edge density: {diagnostics['edge_density']:.5f} of all ordered node pairs",
            "- The graph is regenerated from the same distribution for each run, so the counts vary only through that redraw.",
        ]
    )


def algorithm_block(algorithm: str, metadata_level: str) -> str:
    spec = ALGORITHM_SPECS[algorithm]
    if metadata_level == "sparse":
        return f"Algorithm: {spec.name}"

    lines = [f"Algorithm: {spec.name}", f"- {spec.summary}"]
    if metadata_level == "full":
        lines.append("- Assumptions:")
        lines.extend(f"    - {assumption}" for assumption in spec.assumptions)
        lines.append(f"- Implementation: {spec.implementation}")
    return "\n".join(lines)


def metric_definition_block(condition: Condition) -> str:
    lines = ["Metric definitions:"]
    lines.extend(f"- {metric}: {METRIC_DEFINITIONS[metric]}" for metric in condition.metrics)
    return "\n".join(lines)


def context_block(condition: Condition, spec: PromptSpec, diagnostics: dict | None) -> str:
    """Everything the model is told, assembled by metadata level."""
    if spec.metadata_level != "sparse":
        assert diagnostics is not None, f"metadata_level={spec.metadata_level} requires diagnostics"

    blocks = [sparse_block(condition, spec.naming, diagnostics)]
    if spec.metadata_level in ("diagnostic", "full"):
        blocks.append(diagnostic_block(condition, diagnostics))
    if spec.metadata_level == "full":
        blocks.append(structure_block(condition, diagnostics))
    blocks.append(algorithm_block(condition.algorithm, spec.metadata_level))
    if spec.metadata_level == "full":
        blocks.append(metric_definition_block(condition))
    return "\n\n".join(blocks)


def question_block(formulation: int) -> str:
    """The three wordings. Same task, same target, different framing."""
    if formulation == 1:
        return "Estimate the range each metric's mean will fall in."

    if formulation == 2:
        return (
            "Work through this before answering:\n"
            "1. Which of the algorithm's assumptions does this instance satisfy, and which does it violate?\n"
            "2. How does each violation move each metric, and by roughly how much?\n"
            "3. How much run-to-run variation does the instance size imply?\n"
            "Then give the range each metric's mean will fall in."
        )

    return (
        "Treat this as an interval estimation problem. Give the narrowest interval per metric that "
        "you would still expect to contain the mean. An interval wider than your actual uncertainty "
        "is as much of a failure as one that misses."
    )


FORMULATION_NAMES = {1: "direct", 2: "step_by_step", 3: "interval_estimation"}

# how each metric name is spelled in the requested response format
METRIC_LABELS = {"shd": "SHD", "f1": "F1"}


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def format_block(condition: Condition) -> str:
    return "\n".join(f"{metric_label(metric)}: [low, high]" for metric in condition.metrics)


def bounds_line(condition: Condition) -> str:
    """States each metric's valid range, so an out-of-range answer is the model's error and not an
    ambiguity in the question."""
    bounded = [m for m in condition.metrics if m in ("precision", "recall", "f1")]
    counts = [m for m in condition.metrics if m not in bounded]
    parts = []
    if bounded:
        parts.append(f"{', '.join(metric_label(m) for m in bounded)} are in [0, 1].")
    if counts:
        parts.append(f"{', '.join(metric_label(m) for m in counts)} are non-negative counts, not bounded by 1.")
    return " ".join(parts)


def build_prompt(condition: Condition, spec: PromptSpec, diagnostics: dict | None = None, n_runs: int = 100) -> str:
    """Assemble the full prompt for one request."""
    assert spec.formulation in FORMULATION_NAMES, f"unknown formulation {spec.formulation}"
    metrics = metrics_of(condition.domain)

    return "\n\n".join(
        [
            "You are an expert in algorithms and their empirical performance.",
            context_block(condition, spec, diagnostics),
            TARGET_STATEMENT.format(n_runs=n_runs),
            question_block(spec.formulation),
            FORMAT_INSTRUCTION.format(
                n=len(metrics), format_block=format_block(condition), bounds=bounds_line(condition)
            ),
        ]
    )
