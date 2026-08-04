import pytest

from src.conditions import (
    ALL_INSTANCES,
    METADATA_LEVELS,
    NAMINGS,
    Condition,
    Grid,
    PromptSpec,
    anonymous_name,
    expand_conditions,
    expand_prompt_specs,
    expand_requests,
)
from src.domains import (
    ALL_DOMAINS,
    BENCHMARK_DATASETS,
    CAUSAL,
    NATIVE_VARIANT,
    SHORTEST_PATH,
    SORTING,
    domain,
    domain_of_algorithm,
    domain_of_instance,
    max_edges,
    metric_scale,
    metrics_of,
    n_nodes,
    valid_variants,
)


def test_three_domains_span_causal_graph_and_sorting():
    assert set(ALL_DOMAINS) == {CAUSAL, SHORTEST_PATH, SORTING}


def test_causal_keeps_three_paradigms_and_drops_fci():
    algorithms = domain(CAUSAL).algorithms
    assert set(algorithms) == {"pc", "lingam", "notears"}
    # FCI is a second constraint-based method and its PAG output is not comparable to a DAG matrix
    assert "fci" not in algorithms


def test_each_non_causal_domain_has_two_algorithms():
    assert len(domain(SORTING).algorithms) == 2
    assert len(domain(SHORTEST_PATH).algorithms) == 2


def test_every_algorithm_resolves_to_exactly_one_domain():
    for name in ALL_DOMAINS:
        for algorithm in domain(name).algorithms:
            assert domain_of_algorithm(algorithm) == name
    with pytest.raises(ValueError):
        domain_of_algorithm("bogosort")


def test_every_instance_resolves_to_exactly_one_domain():
    for name in ALL_DOMAINS:
        for instance in domain(name).instances:
            assert domain_of_instance(instance) == name


def test_metrics_differ_by_domain_and_never_collide():
    assert metrics_of(CAUSAL) == ("precision", "recall", "f1", "shd")
    assert metrics_of(SORTING) == ("comparisons", "moves", "max_depth")
    assert metrics_of(SHORTEST_PATH) == ("relaxations", "successful_relaxations", "nodes_settled")
    # no metric name is shared across domains, so a scored row is never ambiguous
    distinct = {m for name in ALL_DOMAINS for m in metrics_of(name)}
    assert len(distinct) == sum(len(metrics_of(name)) for name in ALL_DOMAINS)


def test_a_benchmark_network_has_one_variant_and_a_generated_instance_has_several():
    assert valid_variants("asia") == (NATIVE_VARIANT,)
    assert len(valid_variants("synthetic_12")) == 4
    assert set(valid_variants("array_1k")) == {"random", "sorted", "reverse", "few_unique"}
    assert set(valid_variants("graph_200")) == {"sparse", "dense"}


def test_expand_conditions_covers_every_domain_by_default():
    assert {c.domain for c in expand_conditions()} == set(ALL_DOMAINS)


def test_filtering_by_algorithm_restricts_to_its_domain():
    conditions = expand_conditions(algorithms=["quicksort"])
    assert {c.domain for c in conditions} == {SORTING}
    assert {c.algorithm for c in conditions} == {"quicksort"}


def test_filtering_by_instance_restricts_to_its_domain():
    assert {c.domain for c in expand_conditions(instances=["graph_200"])} == {SHORTEST_PATH}


def test_a_variant_an_instance_cannot_take_is_skipped_not_faked():
    conditions = expand_conditions(instances=["asia", "array_1k"], variants=["native", "reverse"])
    assert {(c.instance, c.variant) for c in conditions} == {("asia", "native"), ("array_1k", "reverse")}


def test_condition_key_round_trips_the_axes():
    condition = Condition(SORTING, "array_10k", "reverse", "quicksort")
    assert condition.key == "sorting__array_10k__reverse__quicksort"
    assert condition.metrics == metrics_of(SORTING)
    assert condition.size == 10000


def test_only_causal_benchmark_networks_count_as_non_synthetic():
    # every other instance in the study is generated, so nothing about it can have been memorised
    assert not Condition(CAUSAL, "asia", "native", "pc").is_synthetic
    assert Condition(CAUSAL, "synthetic_12", "gaussian", "pc").is_synthetic
    assert Condition(SORTING, "array_1k", "random", "quicksort").is_synthetic
    assert Condition(SHORTEST_PATH, "graph_200", "sparse", "dijkstra").is_synthetic


def test_anonymous_names_are_unique_across_every_domain():
    names = [anonymous_name(instance) for instance in ALL_INSTANCES]
    assert len(set(names)) == len(names)
    for instance, name in zip(ALL_INSTANCES, names):
        assert instance.lower() not in name.lower()


def test_node_counts_are_defined_for_every_instance():
    for instance in ALL_INSTANCES:
        assert n_nodes(instance) > 0


def test_max_edges_is_the_dag_upper_bound():
    assert max_edges("asia") == 28
    assert max_edges("synthetic_12") == 66


def test_metric_scale_normalises_counts_by_their_asymptotic_bound():
    # a comparison count on 1k and on 100k elements is not the same quantity until it is divided
    small = metric_scale(SORTING, "comparisons", "array_1k")
    large = metric_scale(SORTING, "comparisons", "array_100k")
    assert large > small * 100
    assert metric_scale(CAUSAL, "shd", "asia") == 28.0
    assert metric_scale(CAUSAL, "f1", "asia") == 1.0


def test_shortest_path_scale_uses_the_edge_count_for_relaxations():
    sparse = metric_scale(SHORTEST_PATH, "relaxations", "graph_1000", "sparse")
    dense = metric_scale(SHORTEST_PATH, "relaxations", "graph_1000", "dense")
    assert dense > sparse
    assert metric_scale(SHORTEST_PATH, "nodes_settled", "graph_1000", "sparse") == 1000.0


def test_prompt_specs_cross_all_three_axes():
    specs = expand_prompt_specs(["real"], ["sparse", "full"], [1, 2])
    assert len(specs) == 4
    assert PromptSpec("real", "full", 2) in specs


def test_requests_are_one_per_condition_spec_and_model():
    conditions = expand_conditions(instances=["asia"], algorithms=["pc"])
    specs = expand_prompt_specs(["real"], ["sparse"], [1, 2])
    requests = expand_requests(conditions, specs, ["gpt", "grok"])
    assert len(requests) == 4
    assert len({r.id for r in requests}) == 4


def test_grid_counts_scored_rows_per_domain_metric_count():
    grid = Grid(
        domains=[SORTING, SHORTEST_PATH],
        instances=["array_1k", "graph_200"],
        variants=["random", "sparse"],
        namings=["real"],
        metadata_levels=["sparse"],
        formulations=[1],
        models=["gpt"],
    )
    described = grid.describe()
    # two sorting algorithms and two shortest-path algorithms, one instance and variant each
    assert described["n_conditions"] == 4
    assert described["n_conditions_by_domain"] == {SORTING: 2, SHORTEST_PATH: 2}
    # both domains have 3 metrics, so 4 conditions x 3 metrics x 1 spec x 1 model
    assert described["n_scored_rows"] == 12


def test_the_default_grid_spans_every_domain_and_prompt_cell():
    described = Grid(models=["gpt"]).describe()
    assert set(described["n_conditions_by_domain"]) == set(ALL_DOMAINS)
    assert described["n_prompt_specs"] == len(NAMINGS) * len(METADATA_LEVELS) * 3


def test_conditions_are_hashable_and_sorted():
    conditions = expand_conditions(instances=["asia", "cancer"], algorithms=["pc"])
    assert len(set(conditions)) == 2
    assert sorted(conditions) == conditions


def test_benchmark_datasets_are_a_subset_of_all_instances():
    assert set(BENCHMARK_DATASETS) <= set(ALL_INSTANCES)
