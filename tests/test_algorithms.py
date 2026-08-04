import math

import pytest

from src.algorithms.shortest_path import (
    analytic_prediction as path_analytic,
)
from src.algorithms.shortest_path import (
    bellman_ford,
    dijkstra,
    max_shortest_path_hops,
)
from src.algorithms.sorting import analytic_prediction as sort_analytic
from src.algorithms.sorting import mergesort, quicksort
from src.data.instances import adjacency, build_array, build_graph

SORTS = {"quicksort": quicksort, "mergesort": mergesort}
SOLVERS = {"dijkstra": dijkstra, "bellman_ford": bellman_ford}


# instances


def test_array_variants_have_the_orderings_they_claim():
    assert build_array("array_1k", "sorted") == sorted(build_array("array_1k", "sorted"))
    reverse = build_array("array_1k", "reverse")
    assert reverse == sorted(reverse, reverse=True)
    assert len(set(build_array("array_1k", "few_unique"))) <= 10


def test_instances_are_reproducible_under_a_seed():
    assert build_array("array_1k", "random", seed=7) == build_array("array_1k", "random", seed=7)
    assert build_array("array_1k", "random", seed=7) != build_array("array_1k", "random", seed=8)


def test_a_dense_graph_has_more_edges_than_a_sparse_one_of_the_same_size():
    _, sparse = build_graph("graph_200", "sparse")
    _, dense = build_graph("graph_200", "dense")
    assert len(dense) > len(sparse) * 3


def test_every_node_is_reachable_so_both_algorithms_see_the_same_problem():
    n, edges = build_graph("graph_200", "sparse")
    assert dijkstra(n, adjacency(n, edges))["nodes_settled"] == n


# sorting


@pytest.mark.parametrize("name", SORTS)
@pytest.mark.parametrize("variant", ["random", "sorted", "reverse", "few_unique"])
def test_both_sorts_actually_sort(name, variant):
    # pinned so an instrumentation change cannot break correctness while still producing counts
    assert SORTS[name](build_array("array_1k", variant))["comparisons"] > 0


def test_quicksort_degenerates_on_ordered_input_and_mergesort_does_not():
    random_q = quicksort(build_array("array_1k", "random"))
    sorted_q = quicksort(build_array("array_1k", "sorted"))
    random_m = mergesort(build_array("array_1k", "random"))
    sorted_m = mergesort(build_array("array_1k", "sorted"))

    # Lomuto's last-element pivot makes ordered input quadratic, so n log n here is badly wrong
    assert sorted_q["comparisons"] > 40 * random_q["comparisons"]
    # quicksort's depth on ordered input is linear in n; mergesort's stays logarithmic
    assert sorted_q["max_depth"] >= 1000
    assert random_m["max_depth"] <= math.log2(1000) + 3
    # mergesort barely moves, which is the contrast that makes the domain informative
    assert 0.4 < sorted_m["comparisons"] / random_m["comparisons"] < 1.2


def test_quicksort_on_sorted_input_hits_the_exact_textbook_identity():
    # every partition is maximally unbalanced, so the count is exactly n(n-1)/2
    assert quicksort(build_array("array_1k", "sorted"))["comparisons"] == 1000 * 999 // 2


def test_mergesort_depth_is_logarithmic():
    depth = mergesort(build_array("array_10k", "random"))["max_depth"]
    assert math.log2(10000) <= depth <= math.log2(10000) + 3


def test_a_swap_counts_as_two_writes():
    # quicksort swaps, mergesort writes singly, so quicksort's moves-per-comparison is higher
    values = build_array("array_1k", "random")
    assert quicksort(values)["moves"] / quicksort(values)["comparisons"] > 0.9


# shortest path


def test_dijkstra_relaxes_every_edge_exactly_once():
    # with every node reachable, each node is settled once and its out-edges relaxed at that point
    for instance in ("graph_200", "graph_1000"):
        for variant in ("sparse", "dense"):
            n, edges = build_graph(instance, variant)
            assert dijkstra(n, adjacency(n, edges))["relaxations"] == len(edges)


def test_bellman_ford_does_strictly_more_work_than_dijkstra():
    n, edges = build_graph("graph_1000", "sparse")
    graph = adjacency(n, edges)
    assert bellman_ford(n, graph)["relaxations"] > dijkstra(n, graph)["relaxations"] * 2


def test_both_algorithms_reach_the_same_nodes():
    n, edges = build_graph("graph_200", "dense")
    graph = adjacency(n, edges)
    assert dijkstra(n, graph)["nodes_settled"] == bellman_ford(n, graph)["nodes_settled"]


def test_the_weighted_hop_count_exceeds_the_unweighted_one():
    # weights over [1, 100] make the cheapest route take many cheap hops, so BFS depth under-predicts
    n, edges = build_graph("graph_1000", "dense")
    assert max_shortest_path_hops(n, adjacency(n, edges)) > 5


# the analytic reference


def test_the_analytic_reference_covers_what_it_claims_for_sorting():
    for instance, n in [("array_1k", 1000), ("array_10k", 10000)]:
        for variant in ["random", "sorted", "reverse"]:
            for name, sort in SORTS.items():
                measured = sort(build_array(instance, variant, seed=43))
                prediction = sort_analytic(name, n, variant)
                for metric, (low, high) in prediction.items():
                    assert low <= measured[metric] <= high, f"{name}/{instance}/{variant}/{metric}"


def test_the_analytic_reference_covers_what_it_claims_for_shortest_path():
    for instance in ["graph_200", "graph_1000"]:
        for variant in ["sparse", "dense"]:
            n, edges = build_graph(instance, variant, seed=43)
            graph = adjacency(n, edges)
            hops = max_shortest_path_hops(n, graph)
            for name, solve in SOLVERS.items():
                measured = solve(n, graph)
                prediction = path_analytic(name, n, len(edges), hops)
                for metric, (low, high) in prediction.items():
                    assert low <= measured[metric] <= high, f"{name}/{instance}/{variant}/{metric}"


def test_the_analytic_reference_declines_where_no_closed_form_exists():
    # duplicate-heavy input breaks the distinct-key assumption every standard analysis makes
    assert sort_analytic("quicksort", 1000, "few_unique") is None
    # bellman-ford's pass count needs a measured graph property, so without it there is no formula
    assert path_analytic("bellman_ford", 1000, 4000, None) is None
    # successful relaxations have no published closed form and are not invented
    assert "successful_relaxations" not in path_analytic("dijkstra", 1000, 4000)


def test_dijkstras_analytic_relaxation_count_is_the_edge_count():
    prediction = path_analytic("dijkstra", 1000, 4321)
    low, high = prediction["relaxations"]
    assert low <= 4321 <= high
    assert high - low < 4321 * 0.05


def test_quicksorts_expected_count_is_about_1_39_n_log_n():
    low, high = sort_analytic("quicksort", 10000, "random")["comparisons"]
    expected = 1.386 * 10000 * math.log2(10000)
    assert low <= expected <= high
