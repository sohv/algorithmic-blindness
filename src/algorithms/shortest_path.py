# dijkstra and bellman-ford, instrumented to count relaxations exactly.

import heapq
import logging

LOGGER = logging.getLogger(__name__)

INFINITY = float("inf")


def dijkstra(n: int, graph: list[list[tuple[int, float]]], source: int = 0) -> dict:
    """Binary-heap Dijkstra. Each node is settled once; its outgoing edges are relaxed at that point.

    Stale heap entries are skipped rather than decreased in place, which is the standard library
    implementation and the one whose operation count is worth predicting.
    """
    distances = [INFINITY] * n
    distances[source] = 0.0
    settled = [False] * n
    heap = [(0.0, source)]

    relaxations = 0
    successful = 0
    nodes_settled = 0

    while heap:
        distance, node = heapq.heappop(heap)
        if settled[node]:
            continue
        settled[node] = True
        nodes_settled += 1

        for neighbour, weight in graph[node]:
            relaxations += 1
            if distance + weight < distances[neighbour]:
                distances[neighbour] = distance + weight
                successful += 1
                heapq.heappush(heap, (distances[neighbour], neighbour))

    return {"relaxations": relaxations, "successful_relaxations": successful, "nodes_settled": nodes_settled}


def bellman_ford(n: int, graph: list[list[tuple[int, float]]], source: int = 0) -> dict:
    """Bellman-Ford with the standard early exit when a pass changes nothing.

    The early exit is what makes the count depend on the graph rather than on V alone, and it is
    why the Dijkstra ratio is a real prediction rather than an arithmetic identity.
    """
    distances = [INFINITY] * n
    distances[source] = 0.0

    relaxations = 0
    successful = 0
    passes = 0

    for _ in range(n - 1):
        passes += 1
        changed = False
        for node in range(n):
            if distances[node] == INFINITY:
                continue
            for neighbour, weight in graph[node]:
                relaxations += 1
                if distances[node] + weight < distances[neighbour]:
                    distances[neighbour] = distances[node] + weight
                    successful += 1
                    changed = True
        if not changed:
            break

    # counted for comparability with dijkstra: the nodes this run actually reached
    nodes_settled = sum(1 for d in distances if d < INFINITY)
    return {"relaxations": relaxations, "successful_relaxations": successful, "nodes_settled": nodes_settled}


ALGORITHMS = {"dijkstra": dijkstra, "bellman_ford": bellman_ford}


def max_shortest_path_hops(n: int, graph: list[list[tuple[int, float]]], source: int = 0) -> int:
    """Most edges on any shortest path from the source, counting hops along the weighted tree.

    This is the textbook bound on Bellman-Ford's pass count (CLRS): it converges after as many
    passes as the longest shortest path has edges. It is emphatically NOT the unweighted BFS
    eccentricity — with weights spread over [1, 100] the cheapest route usually takes many cheap
    hops rather than few expensive ones, so the weighted hop count is several times the BFS depth
    and using the latter under-predicts the work by a wide margin.
    """
    distances = [INFINITY] * n
    hops = [0] * n
    distances[source] = 0.0
    heap = [(0.0, source)]
    settled = [False] * n

    while heap:
        distance, node = heapq.heappop(heap)
        if settled[node]:
            continue
        settled[node] = True
        for neighbour, weight in graph[node]:
            if distance + weight < distances[neighbour]:
                distances[neighbour] = distance + weight
                hops[neighbour] = hops[node] + 1
                heapq.heappush(heap, (distances[neighbour], neighbour))

    return max((h for node, h in enumerate(hops) if settled[node]), default=0)


def analytic_prediction(
    algorithm: str, n_nodes: int, n_edges: int, max_hops: int | None = None
) -> dict[str, tuple[float, float]] | None:
    """The textbook counts, as intervals that state how precise each formula actually is.

    Dijkstra's relaxation count is an exact identity: every reachable node is settled once and each
    of its outgoing edges relaxed once, so the total is E. The band is nominal.

    Bellman-Ford's is a genuine upper BOUND, not an expectation. The published result is that it
    converges after as many passes as the longest shortest path has edges — but that assumes an
    adversarial edge order, and sweeping in node-index order propagates several hops per pass. The
    honest textbook statement is therefore "between two passes and the bound", and the interval says
    exactly that. It is wide, and it scores poorly on sharpness as a result — which is the correct
    outcome: the textbook bound genuinely is not tight here, and fitting a constant to the measured
    counts would stop this being a textbook prediction at all.

    `successful_relaxations` is declined for both algorithms. It has no published closed form — it
    depends on the order distances happen to be improved in — and inventing one would make this
    predictor a fitted baseline wearing a textbook label.
    """
    if algorithm == "dijkstra":
        return {
            "relaxations": (n_edges * 0.98, n_edges * 1.02),
            "nodes_settled": (n_nodes * 0.98, n_nodes * 1.02),
        }

    if algorithm == "bellman_ford":
        if max_hops is None:
            return None
        # two passes is the floor: one to propagate, one to detect no change. the bound is the top.
        return {
            "relaxations": (n_edges * 2 * 0.9, n_edges * (max_hops + 1) * 1.05),
            "nodes_settled": (n_nodes * 0.98, n_nodes * 1.02),
        }

    return None
