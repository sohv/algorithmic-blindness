# quicksort and mergesort, instrumented to count operations exactly.

import logging
import math
import sys
from dataclasses import dataclass, field

LOGGER = logging.getLogger(__name__)

# quicksort's depth is linear in n on the degenerate inputs, not logarithmic
RECURSION_HEADROOM = 100_000


@dataclass
class Counter:
    """Operation counts for one sort. `moves` counts element writes, not swaps: a swap is two."""

    comparisons: int = 0
    moves: int = 0
    max_depth: int = 0
    _depth: int = field(default=0, repr=False)

    def enter(self) -> None:
        self._depth += 1
        self.max_depth = max(self.max_depth, self._depth)

    def leave(self) -> None:
        self._depth -= 1

    def as_metrics(self) -> dict:
        return {"comparisons": self.comparisons, "moves": self.moves, "max_depth": self.max_depth}


def quicksort(values: list[int]) -> dict:
    """Textbook quicksort with a Lomuto partition on the last element.

    That pivot choice is what makes the `sorted` and `reverse` variants quadratic — the behaviour
    the variant axis exists to expose, and the thing a predictor most often gets wrong.
    """
    data = list(values)
    counter = Counter()

    def partition(low: int, high: int) -> int:
        pivot = data[high]
        i = low - 1
        for j in range(low, high):
            counter.comparisons += 1
            if data[j] <= pivot:
                i += 1
                data[i], data[j] = data[j], data[i]
                counter.moves += 2
        data[i + 1], data[high] = data[high], data[i + 1]
        counter.moves += 2
        return i + 1

    def sort(low: int, high: int) -> None:
        counter.enter()
        if low < high:
            split = partition(low, high)
            sort(low, split - 1)
            sort(split + 1, high)
        counter.leave()

    previous_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(previous_limit, len(data) + RECURSION_HEADROOM))
    try:
        sort(0, len(data) - 1)
    finally:
        sys.setrecursionlimit(previous_limit)

    assert data == sorted(values), "quicksort did not sort its input"
    return counter.as_metrics()


def mergesort(values: list[int]) -> dict:
    """Top-down mergesort. Comparisons barely move with the input distribution, which is the
    contrast against quicksort that makes the domain informative rather than a single data point."""
    data = list(values)
    counter = Counter()

    def merge(low: int, middle: int, high: int) -> None:
        left = data[low:middle]
        right = data[middle:high]
        i = j = 0
        k = low
        while i < len(left) and j < len(right):
            counter.comparisons += 1
            if left[i] <= right[j]:
                data[k] = left[i]
                i += 1
            else:
                data[k] = right[j]
                j += 1
            counter.moves += 1
            k += 1
        while i < len(left):
            data[k] = left[i]
            counter.moves += 1
            i += 1
            k += 1
        while j < len(right):
            data[k] = right[j]
            counter.moves += 1
            j += 1
            k += 1

    def sort(low: int, high: int) -> None:
        counter.enter()
        if high - low > 1:
            middle = (low + high) // 2
            sort(low, middle)
            sort(middle, high)
            merge(low, middle, high)
        counter.leave()

    previous_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(previous_limit, len(data) + RECURSION_HEADROOM))
    try:
        sort(0, len(data))
    finally:
        sys.setrecursionlimit(previous_limit)

    assert data == sorted(values), "mergesort did not sort its input"
    return counter.as_metrics()


ALGORITHMS = {"quicksort": quicksort, "mergesort": mergesort}


def harmonic(n: int) -> float:
    return sum(1.0 / i for i in range(1, n + 1))


def analytic_prediction(algorithm: str, n: int, variant: str) -> dict[str, tuple[float, float]] | None:
    """The textbook counts, as intervals that state how precise each formula actually is.

    This is the reference a model is measured against: not a baseline that learned anything, just
    the published analysis. Each entry carries its own band rather than a shared one, because the
    formulas differ in kind - quicksort on sorted input is an exact identity, quicksort on random
    input is an expectation, and mergesort's bounds are near-tight.

    `few_unique` is declined: the standard analyses assume distinct keys, and quoting a distinct-key
    formula for duplicate-heavy input would be wrong rather than merely imprecise.
    """
    if variant == "few_unique":
        return None

    def band(centre: float, low: float, high: float) -> tuple[float, float]:
        return (centre * low, centre * high)

    if algorithm == "quicksort":
        if variant in ("sorted", "reverse"):
            # Lomuto on an ordered array is maximally unbalanced, giving the identity n(n-1)/2
            comparisons = n * (n - 1) / 2
            return {
                "comparisons": band(comparisons, 0.9, 1.02),
                "moves": band(n * (n + 1), 0.4, 1.1),
                "max_depth": band(n + 1, 0.9, 1.02),
            }
        # expectation over random permutations: 2(n+1)H_n - 4n
        comparisons = 2 * (n + 1) * harmonic(n) - 4 * n
        return {
            "comparisons": band(comparisons, 0.85, 1.2),
            "moves": band(comparisons, 0.7, 1.4),
            "max_depth": band(2 * math.log2(n) + 1, 0.6, 1.8),
        }

    if algorithm == "mergesort":
        ceil_log = math.ceil(math.log2(n))
        # a merge stops when one side empties, so ordered input hits the best case of about half n*log2(n)
        comparisons = n * ceil_log / 2 if variant in ("sorted", "reverse") else n * math.log2(n) - n + 1
        return {
            "comparisons": band(comparisons, 0.9, 1.1),
            # every element is written once per level, regardless of order
            "moves": band(n * ceil_log, 0.9, 1.1),
            "max_depth": band(ceil_log + 1, 0.9, 1.1),
        }

    return None
