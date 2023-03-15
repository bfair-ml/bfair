from collections import defaultdict
from typing import List, Tuple, Sequence, Callable
from bfair.sensors.embedding.word import WordEmbedding


class Aggregator:
    def __call__(
        self,
        scored_tokens: List[Sequence[Tuple[str, float]]],
    ) -> Sequence[Tuple[str, float]]:
        raise NotImplementedError()


class CountAggregator(Aggregator):
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(
        self,
        scored_tokens: List[Sequence[Tuple[str, float]]],
    ) -> Sequence[Tuple[str, float]]:

        counter = defaultdict(int)

        for attributes in scored_tokens:
            for attr, _ in attributes:
                counter[attr] += 1

        max_count = max(counter.values())
        return tuple(
            (attr, score)
            for attr, count in counter.items()
            for score in (count / max_count,)
            if score > self.threshold
        )  # DO NOT CHANGE TUPLE


class ActivationAggregator(Aggregator):
    def __init__(
        self,
        threshold: float,
        activation_func: Callable[[float, float], float] = max,
    ):
        self.threshold = threshold
        self.activation_func = activation_func

    def __call__(
        self,
        scored_tokens: List[Sequence[Tuple[str, float]]],
    ) -> Sequence[Tuple[str, float]]:

        activation = defaultdict(float)
        for attributes in scored_tokens:
            for attr, activation in attributes:
                activation[attr] = self.activation_func(activation[attr], activation)

        return tuple(
            (attr, activation)
            for attr, activation in activation.items()
            if activation > self.threshold
        )  # DO NOT CHANGE TUPLE
