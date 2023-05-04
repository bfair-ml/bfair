from collections import defaultdict
from typing import Tuple, Sequence, Callable
from bfair.sensors.text.embedding.filters import Filter, LargeEnoughFilter


class Aggregator:
    def __call__(
        self,
        scored_tokens: Sequence[Sequence[Tuple[str, float]]],
    ) -> Sequence[Tuple[str, float]]:
        raise NotImplementedError()

    def __init__(
        self,
        *,
        attr_filter: Filter = None,
        threshold: float = None,
    ):
        if attr_filter is not None and threshold is not None:
            raise ValueError(
                "Only one between `attr_filter` and `threshold` should be provided."
            )

        self.filter = attr_filter or LargeEnoughFilter(threshold or 0.05)

    def do_filter(
        self,
        attributes: Sequence[Tuple[str, float]],
    ) -> Sequence[Tuple[str, float]]:

        batch = [("<BATCH>", attributes)]
        filtered_batch = self.filter(batch)
        _, content = filtered_batch[0]
        return content


class CountAggregator(Aggregator):
    def __call__(
        self,
        scored_tokens: Sequence[Sequence[Tuple[str, float]]],
    ) -> Sequence[Tuple[str, float]]:

        counter = defaultdict(int)

        for attributes in scored_tokens:
            for attr, _ in attributes:
                counter[attr] += 1

        if not len(counter):
            return []

        max_count = max(counter.values())
        scored_attributes = [
            (attr, count / max_count) for attr, count in counter.items()
        ]
        selected_attributes = self.do_filter(scored_attributes)
        return selected_attributes


class ActivationAggregator(Aggregator):
    def __init__(
        self,
        *,
        activation_func: Callable[[float, float], float] = max,
        attr_filter: Filter = None,
        threshold: float = None,
    ):

        self.activation_func = activation_func
        super().__init__(attr_filter=attr_filter, threshold=threshold)

    def __call__(
        self,
        scored_tokens: Sequence[Sequence[Tuple[str, float]]],
    ) -> Sequence[Tuple[str, float]]:

        activation = defaultdict(float)
        for attributes in scored_tokens:
            for attr, score in attributes:
                activation[attr] = self.activation_func(activation[attr], score)

        selected_attributes = self.do_filter(activation.items())
        return selected_attributes


class UnionAggregator(ActivationAggregator):
    def __init__(
        self,
    ):
        super().__init__(activation_func=max, threshold=float("-inf"))


class SumAggregator(ActivationAggregator):
    def __init__(
        self,
        *,
        attr_filter: Filter = None,
        threshold: float = None,
    ):
        super().__init__(
            activation_func=self.add,
            attr_filter=attr_filter,
            threshold=threshold,
        )

    def add(self, x, y):
        return x + y


class VotingAggregator(Aggregator):
    def __call__(
        self, scored_tokens: Sequence[Sequence[Tuple[str, float]]]
    ) -> Sequence[Tuple[str, float]]:

        total = 0
        total_votes = defaultdict(int)

        for attributes in scored_tokens:
            total += 1
            for attr, _ in attributes:
                total_votes[attr] += 1

        if not total:
            return []

        weights = [(attr, votes / total) for attr, votes in total_votes.items()]
        selected_attributes = self.do_filter(weights)
        return selected_attributes
