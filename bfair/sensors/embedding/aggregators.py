from typing import List, Tuple, Sequence
from bfair.sensors.embedding.word import WordEmbedding


class Aggregator:
    def __call__(
        self,
        attributed_words: List[Tuple[str, Sequence[str]]],
        embedding: WordEmbedding,
    ) -> List[Tuple[str, Sequence[str]]]:
        raise NotImplementedError()


class CountAggregator(Aggregator):
    pass

class ActivationAggregator(Aggregator):
    pass