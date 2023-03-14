from typing import List, Tuple, Sequence
from bfair.sensors.embedding.word import WordEmbedding


class Filter:
    def __call__(
        self,
        attributed_words: List[Tuple[str, Sequence[str]]],
        embedding: WordEmbedding,
    ) -> List[Tuple[str, Sequence[str]]]:
        raise NotImplementedError()


class NonEmptyFilter(Filter):
    def __call__(
        self,
        attributed_words: List[Tuple[str, Sequence[str]]],
        embedding: WordEmbedding,
    ) -> List[Tuple[str, Sequence[str]]]:
        return [
            (word, attributes) for word, attributes in attributed_words if attributes
        ]


class LargeEnoughFilter(Filter):
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(
        self,
        attributed_words: List[Tuple[str, Sequence[str]]],
        embedding: WordEmbedding,
    ) -> List[Tuple[str, Sequence[str]]]:
        return [
            (
                word,
                tuple(
                    attr
                    for attr in attributes
                    if embedding.u_similarity(word, attr) > self.threshold
                ),
            )
            for word, attributes in attributed_words
        ]


class RelativeDifferenceFilter(Filter):
    def __init__(self, threshold=0.75, zero_threshold=0):
        self.threshold = threshold
        self.zero_threshold = zero_threshold

    def __call__(
        self,
        attributed_words: List[Tuple[str, Sequence[str]]],
        embedding: WordEmbedding,
    ) -> List[Tuple[str, Sequence[str]]]:
        output = []
        for word, attributes in attributed_words:
            similarities = [embedding.u_similarity(word, attr) for attr in attributes]
            max_similarity = max(similarities)
            selected_attributes = [
                attr
                for attr, similarity in zip(attributes, similarities)
                if self._is_close_enough(similarity, max_similarity)
            ]
            output.append((word, selected_attributes))
        return output

    def _is_close_enough(self, similarity, max_similarity):
        return (
            max_similarity > self.zero_threshold
            and similarity / max_similarity > self.threshold
        )


class NoStopWordsFilter(Filter):
    def __init__(self, stopwords):
        self.stopwords = stopwords

    def __call__(
        self,
        attributed_words: List[Tuple[str, Sequence[str]]],
        embedding: WordEmbedding,
    ) -> List[Tuple[str, Sequence[str]]]:
        return [
            (word, attributes)
            for word, attributes in attributed_words
            if word not in self.stopwords
        ]
