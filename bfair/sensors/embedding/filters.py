from typing import Tuple, Sequence


class Filter:
    def __call__(
        self,
        attributed_tokens: Sequence[Tuple[str, Sequence[Tuple[str, float]]]],
    ) -> Sequence[Tuple[str, Sequence[Tuple[str, float]]]]:
        raise NotImplementedError()


class NonEmptyFilter(Filter):
    def __call__(
        self,
        attributed_tokens: Sequence[Tuple[str, Sequence[Tuple[str, float]]]],
    ) -> Sequence[Tuple[str, Sequence[Tuple[str, float]]]]:
        return [
            (token, attributes) for token, attributes in attributed_tokens if attributes
        ]


class LargeEnoughFilter(Filter):
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(
        self,
        attributed_tokens: Sequence[Tuple[str, Sequence[Tuple[str, float]]]],
    ) -> Sequence[Tuple[str, Sequence[Tuple[str, float]]]]:
        return [
            (
                token,
                [(attr, score) for attr, score in attributes if score > self.threshold],
            )
            for token, attributes in attributed_tokens
        ]


class RelativeDifferenceFilter(Filter):
    def __init__(self, threshold=0.75, zero_threshold=0):
        self.threshold = threshold
        self.zero_threshold = zero_threshold

    def __call__(
        self,
        attributed_tokens: Sequence[Tuple[str, Sequence[Tuple[str, float]]]],
    ) -> Sequence[Tuple[str, Sequence[Tuple[str, float]]]]:
        output = []
        for token, attributes in attributed_tokens:
            max_score = max(score for _, score in attributes)
            selected_attributes = [
                (attr, score)
                for attr, score in attributes
                if self._is_close_enough(score, max_score)
            ]
            output.append((token, selected_attributes))
        return output

    def _is_close_enough(self, score, max_score):
        return max_score > self.zero_threshold and score / max_score > self.threshold


class NoStopWordsFilter(Filter):
    def __init__(self, stopwords):
        self.stopwords = stopwords

    def __call__(
        self,
        attributed_tokens: Sequence[Tuple[str, Sequence[Tuple[str, float]]]],
    ) -> Sequence[Tuple[str, Sequence[Tuple[str, float]]]]:
        return [
            (word, attributes)
            for word, attributes in attributed_tokens
            if word not in self.stopwords
        ]
