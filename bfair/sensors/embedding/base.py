from autogoal.kb import SemanticType, Text
from bfair.sensors import Sensor
from bfair.sensors.embedding.tokenizers import TextTokenizer
from bfair.sensors.embedding.filters import (
    LargeEnoughFilter,
    RelativeDifferenceFilter,
    NonEmptyFilter,
)
from bfair.sensors.embedding.word import EmbeddingLoader


class EmbeddingBasedSensor(Sensor):
    def __init__(
        self, embedding, tokenization_pipeline, filtering_pipeline, aggregation_pipeline
    ):
        self.embedding = embedding
        self.tokenization_pipeline = tokenization_pipeline
        self.filtering_pipeline = filtering_pipeline
        self.aggregation_pipeline = aggregation_pipeline

    @classmethod
    def build(
        cls,
        embedding=None,
        language=None,
        source=None,
        tokenization_pipeline=(),
        filtering_pipeline=(),
        aggregation_pipeline=(),
        **kwargs,
    ):
        if embedding is None and (language is None or source is None):
            raise ValueError(
                "If no `embedding` object is given, then both `language` and `source` should be provided."
            )

        embedding = (
            EmbeddingLoader.load(language=language, source=source)
            if embedding is None
            else embedding
        )
        return cls(
            embedding,
            tokenization_pipeline,
            filtering_pipeline,
            aggregation_pipeline,
            **kwargs,
        )

    def __call__(self, text, attributes):
        tokens = [text]
        embedding = self.embedding
        attributes = tuple(attributes)

        for component in self.tokenization_pipeline:
            tokens = component(tokens)

        attributed_words = [(word, attributes) for word in tokens]

        for component in self.filtering_pipeline:
            attributed_words = component(attributed_words, embedding)

        for component in self.aggregation_pipeline:
            attributed_words = component(attributed_words, embedding)

        if len(attributed_words) != 1:
            raise RuntimeError(
                f"The `aggregation_pipeline` failed to aggregate all input. Items remaining: {len(attributed_words)}."
            )

        _, labels = attributed_words[0]
        return labels

    def _get_input_type(self) -> SemanticType:
        return Text


class DefaultEmbeddingBasedSensor(EmbeddingBasedSensor):
    @classmethod
    def build(
        cls,
        *,
        norm_threshold=0.5,
        relative_threshold=0.75,
        embedding=None,
        language=None,
        source=None,
    ):
        return super().build(
            embedding=embedding,
            language=language,
            source=source,
            tokenization_pipeline=[
                TextTokenizer(),
            ],
            filtering_pipeline=[
                RelativeDifferenceFilter(relative_threshold, norm_threshold),
                NonEmptyFilter(),
            ],
            aggregation_pipeline=[
                # TODO
            ],
        )
