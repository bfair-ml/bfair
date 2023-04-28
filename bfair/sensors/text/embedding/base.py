from typing import Sequence, List, Set, Union
from autogoal.kb import SemanticType, Text
from bfair.sensors.base import Sensor
from bfair.sensors.text.embedding.tokenizers import (
    TextTokenizer,
    Tokenizer,
    TextSplitter,
    SentenceTokenizer,
)
from bfair.sensors.text.embedding.filters import (
    Filter,
    BestScoreFilter,
    NonEmptyFilter,
    NonNeutralWordsFilter,
)
from bfair.sensors.text.embedding.aggregators import (
    Aggregator,
    ActivationAggregator,
    UnionAggregator,
)
from bfair.sensors.text.embedding.word import EmbeddingLoader, WordEmbedding


class Level(tuple):
    pass


class EmbeddingBasedSensor(Sensor):
    def __init__(
        self,
        embedding: WordEmbedding,
        tokenization_pipeline: Sequence[Tokenizer],
        filtering_pipeline: Sequence[Filter],
        aggregation_pipeline: Sequence[Aggregator],
        restricted_to: Union[str, Set[str]] = None,
    ):
        self.embedding = embedding
        self.tokenization_pipeline = tokenization_pipeline
        self.filtering_pipeline = filtering_pipeline
        self.aggregation_pipeline = aggregation_pipeline
        super().__init__(restricted_to)

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

    @classmethod
    def _apply_over_leaves(cls, func, collection, *args, level_up=False, **kargs):
        if not isinstance(collection, Level):
            leaf = func(collection, *args, **kargs)
            return Level(leaf) if level_up else leaf
        return Level(
            cls._apply_over_leaves(func, item, *args, level_up=level_up, **kargs)
            for item in collection
        )

    @classmethod
    def _apply_in_last_level(cls, func, collection, *args, keep_level=False, **kargs):
        is_empty = not any(True for _ in collection)
        if is_empty or any(not isinstance(item, Level) for item in collection):
            leaf = func(collection, *args, **kargs)
            return Level(leaf) if keep_level else leaf
        return Level(
            cls._apply_in_last_level(func, item, *args, keep_level=keep_level, **kargs)
            for item in collection
        )

    def __call__(self, text, attributes: List[str], attr_cls: str):
        def add_attributes(token, attributes, embedding):
            return (
                token,
                [(attr, embedding.u_similarity(token, attr)) for attr in attributes],
            )

        def select_scores(attr_token):
            _, attributes = attr_token
            return attributes

        embedding = self.embedding
        attributes = tuple(attributes)

        tokens = Level([text])

        # TOKENIZATION_PIPELINE:    tokens = component(tokens)
        for component in self.tokenization_pipeline:
            tokens = self._apply_over_leaves(
                component,
                tokens,
                level_up=True,
            )

        attributed_tokens = self._apply_over_leaves(
            add_attributes, tokens, attributes, embedding
        )

        # FILTERING_PIPELINE:   attributed_words = component(attributed_words, embedding)
        for component in self.filtering_pipeline:
            attributed_tokens = self._apply_in_last_level(
                component,
                attributed_tokens,
                keep_level=True,
            )

        scored_tokens = self._apply_over_leaves(select_scores, attributed_tokens)

        # AGGREGATION_PIPELINE: scored_tokens = component(scored_tokens)
        for component in self.aggregation_pipeline:
            scored_tokens = self._apply_in_last_level(
                component,
                scored_tokens,
            )

        if len(scored_tokens) != 1:
            raise RuntimeError(
                f"The `aggregation_pipeline` failed to aggregate all input. Items remaining: {len(attributed_tokens)}."
            )

        syntesis = scored_tokens[0]
        labels = [attr for attr, _ in syntesis]
        return labels

    def _get_input_type(self) -> SemanticType:
        return Text

    @classmethod
    def build_default_in_plain_mode(
        cls,
        *,
        norm_threshold=0.05,
        relative_threshold=0.75,
        embedding=None,
        language=None,
        source=None,
    ):
        return cls.build(
            embedding=embedding,
            language=language,
            source=source,
            tokenization_pipeline=[
                TextTokenizer(),
            ],
            filtering_pipeline=[
                NonNeutralWordsFilter(relative_threshold, norm_threshold),
                NonEmptyFilter(),
            ],
            aggregation_pipeline=[
                ActivationAggregator(
                    activation_func=max,
                    attr_filter=BestScoreFilter(
                        relative_threshold,
                        norm_threshold,
                    ),
                )
            ],
        )

    @classmethod
    def build_default_in_hierarchy_mode(
        cls,
        *,
        norm_threshold=0.05,
        relative_threshold=0.75,
        embedding=None,
        language=None,
        source=None,
    ):
        return cls.build(
            embedding=embedding,
            language=language,
            source=source,
            tokenization_pipeline=[
                TextSplitter(),
                SentenceTokenizer(),
            ],
            filtering_pipeline=[
                NonNeutralWordsFilter(relative_threshold, norm_threshold),
                NonEmptyFilter(),
            ],
            aggregation_pipeline=[
                ActivationAggregator(
                    activation_func=max,
                    attr_filter=BestScoreFilter(
                        relative_threshold,
                        norm_threshold,
                    ),
                ),
                UnionAggregator(),
            ],
        )
