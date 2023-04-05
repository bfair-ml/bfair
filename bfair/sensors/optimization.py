from bfair.methods.autogoal.ensembling.sampling import (
    LogSampler,
    SampleModel,
    LockedSampler,
)
from bfair.sensors.handler import SensorHandler, UnionMerge, IntersectionMerge
from bfair.sensors.text import (
    EmbeddingBasedSensor,
    TextTokenizer,
    TextSplitter,
    SentenceTokenizer,
    NonEmptyFilter,
    LargeEnoughFilter,
    BestScoreFilter,
    NoStopWordsFilter,
    NonNeutralWordsFilter,
    IdentityFilter,
    CountAggregator,
    ActivationAggregator,
    UnionAggregator,
    CoreferenceNERSensor,
    DBPediaSensor,
)
from autogoal.kb import Text
from autogoal.sampling import Sampler
from autogoal.search import PESearch, ConsoleLogger
from nltk.corpus import stopwords
from statistics import mean
from functools import partial

PRECISION = "precision"
RECALL = "recall"
F1 = "f1"
MACRO_F1 = "macro-f1"


def optimize(
    X_train,
    y_train,
    X_test,
    y_test,
    attributes,
    attr_cls,
    *,
    pop_size,
    search_iterations,
    evaluation_timeout,
    memory_limit,
    search_timeout,
    errors="warn",
    telegram_token=None,
    telegram_channel=None,
    telegram_title="",
    log_path=None,
    inspect=False,
    output_stream=None,
    language="english",
):
    loggers = get_loggers(
        telegram_token=telegram_token,
        telegram_channel=telegram_channel,
        telegram_title=telegram_title,
        log_path=log_path,
    )

    search = PESearch(
        generator_fn=partial(generate, language=language),
        fitness_fn=build_fn(
            X_train,
            y_train,
            Text,
            attributes,
            attr_cls,
            score_func=lambda x, y: compute_scores(compute_errors(x, y, attributes))[
                MACRO_F1
            ],
        ),
        maximize=True,
        pop_size=pop_size,
        evaluation_timeout=evaluation_timeout,
        memory_limit=memory_limit,
        search_timeout=search_timeout,
        errors=errors,
    )
    best_solution, best_fn = search.run(generations=search_iterations, logger=loggers)

    if inspect:
        y_pred, counter, scores = evaluate(
            best_solution,
            X_train,
            y_train,
            attributes,
            attr_cls,
        )
        print("Results @ Training ...", file=output_stream)
        print(counter, file=output_stream)
        print(scores, file=output_stream)
        print(y_pred, file=output_stream, flush=True)

        y_pred, counter, scores = evaluate(
            best_solution,
            X_test,
            y_test,
            attributes,
            attr_cls,
        )
        print("Results @ Testing ....", file=output_stream)
        print(counter, file=output_stream)
        print(scores, file=output_stream)
        print(y_pred, file=output_stream, flush=True)

    return best_solution, best_fn


def get_loggers(
    *,
    telegram_token=None,
    telegram_channel=None,
    telegram_title="",
    log_path=None,
):
    loggers = [ConsoleLogger()]

    if telegram_token:
        from autogoal.contrib.telegram import TelegramLogger

        telegram = TelegramLogger(
            token=telegram_token,
            name=telegram_title.upper(),
            channel=telegram_channel,
        )
        loggers.append(telegram)

    if log_path:
        from bfair.utils.autogoal import FileLogger

        file_logger = FileLogger(output_path=log_path)
        loggers.append(file_logger)

    return loggers


def evaluate(solution, X, y, attributes, attr_cls):
    handler: SensorHandler = solution.model
    y_pred = [handler.annotate(item, Text, attributes, attr_cls) for item in X]
    counter = compute_errors(y, y_pred, attributes)
    scores = compute_scores(counter)
    return y_pred, counter, scores


def generate(sampler: Sampler, language="english"):
    sampler = LogSampler(sampler)

    sensors = []

    if sampler.boolean("include-embedding-sensor"):
        sensor = get_embedding_based_sensor(sampler, language)
        sensors.append(sensor)

    if sampler.boolean("include-coreference-sensor"):
        sensor = get_coreference_ner_sensor(sampler, language)
        sensors.append(sensor)

    if sampler.boolean("include-dbpedia-sensor"):
        sensor = get_dbpedia_sensor(sampler, language)
        sensors.append(sensor)

    if len(sensors) > 1:
        merge_mode = sampler.choice(["union", "intersection"], handle="merge-mode")
        if merge_mode == "union":
            merge = UnionMerge()
        elif merge_mode == "intersection":
            merge = IntersectionMerge()
        else:
            raise ValueError(merge_mode)
    else:
        merge = None

    handler = SensorHandler(sensors, merge)
    return SampleModel(sampler, handler)


def get_embedding_based_sensor(sampler: LogSampler, language):
    prefix = "embedding-sensor."

    tokenization_pipeline, plain_mode = get_tokenization_pipeline(sampler, prefix)
    filtering_pipeline = get_filtering_pipeline(sampler, language, prefix)
    aggregation_pipeline = get_aggregation_pipeline(sampler, plain_mode, prefix)

    source = sampler.choice(
        ["word2vec", "word2vec-debiased"], handle=f"{prefix}embedding-source"
    )
    sensor = EmbeddingBasedSensor.build(
        language=language,
        source=source,
        tokenization_pipeline=tokenization_pipeline,
        filtering_pipeline=filtering_pipeline,
        aggregation_pipeline=aggregation_pipeline,
    )
    return sensor


def get_tokenization_pipeline(sampler: LogSampler, prefix=""):
    return (
        ([TextTokenizer()], True)
        if sampler.boolean(f"{prefix}plain_mode")
        else (
            [
                TextSplitter(),
                SentenceTokenizer(),
            ],
            False,
        )
    )


def get_filtering_pipeline(sampler: LogSampler, language, prefix=""):
    filtering_pipeline = []

    if sampler.boolean(f"{prefix}remove-stopwords"):
        words = stopwords.words(language)
        filter = NoStopWordsFilter(words)
        filtering_pipeline.append(filter)

    filter = get_filter(sampler, allow_none=True, prefix=f"{prefix}filter")
    filtering_pipeline.append(filter)

    filter = NonEmptyFilter()
    filtering_pipeline.append(filter)

    return filtering_pipeline


def get_filter(sampler: LogSampler, allow_none: bool, prefix: str):
    options = ["LargeEnoughFilter", "BestScoreFilter", "NonNeutralWordsFilter"]
    if allow_none:
        options.append("None")

    filter_name = sampler.choice(options, handle=f"{prefix}-filter")

    if filter_name == "LargeEnoughFilter":
        norm_threshold = sampler.continuous(
            0, 1, handle=f"{prefix}-large-norm-threshold"
        )
        return LargeEnoughFilter(norm_threshold)

    elif filter_name == "BestScoreFilter":
        relative_threshold = sampler.continuous(
            0, 1, handle=f"{prefix}-best-relative-threshold"
        )
        norm_threshold = sampler.continuous(
            0, 1, handle=f"{prefix}-best-norm-threshold"
        )
        return BestScoreFilter(
            threshold=relative_threshold,
            zero_threshold=norm_threshold,
        )

    elif filter_name == "NonNeutralWordsFilter":
        relative_threshold = sampler.continuous(
            0, 1, handle=f"{prefix}-neutral-relative-threshold"
        )
        norm_threshold = sampler.continuous(
            0, 1, handle=f"{prefix}-neutral-norm-threshold"
        )
        return NonNeutralWordsFilter(
            threshold=relative_threshold,
            zero_threshold=norm_threshold,
        )

    elif filter_name == "None" and allow_none:
        return IdentityFilter()

    else:
        raise ValueError(filter_name)


def get_aggregation_pipeline(sampler: LogSampler, plain_mode, prefix=""):
    n_iters = 1 if plain_mode else 2
    aggregation_pipeline = []

    for i in range(n_iters):
        aggregator_name = sampler.choice(
            ["CountAggregator", "ActivationAggregator", "UnionAggregator"],
            handle=f"{prefix}aggretator-{i}",
        )
        if aggregator_name == "CountAggregator":
            filter = get_filter(sampler, allow_none=True, prefix=f"{prefix}count")
            aggregator = CountAggregator(attr_filter=filter)

        if aggregator_name == "ActivationAggregator":
            filter = get_filter(sampler, allow_none=True, prefix=f"{prefix}activation")

            activation_name = sampler.choice(
                ["max", "sum", "mult"], handle=f"{prefix}activation-function"
            )
            if activation_name == "max":
                activation_func = max
            elif activation_name == "sum":
                activation_func = lambda x, y: x + y
            elif activation_name == "mult":
                activation_func = lambda x, y: x * y
            else:
                raise ValueError(activation_name)

            aggregator = ActivationAggregator(
                activation_func=activation_func, attr_filter=filter
            )

        if aggregator_name == "UnionAggregator":
            aggregator = UnionAggregator()

        aggregation_pipeline.append(aggregator)

    return aggregation_pipeline


def get_coreference_ner_sensor(sampler: LogSampler, language):
    prefix = "coreference-sensor."

    aggregator = get_aggregation_pipeline(
        sampler,
        plain_mode=True,
        prefix=prefix,
    )[0]
    sensor = CoreferenceNERSensor.build(
        language=language,
        aggregator=aggregator,
    )
    return sensor


def get_dbpedia_sensor(sampler: LogSampler, language):
    prefix = "dbpedia-sensor."

    cutoff = sampler.continuous(min=0, max=1, handle=f"{prefix}cutoff")
    aggregator = get_aggregation_pipeline(
        sampler,
        plain_mode=True,
        prefix=prefix,
    )[0]

    sensor = DBPediaSensor.build(
        language=language,
        fuzzy_cutoff=cutoff,
        aggregator=aggregator,
    )
    return sensor


def build_fn(X_test, y_test, stype, attributes, attr_cls, score_func):
    def fn(generated: SampleModel):
        handler: SensorHandler = generated.model
        y_pred = [
            handler.annotate(item, stype, attributes, attr_cls) for item in X_test
        ]
        score = score_func(y_test, y_pred)
        return score

    return fn


def compute_errors(y_test, y_pred, attributes):
    counter = {}
    for value in attributes:
        correct = 0
        spurious = 0
        missing = 0

        for true_ann, pred_ann in zip(y_test, y_pred):
            if value in true_ann and value not in pred_ann:
                missing += 1
            elif value in pred_ann and value not in true_ann:
                spurious += 1
            else:  # if true_ann or pred_ann contains values not in attributes, errors are not detected.
                correct += 1

        counter[value] = (correct, spurious, missing)
    return counter


def compute_scores(counter):
    scores = {}
    for value, (correct, spurious, missing) in counter.items():
        precision = safe_division(correct, correct + spurious)
        recall = safe_division(correct, correct + missing)
        f1 = safe_division(2 * precision * recall, precision + recall)
        scores[value] = {
            PRECISION: precision,
            RECALL: recall,
            F1: f1,
        }
    scores[MACRO_F1] = mean(group[F1] for group in scores.values())
    return scores


def safe_division(numerator, denominator, default=0):
    return numerator / denominator if denominator else default


def load(configuration, language="english", root=generate):
    sampler = LockedSampler(configuration, ensure_handle=True)
    model = root(sampler, language)
    return model.model
