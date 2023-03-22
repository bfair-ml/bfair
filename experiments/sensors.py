import argparse

from bfair.datasets import load_review
from bfair.datasets.reviews import REVIEW_COLUMN, GENDER_COLUMN
from bfair.methods.autogoal.ensembling.sampling import LogSampler, SampleModel
from bfair.sensors import (
    SensorHandler,
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
    P_GENDER,
)
from autogoal.kb import Text
from autogoal.sampling import Sampler
from autogoal.search import PESearch, ConsoleLogger
from nltk.corpus import stopwords
from statistics import mean


GENDER_VALUES = ["male", "female"]
PRECISION = "precision"
RECALL = "recall"
F1 = "f1"
MACRO_F1 = "macro-f1"


def run_all():
    dataset = load_review(split_seed=None)
    sensor = EmbeddingBasedSensor.build_default_in_hierarchy_mode(
        language="english", source="word2vec-debiased"
    )
    handler = SensorHandler(sensors=[sensor])
    reviews = dataset.data[REVIEW_COLUMN]
    predicted = []
    for text in reviews:
        annotations = handler.annotate(text, Text, GENDER_VALUES)
        predicted.append(annotations)
    gold = dataset.data[GENDER_COLUMN]
    errors = compute_errors(gold, predicted, GENDER_VALUES)
    scores = compute_scores(errors)
    print(scores)


def setup():
    parser = argparse.ArgumentParser()

    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--memory", type=int, default=4)
    parser.add_argument("--popsize", type=int, default=50)
    parser.add_argument("--global-timeout", type=int, default=60 * 60)
    parser.add_argument("--token", default=None)
    parser.add_argument("--channel", default=None)
    parser.add_argument("--output", default="")
    parser.add_argument("--title", default=None)

    return parser.parse_args()


def optimize(
    *,
    pop_size,
    search_iterations,
    memory_limit,
    search_timeout,
    errors="warn",
    telegram_token=None,
    telegram_channel=None,
    telegram_title="",
    log_path=None,
):
    dataset = load_review(split_seed=0)
    X_train = dataset.data[REVIEW_COLUMN]
    y_train = dataset.data[GENDER_COLUMN]

    loggers = get_loggers(
        telegram_token=telegram_token,
        telegram_channel=telegram_channel,
        telegram_title=telegram_title,
        log_path=log_path,
    )

    search = PESearch(
        generator_fn=generate,
        fitness_fn=build_fn(
            X_train,
            y_train,
            Text,
            GENDER_VALUES,
            P_GENDER,
            score_func=lambda x, y: compute_scores(compute_errors(x, y, GENDER_VALUES))[
                MACRO_F1
            ],
        ),
        maximize=True,
        pop_size=pop_size,
        memory_limit=memory_limit,
        search_timeout=search_timeout,
        errors=errors,
    )
    best_solution, best_fn = search.run(generations=search_iterations, logger=loggers)
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


def generate(sampler: Sampler, language="english"):
    sampler = LogSampler(sampler)

    tokenization_pipeline, plain_mode = get_tokenization_pipeline(sampler)
    filtering_pipeline = get_filtering_pipeline(sampler, language)
    aggregation_pipeline = get_aggregation_pipeline(sampler, plain_mode)

    source = sampler.choice(
        ["word2vec", "word2vec-debiased"], handle="embedding-source"
    )
    sensor = EmbeddingBasedSensor.build(
        language=language,
        source=source,
        tokenization_pipeline=tokenization_pipeline,
        filtering_pipeline=filtering_pipeline,
        aggregation_pipeline=aggregation_pipeline,
    )
    handler = SensorHandler([sensor])
    return SampleModel(sampler, handler)


def get_tokenization_pipeline(sampler: LogSampler):
    return (
        ([TextTokenizer()], True)
        if sampler.boolean("plain_mode")
        else (
            [
                TextSplitter(),
                SentenceTokenizer(),
            ],
            False,
        )
    )


def get_filtering_pipeline(sampler: LogSampler, language):
    filtering_pipeline = []

    if sampler.boolean("remove-stopwords"):
        words = stopwords.words(language)
        filter = NoStopWordsFilter(words)
        filtering_pipeline.append(filter)

    filter = get_filter(sampler, allow_none=True, prefix="filter")
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


def get_aggregation_pipeline(sampler: LogSampler, plain_mode):
    n_iters = 1 if plain_mode else 2
    aggregation_pipeline = []

    for i in range(n_iters):
        aggregator_name = sampler.choice(
            ["CountAggregator", "ActivationAggregator", "UnionAggregator"],
            handle=f"aggretator-{i}",
        )
        if aggregator_name == "CountAggregator":
            filter = get_filter(sampler, allow_none=True, prefix="count")
            aggregator = CountAggregator(attr_filter=filter)

        if aggregator_name == "ActivationAggregator":
            filter = get_filter(sampler, allow_none=True, prefix="activation")

            activation_name = sampler.choice(
                ["max", "sum", "mult"], handle="activation-function"
            )
            if activation_name == "max":
                activation_func = max
            elif activation_name == "sum":
                activation_func = sum
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
            else:
                correct += 1

        counter[value] = (correct, spurious, missing)
    return counter


def compute_scores(counter):
    scores = {}
    for value, (correct, spurious, missing) in counter.items():
        precision = correct / (correct + spurious)
        recall = correct / (correct + missing)
        f1 = 2 * precision * recall / (precision + recall)
        scores[value] = {
            PRECISION: precision,
            RECALL: recall,
            F1: f1,
        }
    scores[MACRO_F1] = mean(group[F1] for group in scores.values())
    return scores


if __name__ == "__main__":
    args = setup()

    best_solution, best_fn = optimize(
        pop_size=args.popsize,
        search_iterations=args.iterations,
        memory_limit=args.memory * 1024**3,
        search_timeout=args.global_timeout,
        errors="warn",
        telegram_token=args.token,
        telegram_channel=args.channel,
        telegram_title=args.title,
        log_path=args.output,
    )

    print(best_fn)
    print(best_fn)
