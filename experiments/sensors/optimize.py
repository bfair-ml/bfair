import argparse
import traceback
from pathlib import Path

from bfair.datasets import load_review, load_mdgender
from bfair.datasets.reviews import (
    REVIEW_COLUMN,
    GENDER_COLUMN as GENDER_COLUMN_REVIEW,
    GENDER_VALUES,
)
from bfair.datasets.mdgender import (
    TEXT_COLUMN,
    GENDER_COLUMN as GENDER_COLUMN_MDGENDER,
)
from bfair.sensors import SensorHandler, EmbeddingBasedSensor, P_GENDER
from bfair.sensors.optimization import (
    optimize,
    compute_errors,
    compute_scores,
    MACRO_F1,
    MACRO_ACC,
    MICRO_ACC,
)
from autogoal.kb import Text


DB_REVIEWS = "reviews"
DB_MDGENDER = "mdgender"

SKIP_EMBEDDING = "embedding"
SKIP_COREFERENCE = "coreference"
SKIP_DBPEDIA = "dbpedia"


def run_all():
    dataset = load_review(split_seed=None)
    sensor = EmbeddingBasedSensor.build_default_in_hierarchy_mode(
        language="english", source="word2vec-debiased"
    )
    handler = SensorHandler(sensors=[sensor])
    reviews = dataset.data[REVIEW_COLUMN]
    predicted = []
    for text in reviews:
        annotations = handler.annotate(text, Text, GENDER_VALUES, P_GENDER)
        predicted.append(annotations)
    gold = dataset.data[GENDER_COLUMN]
    errors = compute_errors(gold, predicted, GENDER_VALUES)
    scores = compute_scores(errors)
    print(scores)


def setup():
    parser = argparse.ArgumentParser()

    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--eval-timeout", type=int, default=None)
    parser.add_argument("--memory", type=int, default=None)
    parser.add_argument("--popsize", type=int, default=50)
    parser.add_argument("--global-timeout", type=int, default=60 * 60)
    parser.add_argument("--token", default=None)
    parser.add_argument("--channel", default=None)
    parser.add_argument("--output", default="")
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--metric",
        type=str,
        choices=[MACRO_F1, MACRO_ACC, MICRO_ACC],
        default=MACRO_F1,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[DB_REVIEWS, DB_MDGENDER],
        default=DB_REVIEWS,
    )
    parser.add_argument(
        "--skip",
        action="append",
        choices=[SKIP_EMBEDDING, SKIP_COREFERENCE, SKIP_DBPEDIA],
        default=[],
    )

    return parser.parse_args()


def main():
    args = setup()

    if args.output:
        Path(args.output).parent.mkdir(exist_ok=True)
        output_stream = open(args.output, mode="a")
    else:
        output_stream = None

    try:
        if args.dataset == DB_REVIEWS:
            load_dataset_func = load_review
            text_column, sensitive_column = REVIEW_COLUMN, GENDER_COLUMN_REVIEW
        elif args.dataset == DB_MDGENDER:
            load_dataset_func = load_mdgender
            text_column, sensitive_column = TEXT_COLUMN, GENDER_COLUMN_MDGENDER
        else:
            raise ValueError(f'Invalid dataset: "{args.dataset}"')

        dataset = load_dataset_func(split_seed=0)
        X_train = dataset.data[text_column]
        y_train = dataset.data[sensitive_column]
        X_test = dataset.test[text_column]
        y_test = dataset.test[sensitive_column]

        best_solution, best_fn = optimize(
            X_train,
            y_train,
            X_test,
            y_test,
            GENDER_VALUES,
            P_GENDER,
            score_key=args.metric,
            consider_embedding_sensors=SKIP_EMBEDDING not in args.skip,
            consider_coreference_sensor=SKIP_COREFERENCE not in args.skip,
            consider_dbpedia_sensor=SKIP_DBPEDIA not in args.skip,
            pop_size=args.popsize,
            search_iterations=args.iterations,
            evaluation_timeout=args.eval_timeout,
            memory_limit=args.memory * 1024**3 if args.memory else None,
            search_timeout=args.global_timeout,
            errors="warn",
            telegram_token=args.token,
            telegram_channel=args.channel,
            telegram_title=args.title,
            log_path=args.output,
            inspect=True,
            output_stream=output_stream,
        )

        print(best_fn, file=output_stream)
        print(best_solution, file=output_stream, flush=True)

    except Exception as e:
        print(
            "\n",
            "ERROR",
            "\n",
            str(e),
            "\n",
            traceback.format_exc(),
            "\n",
            file=output_stream,
            flush=True,
        )
    finally:
        if output_stream is not None:
            output_stream.close()


if __name__ == "__main__":
    main()
