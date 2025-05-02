import argparse
import pandas as pd
import datasets as db

from tqdm import tqdm
from pathlib import Path
from functools import partial

from bfair.sensors import (
    P_GENDER,
    SensorHandler,
    EmbeddingBasedSensor,
    CoreferenceNERSensor,
    DBPediaSensor,
    NameGenderSensor,
    StaticWordsSensor,
    TwitterNERSensor,
)
from bfair.sensors.optimization import (
    load,
    compute_errors,
    compute_scores,
    compute_multiclass_scores,
    compute_multiclass_errors,
    attributes_to_class,
)
from bfair.sensors.mocks import FixValueSensor, RandomValueSensor
from bfair.datasets import (
    load_review,
    load_mdgender,
    load_image_chat,
    load_funpedia,
    load_toxicity,
    load_villanos,
)
from bfair.datasets.reviews import (
    REVIEW_COLUMN as TEXT_COLUMN_REVIEW,
    GENDER_COLUMN as GENDER_COLUMN_REVIEW,
    GENDER_VALUES as GENDER_VALUES_REVIEW,
    SENTIMENT_COLUMN as TARGET_COLUMN_REVIEW,
    POSITIVE_VALUE as POSITIVE_VALUE_REVIEW,
)
from bfair.datasets.mdgender import (
    TEXT_COLUMN as TEXT_COLUMN_MDGENDER,
    GENDER_COLUMN as GENDER_COLUMN_MDGENDER,
    GENDER_VALUES as GENDER_VALUES_MDGENDER,
)
from bfair.datasets.imagechat import (
    TEXT_COLUMN as TEXT_COLUMN_IMAGECHAT,
    GENDER_COLUMN as GENDER_COLUMN_IMAGECHAT,
    GENDER_VALUES as GENDER_VALUES_IMAGECHAT,
)
from bfair.datasets.funpedia import (
    TEXT_COLUMN as TEXT_COLUMN_FUNPEDIA,
    GENDER_COLUMN as GENDER_COLUMN_FUNPEDIA,
    GENDER_VALUES as GENDER_VALUES_FUNPEDIA,
    NEUTRAL_VALUE as NEUTRAL_VALUE_FUNPEDIA,
)
from bfair.datasets.toxicity import (
    TEXT_COLUMN as TEXT_COLUMN_TOXICITY,
    GENDER_COLUMN as GENDER_COLUMN_TOXICITY,
    GENDER_VALUES as GENDER_VALUES_TOXICITY,
)
from bfair.datasets.villanos import (
    TEXT_COLUMN as TEXT_COLUMN_VILLANOS,
    LABEL_COLUMN as TARGET_COLUMN_VILLANOS,
    POSITIVE_VALUE as POSITIVE_VALUE_VILLANOS,
    GENDER_COLUMN_A as GENDER_COLUMN_A_VILLANOS,
    GENDER_COLUMN_B as GENDER_COLUMN_B_VILLANOS,
    GENDER_VALUES as GENDER_VALUES_VILLANOS,
)
from bfair.metrics import (
    exploded_statistical_parity,
    exploded_representation_disparity,
    MODES,
)

DB_REVIEWS = "reviews"
DB_MDGENDER = "mdgender"
DB_IMAGECHAT = "imagechat"
DB_FUNPEDIA = "funpedia"
DB_TOXICITY = "toxicity"
DB_VILLANOS = "villanos"
DB_VILLANOS_GENDERED = "villanos-gendered"


def get_args():
    # - SETUP ---
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", action="append")
    group.add_argument("--annotations-from", default=None)

    parser.add_argument("--eval-all", action="store_true")
    parser.add_argument("--dump-path", default=None)
    parser.add_argument(
        "--datasets",
        action="append",
        choices=[
            DB_REVIEWS,
            DB_MDGENDER,
            DB_IMAGECHAT,
            DB_FUNPEDIA,
            DB_TOXICITY,
            DB_VILLANOS,
            DB_VILLANOS_GENDERED,
        ],
    )
    parser.add_argument("--eval-annotation", action="store_true")
    parser.add_argument("--eval-fairness", action="store_true")
    args = parser.parse_args()
    return args


def get_handlers(config):
    handlers = []

    for config_str in config:
        # - Load CONFIG ---

        if config_str == "always-male":
            handler = SensorHandler(sensors=[FixValueSensor("male")])
        elif config_str == "always-female":
            handler = SensorHandler(sensors=[FixValueSensor("female")])
        elif config_str == "always-both":
            handler = SensorHandler(sensors=[FixValueSensor(["male", "female"])])
        elif config_str == "always-none":
            handler = SensorHandler(sensors=[FixValueSensor([])])
        elif config_str.startswith("random-"):
            config = config_str.split("#")
            seed = 0 if len(config) < 2 else config[-1]
            if config_str.startswith("random-uniform"):
                handler = SensorHandler(sensors=[RandomValueSensor(seed=seed)])
            elif config_str.startswith("random-for-review-training"):
                handler = SensorHandler(
                    sensors=[
                        RandomValueSensor(
                            seed=seed, distribution={"female": 24 / 49, "male": 33 / 49}
                        )
                    ]
                )
            elif config_str.startswith("random-for-imagechat-testing"):
                handler = SensorHandler(
                    sensors=[
                        RandomValueSensor(
                            seed=seed,
                            distribution={"female": 468 / 5000, "male": 998 / 5000},
                        )
                    ]
                )
            else:
                print(f"Invalid handler configuration. {config_str}")
                exit()
        elif config_str.startswith("defaults"):
            _, *params = config_str.split("|")
            kwargs = {
                name: value for param in params for name, value in (param.split(":"),)
            }

            def select(*arguments):
                return {
                    name: value for name, value in kwargs.items() if name in arguments
                }

            names = NameGenderSensor.build(**select("language"))
            handler = SensorHandler(
                sensors=[
                    EmbeddingBasedSensor.build_default_in_plain_mode(
                        **select("language", "source")
                    ),
                    CoreferenceNERSensor.build(**select("language")),
                    # DBPediaSensor.build(**select("language")),
                    names,
                    StaticWordsSensor.build(**select("language")),
                    # TwitterNERSensor(names, cache_path="twitter-cache.json", access_token=""),
                ],
            )
        else:
            try:
                config = eval(config_str)
            except Exception as e:
                print(f"Invalid handler configuration. {e}")
                exit()

            config = list(config.items())
            print("\nConfiguration:")
            for key, value in config:
                print(f"- {key}: {value}")

            generated = load(config)
            handler = generated.model
            print("Loaded!")

        handlers.append((config_str, handler))

    return handlers


def get_datasets(selected_datasets=None):
    # - Load DATASETS ---
    all_datasets = [
        (
            f"{DB_REVIEWS} (complete)",
            lambda: load_review().data,
            TEXT_COLUMN_REVIEW,
            GENDER_COLUMN_REVIEW,
            GENDER_VALUES_REVIEW,
            TARGET_COLUMN_REVIEW,
            POSITIVE_VALUE_REVIEW,
            None,
        ),
        (
            f"{DB_REVIEWS} (training)",
            lambda: load_review(split_seed=0).data,
            TEXT_COLUMN_REVIEW,
            GENDER_COLUMN_REVIEW,
            GENDER_VALUES_REVIEW,
            TARGET_COLUMN_REVIEW,
            POSITIVE_VALUE_REVIEW,
            None,
        ),
        (
            f"{DB_REVIEWS} (testing)",
            lambda: load_review(split_seed=0).test,
            TEXT_COLUMN_REVIEW,
            GENDER_COLUMN_REVIEW,
            GENDER_VALUES_REVIEW,
            TARGET_COLUMN_REVIEW,
            POSITIVE_VALUE_REVIEW,
            None,
        ),
        (
            DB_MDGENDER,
            lambda: load_mdgender().data,
            TEXT_COLUMN_MDGENDER,
            GENDER_COLUMN_MDGENDER,
            GENDER_VALUES_MDGENDER,
            None,
            None,
            None,
        ),
        (
            f"{DB_IMAGECHAT} (training)",
            lambda: load_image_chat().data,
            TEXT_COLUMN_IMAGECHAT,
            GENDER_COLUMN_IMAGECHAT,
            GENDER_VALUES_IMAGECHAT,
            None,
            None,
            None,
        ),
        (
            f"{DB_IMAGECHAT} (testing)",
            lambda: load_image_chat().test,
            TEXT_COLUMN_IMAGECHAT,
            GENDER_COLUMN_IMAGECHAT,
            GENDER_VALUES_IMAGECHAT,
            None,
            None,
            None,
        ),
        (
            f"{DB_TOXICITY} (training @ 0.0)",
            lambda: load_toxicity(threshold=0).data,
            TEXT_COLUMN_TOXICITY,
            GENDER_COLUMN_TOXICITY,
            GENDER_VALUES_TOXICITY,
            None,
            None,
            None,
        ),
        (
            f"{DB_TOXICITY} (training @ 0.5)",
            lambda: load_toxicity(threshold=0.5).data,
            TEXT_COLUMN_TOXICITY,
            GENDER_COLUMN_TOXICITY,
            GENDER_VALUES_TOXICITY,
            None,
            None,
            None,
        ),
        (
            f"{DB_FUNPEDIA} (testing)",
            lambda: load_funpedia().test,
            TEXT_COLUMN_FUNPEDIA,
            GENDER_COLUMN_FUNPEDIA,
            GENDER_VALUES_FUNPEDIA,
            None,
            None,
            NEUTRAL_VALUE_FUNPEDIA,
        ),
        (
            f"{DB_FUNPEDIA} (all data)",
            lambda: load_funpedia().all_data,
            TEXT_COLUMN_FUNPEDIA,
            GENDER_COLUMN_FUNPEDIA,
            GENDER_VALUES_FUNPEDIA,
            None,
            None,
            NEUTRAL_VALUE_FUNPEDIA,
        ),
        (
            f"{DB_VILLANOS} (all data)",
            lambda: load_villanos().all_data,
            TEXT_COLUMN_VILLANOS,
            None,
            GENDER_VALUES_VILLANOS,
            TARGET_COLUMN_VILLANOS,
            POSITIVE_VALUE_VILLANOS,
            None,
        ),
        (
            f"{DB_VILLANOS_GENDERED} ({GENDER_COLUMN_A_VILLANOS})",
            lambda: load_villanos(gendered=True).all_data,
            TEXT_COLUMN_VILLANOS,
            GENDER_COLUMN_A_VILLANOS,
            GENDER_VALUES_VILLANOS,
            TARGET_COLUMN_VILLANOS,
            POSITIVE_VALUE_VILLANOS,
            None,
        ),
        (
            f"{DB_VILLANOS_GENDERED} ({GENDER_COLUMN_B_VILLANOS})",
            lambda: load_villanos(gendered=True).all_data,
            TEXT_COLUMN_VILLANOS,
            GENDER_COLUMN_B_VILLANOS,
            GENDER_VALUES_VILLANOS,
            TARGET_COLUMN_VILLANOS,
            POSITIVE_VALUE_VILLANOS,
            None,
        ),
    ]

    datasets = [
        ds
        for ds in all_datasets
        if not selected_datasets or ds[0].split(" ")[0] in selected_datasets
    ]

    return datasets


def do_and_dump_annotations(
    handler,
    gender_values,
    data,
    text_column,
    dataset_name,
    dump_path=None,
    eval_all=False,
):
    texts_to_annotate = data[text_column]

    predictions = [
        handler(text, gender_values, P_GENDER)
        for text in tqdm(texts_to_annotate, desc=f"Texts {dataset_name}")
    ]

    if dump_path is not None:
        all_predictions = [("Handler", predictions)]

        if eval_all:
            for sensor in handler.sensors:
                pred = [
                    sensor(text, gender_values, P_GENDER) for text in texts_to_annotate
                ]
                all_predictions.append((type(sensor).__name__, pred))

        results = pd.concat(
            (
                data,
                *[
                    pd.Series(pred, name=name, index=data.index)
                    .apply(sorted)
                    .str.join(" & ")
                    for name, pred in all_predictions
                ],
            ),
            axis=1,
        )
        results.to_csv((dump_path / dataset_name).with_suffix(".csv"))

    return predictions


def evaluate_annotation(gold, predictions, gender_values, neutral_value=None):
    def multilabel_errors(true_labels, predictions, gender_values):
        errors = compute_errors(true_labels, predictions, gender_values)
        print(errors)
        scores = compute_scores(errors)
        print(scores)

    def multiclass_error(true_classes, predictions, neutral_value):
        errors = compute_multiclass_errors(
            true_classes,
            predictions,
            partial(attributes_to_class, neutral=neutral_value),
        )
        print(errors)
        scores = compute_multiclass_scores(errors)
        print(scores)

    if neutral_value is None:
        multilabel_errors(gold, predictions, gender_values)
    else:
        multiclass_error(gold, predictions, neutral_value)


def compare_fairness(
    data,
    gender_column,
    predictions,
    target_column,
    positive_target,
    values=None,
):
    evaluate_fairness(
        data,
        gender_column,
        target_column,
        positive_target,
        "True",
        values=values,
    )
    auto_annotated = data.copy()
    auto_annotated[gender_column] = [list(x) for x in predictions]
    evaluate_fairness(
        auto_annotated,
        gender_column,
        target_column,
        positive_target,
        "Estimated",
        values=values,
    )


def evaluate_fairness(
    data,
    gender_column,
    target_column,
    positive_target,
    tag,
    values=None,
):
    values = [positive_target] if values is None else values

    for mode in MODES:

        fairness = exploded_statistical_parity(
            data=data,
            protected_attributes=gender_column,
            target_attribute=target_column,
            target_predictions=None,
            positive_target=positive_target,
            return_probs=True,
            ndigits=3,
            mode=mode,
        )
        print(f"## {tag} fairness [{mode}: statistical parity]:", fairness)

        for value in values:
            fairness = exploded_representation_disparity(
                data=data,
                protected_attributes=gender_column,
                target_attribute=target_column,
                target_predictions=None,
                positive_target=value,
                return_probs=True,
                ndigits=3,
                mode=mode,
            )
            print(
                f"## {tag} fairness [{mode}: representation disparity {value}]:",
                fairness,
            )


def evaluate_fairness_equally(
    data,
    gender_column,
    target_column,
    positive_target,
    tag,
    fairness_metrics,
):
    for name, metric in fairness_metrics.items():
        fairness = metric(
            data=data,
            protected_attributes=gender_column,
            target_attribute=target_column,
            target_predictions=None,
            positive_target=positive_target,
            return_probs=True,
            ndigits=3,
        )
        print(f"## {tag} fairness [{name}]:", fairness)


def main():
    args = get_args()
    config = args.config
    eval_all = args.eval_all if config else False
    annotations_from = args.annotations_from
    dump_path = Path(args.dump_path) if args.dump_path is not None else None
    eval_annotation = args.eval_annotation
    eval_fairness = args.eval_fairness

    handlers = (
        get_handlers(args.config) if config else [(f"column:{annotations_from}", None)]
    )
    datasets = get_datasets(args.datasets)

    db.logging.set_verbosity_error()

    for config_str, handler in handlers:

        print(repr(config_str))

        for (
            dataset_name,
            load_dataset,
            text_column,
            gender_column,
            gender_values,
            target_column,
            positive_target,
            neutral_value,
        ) in tqdm(datasets, desc="Datasets"):

            print(f"\n# {dataset_name}")
            data = load_dataset()

            if handler is not None:
                predictions = do_and_dump_annotations(
                    handler,
                    gender_values,
                    data,
                    text_column,
                    dataset_name,
                    dump_path,
                    eval_all,
                )
            elif annotations_from is not None:
                predictions = data[annotations_from]
            else:
                raise ValueError()

            if eval_annotation and gender_column is not None:
                gold = data[gender_column]
                evaluate_annotation(gold, predictions, gender_values, neutral_value)

            if eval_fairness and target_column is not None:
                compare_fairness(
                    data,
                    gender_column,
                    predictions,
                    target_column,
                    positive_target,
                    values=data[target_column].unique(),
                )


if __name__ == "__main__":
    main()
