import argparse
import pandas as pd
import datasets as db

from pathlib import Path
from functools import partial

from bfair.sensors import (
    P_GENDER,
    SensorHandler,
    EmbeddingBasedSensor,
    CoreferenceNERSensor,
    DBPediaSensor,
    NameGenderSensor,
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
)
from bfair.datasets.reviews import (
    REVIEW_COLUMN as TEXT_COLUMN_REVIEW,
    GENDER_COLUMN as GENDER_COLUMN_REVIEW,
    GENDER_VALUES as GENDER_VALUES_REVIEW,
    SENTIMENT_COLUMN as TARGET_COLUMN_REVIEW,
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
from bfair.metrics import exploded_statistical_parity

DB_REVIEWS = "reviews"
DB_MDGENDER = "mdgender"
DB_IMAGECHAT = "imagechat"
DB_FUNPEDIA = "funpedia"
DB_TOXICITY = "toxicity"


def main():
    # - SETUP ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--eval-all", action="store_true")
    parser.add_argument("--dump-path", default=None)
    args = parser.parse_args()
    eval_all = args.eval_all
    dump_path = Path(args.dump_path) if args.dump_path is not None else None

    handlers = []
    for config_str in args.config:
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
        elif config_str == "defaults":
            handler = SensorHandler(
                sensors=[
                    EmbeddingBasedSensor.build_default_in_plain_mode(),
                    CoreferenceNERSensor.build(),
                    # DBPediaSensor.build(),
                    NameGenderSensor.build(),
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

    # - Load DATASETS ---
    datasets = [
        (
            f"{DB_REVIEWS} (complete)",
            lambda: load_review().data,
            TEXT_COLUMN_REVIEW,
            GENDER_COLUMN_REVIEW,
            GENDER_VALUES_REVIEW,
            TARGET_COLUMN_REVIEW,
            None,
        ),
        (
            f"{DB_REVIEWS} (training)",
            lambda: load_review(split_seed=0).data,
            TEXT_COLUMN_REVIEW,
            GENDER_COLUMN_REVIEW,
            GENDER_VALUES_REVIEW,
            TARGET_COLUMN_REVIEW,
            None,
        ),
        (
            f"{DB_REVIEWS} (testing)",
            lambda: load_review(split_seed=0).test,
            TEXT_COLUMN_REVIEW,
            GENDER_COLUMN_REVIEW,
            GENDER_VALUES_REVIEW,
            TARGET_COLUMN_REVIEW,
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
        ),
        (
            f"{DB_IMAGECHAT} (training)",
            lambda: load_image_chat().data,
            TEXT_COLUMN_IMAGECHAT,
            GENDER_COLUMN_IMAGECHAT,
            GENDER_VALUES_IMAGECHAT,
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
        ),
        (
            f"{DB_TOXICITY} (training @ 0.0)",
            lambda: load_toxicity(threshold=0).data,
            TEXT_COLUMN_TOXICITY,
            GENDER_COLUMN_TOXICITY,
            GENDER_VALUES_TOXICITY,
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
        ),
        (
            f"{DB_FUNPEDIA} (testing)",
            lambda: load_funpedia().test,
            TEXT_COLUMN_FUNPEDIA,
            GENDER_COLUMN_FUNPEDIA,
            GENDER_VALUES_FUNPEDIA,
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
            NEUTRAL_VALUE_FUNPEDIA,
        ),
    ]

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
            neutral_value,
        ) in datasets:

            print(f"\n# {dataset_name}")
            data = load_dataset()

            X = data[text_column]
            y = data[gender_column]

            predictions = [handler(text, gender_values, P_GENDER) for text in X]

            if neutral_value is None:
                multilabel_errors(y, predictions, gender_values)
            else:
                multiclass_error(y, predictions, neutral_value)

            if dump_path is not None:
                all_predictions = [("Hander", predictions)]

                if eval_all:
                    for sensor in handler.sensors:
                        pred = [sensor(text, gender_values, P_GENDER) for text in X]
                        all_predictions.append((type(sensor).__name__, pred))

                results = pd.concat(
                    (
                        X,
                        y.str.join(" & ") if neutral_value is None else y,
                        *[
                            pd.Series(pred, name=name, index=X.index)
                            .apply(sorted)
                            .str.join(" & ")
                            for name, pred in all_predictions
                        ],
                    ),
                    axis=1,
                )
                results.to_csv((dump_path / dataset_name).with_suffix(".csv"))

            if target_column is None:
                continue

            fairness = exploded_statistical_parity(
                data=data,
                protected_attributes=gender_column,
                target_attribute=target_column,
                target_predictions=None,
                positive_target="positive",
                return_probs=True,
            )
            print("## True fairness:", fairness)

            auto_annotated = data.copy()
            auto_annotated[gender_column] = [list(x) for x in predictions]

            fairness = exploded_statistical_parity(
                data=auto_annotated,
                protected_attributes=gender_column,
                target_attribute=target_column,
                target_predictions=None,
                positive_target="positive",
                return_probs=True,
            )
            print("## Estimated fairness:", fairness)


if __name__ == "__main__":
    main()
