import argparse
import pandas as pd
import datasets as db

from pathlib import Path

from bfair.sensors import P_GENDER, SensorHandler
from bfair.sensors.optimization import load, compute_errors, compute_scores
from bfair.sensors.mocks import FixValueSensor, RandomValueSensor
from bfair.datasets import load_review, load_mdgender, load_image_chat
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
from bfair.metrics import exploded_statistical_parity

DB_REVIEWS = "reviews"
DB_MDGENDER = "mdgender"
DB_IMAGECHAT = "imagechat"


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
            handler = SensorHandler(sensors=[FixValueSensor("Male")])
        elif config_str == "always-female":
            handler = SensorHandler(sensors=[FixValueSensor("Female")])
        elif config_str.startswith("random-"):
            config = config_str.split("#")
            seed = 0 if len(config) < 2 else config[-1]
            if config_str.startswith("random-uniform"):
                handler = SensorHandler(sensors=[RandomValueSensor(seed=seed)])
            elif config_str.startswith("random-for-review-training"):
                handler = SensorHandler(
                    sensors=[
                        RandomValueSensor(
                            seed=seed, distribution={"Female": 24 / 49, "Male": 33 / 49}
                        )
                    ]
                )
            elif config_str.startswith("random-for-imagechat-testing"):
                handler = SensorHandler(
                    sensors=[
                        RandomValueSensor(
                            seed=seed,
                            distribution={"Female": 468 / 5000, "Male": 998 / 5000},
                        )
                    ]
                )
            else:
                print(f"Invalid handler configuration. {config_str}")
                exit()
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

    # - Load DATASETS ---
    datasets = [
        (
            f"{DB_REVIEWS} (complete)",
            lambda: load_review().data,
            TEXT_COLUMN_REVIEW,
            GENDER_COLUMN_REVIEW,
            GENDER_VALUES_REVIEW,
            TARGET_COLUMN_REVIEW,
        ),
        (
            f"{DB_REVIEWS} (training)",
            lambda: load_review(split_seed=0).data,
            TEXT_COLUMN_REVIEW,
            GENDER_COLUMN_REVIEW,
            GENDER_VALUES_REVIEW,
            TARGET_COLUMN_REVIEW,
        ),
        (
            f"{DB_REVIEWS} (testing)",
            lambda: load_review(split_seed=0).test,
            TEXT_COLUMN_REVIEW,
            GENDER_COLUMN_REVIEW,
            GENDER_VALUES_REVIEW,
            TARGET_COLUMN_REVIEW,
        ),
        (
            DB_MDGENDER,
            lambda: load_mdgender().data,
            TEXT_COLUMN_MDGENDER,
            GENDER_COLUMN_MDGENDER,
            GENDER_VALUES_MDGENDER,
            None,
        ),
        (
            f"{DB_IMAGECHAT} (training)",
            lambda: load_image_chat().data,
            TEXT_COLUMN_IMAGECHAT,
            GENDER_COLUMN_IMAGECHAT,
            GENDER_VALUES_IMAGECHAT,
            None,
        ),
        (
            f"{DB_IMAGECHAT} (testing)",
            lambda: load_image_chat().test,
            TEXT_COLUMN_IMAGECHAT,
            GENDER_COLUMN_IMAGECHAT,
            GENDER_VALUES_IMAGECHAT,
            None,
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
        ) in datasets:

            print(f"\n# {dataset_name}")
            data = load_dataset()

            X = data[text_column]
            y = data[gender_column]

            predictions = [handler(text, gender_values, P_GENDER) for text in X]

            errors = compute_errors(y, predictions, gender_values)
            print(errors)

            scores = compute_scores(errors)
            print(scores)

            if dump_path is not None:
                all_predictions = [("Hander", predictions)]

                if eval_all:
                    for sensor in handler.sensors:
                        pred = [sensor(text, gender_values, P_GENDER) for text in X]
                        all_predictions.append((type(sensor).__name__, pred))

                results = pd.concat(
                    (
                        X,
                        y.str.join(" & "),
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
