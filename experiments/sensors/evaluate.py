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
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval-all", action="store_true")
    parser.add_argument("--dump-path", default=None)
    args = parser.parse_args()
    config_str = args.config
    eval_all = args.eval_all
    dump_path = Path(args.dump_path) if args.dump_path is not None else None

    # - Load CONFIG ---
    print(config_str)

    if config_str == "always-male":
        handler = SensorHandler(sensors=[FixValueSensor("Male")])
    elif config_str == "always-female":
        handler = SensorHandler(sensors=[FixValueSensor("Female")])
    elif config_str == "random-uniform":
        handler = SensorHandler(sensors=[RandomValueSensor(seed=0)])
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
