import argparse
import pandas as pd
import datasets as db

from tqdm import tqdm
from pathlib import Path

from bfair.sensors import (
    P_GENDER,
    SensorHandler,
    EmbeddingBasedSensor,
    CoreferenceNERSensor,
    DBPediaSensor,
    NameGenderSensor,
    StaticWordsSensor,
)
from bfair.sensors.optimization import (
    load,
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
    GENDER_VALUES as GENDER_VALUES_REVIEW,
)
from bfair.datasets.mdgender import (
    TEXT_COLUMN as TEXT_COLUMN_MDGENDER,
    GENDER_VALUES as GENDER_VALUES_MDGENDER,
)
from bfair.datasets.imagechat import (
    TEXT_COLUMN as TEXT_COLUMN_IMAGECHAT,
    GENDER_VALUES as GENDER_VALUES_IMAGECHAT,
)
from bfair.datasets.funpedia import (
    TEXT_COLUMN as TEXT_COLUMN_FUNPEDIA,
    GENDER_VALUES as GENDER_VALUES_FUNPEDIA,
)
from bfair.datasets.toxicity import (
    TEXT_COLUMN as TEXT_COLUMN_TOXICITY,
    GENDER_VALUES as GENDER_VALUES_TOXICITY,
)
from bfair.datasets.villanos import (
    TEXT_COLUMN as TEXT_COLUMN_VILLANOS,
)

DB_REVIEWS = "reviews"
DB_MDGENDER = "mdgender"
DB_IMAGECHAT = "imagechat"
DB_FUNPEDIA = "funpedia"
DB_TOXICITY = "toxicity"
DB_VILLANOS = "villanos"


def main():
    # - SETUP ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
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
        ],
    )
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
        elif config_str.startswith("defaults"):
            _, *params = config_str.split("|")
            kwargs = {
                name: value for param in params for name, value in (param.split(":"),)
            }

            def select(*arguments):
                return {
                    name: value for name, value in kwargs.items() if name in arguments
                }

            handler = SensorHandler(
                sensors=[
                    EmbeddingBasedSensor.build_default_in_plain_mode(
                        **select("language", "source")
                    ),
                    CoreferenceNERSensor.build(**select("language")),
                    # DBPediaSensor.build(**select("language")),
                    NameGenderSensor.build(**select("language")),
                    StaticWordsSensor.build(**select("language")),
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

    # - Load DATASETS ---
    all_datasets = [
        (
            f"{DB_REVIEWS} (complete)",
            lambda: load_review().data,
            TEXT_COLUMN_REVIEW,
            GENDER_VALUES_REVIEW,
        ),
        (
            f"{DB_REVIEWS} (training)",
            lambda: load_review(split_seed=0).data,
            TEXT_COLUMN_REVIEW,
            GENDER_VALUES_REVIEW,
        ),
        (
            f"{DB_REVIEWS} (testing)",
            lambda: load_review(split_seed=0).test,
            TEXT_COLUMN_REVIEW,
            GENDER_VALUES_REVIEW,
        ),
        (
            DB_MDGENDER,
            lambda: load_mdgender().data,
            TEXT_COLUMN_MDGENDER,
            GENDER_VALUES_MDGENDER,
        ),
        (
            f"{DB_IMAGECHAT} (training)",
            lambda: load_image_chat().data,
            TEXT_COLUMN_IMAGECHAT,
            GENDER_VALUES_IMAGECHAT,
        ),
        (
            f"{DB_IMAGECHAT} (testing)",
            lambda: load_image_chat().test,
            TEXT_COLUMN_IMAGECHAT,
            GENDER_VALUES_IMAGECHAT,
        ),
        (
            f"{DB_TOXICITY} (training @ 0.0)",
            lambda: load_toxicity(threshold=0).data,
            TEXT_COLUMN_TOXICITY,
            GENDER_VALUES_TOXICITY,
        ),
        (
            f"{DB_TOXICITY} (training @ 0.5)",
            lambda: load_toxicity(threshold=0.5).data,
            TEXT_COLUMN_TOXICITY,
            GENDER_VALUES_TOXICITY,
        ),
        (
            f"{DB_FUNPEDIA} (testing)",
            lambda: load_funpedia().test,
            TEXT_COLUMN_FUNPEDIA,
            GENDER_VALUES_FUNPEDIA,
        ),
        (
            f"{DB_FUNPEDIA} (all data)",
            lambda: load_funpedia().all_data,
            TEXT_COLUMN_FUNPEDIA,
            GENDER_VALUES_FUNPEDIA,
        ),
        (
            f"{DB_VILLANOS} (all data)",
            lambda: load_villanos().all_data,
            TEXT_COLUMN_VILLANOS,
            ["male", "female"],
        ),
    ]

    datasets = [
        ds
        for ds in all_datasets
        if not args.datasets or ds[0].split(" ")[0] in args.datasets
    ]

    db.logging.set_verbosity_error()

    for config_str, handler in handlers:

        print(repr(config_str))

        for (
            dataset_name,
            load_dataset,
            text_column,
            gender_values,
        ) in tqdm(datasets, desc="Datasets"):

            print(f"\n# {dataset_name}")
            data = load_dataset()

            X = data[text_column]

            predictions = [
                handler(text, gender_values, P_GENDER)
                for text in tqdm(X, desc=f"Texts {dataset_name}")
            ]

            if dump_path is not None:
                all_predictions = [("Hander", predictions)]

                if eval_all:
                    for sensor in handler.sensors:
                        pred = [sensor(text, gender_values, P_GENDER) for text in X]
                        all_predictions.append((type(sensor).__name__, pred))

                results = pd.concat(
                    (
                        X,
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


if __name__ == "__main__":
    main()
