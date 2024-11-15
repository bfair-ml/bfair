import argparse
import pandas as pd
from experiments.sensors.evaluate import evaluate_fairness
from bfair.metrics import exploded_statistical_parity, exploded_representation_disparity


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gender-column", required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--positive-target", default=None)
    parser.add_argument("--paths", action="append", required=True)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    gender_column = args.gender_column
    target_column = args.target_column
    positive_target = args.positive_target
    dataset_paths = args.paths

    for path in dataset_paths:
        data = pd.read_csv(path)
        data[gender_column] = data[gender_column].str.split(" & ")

        targets = (
            [positive_target]
            if positive_target is not None
            else data[target_column].unique()
        )

        for target in targets:
            evaluate_fairness(
                data,
                gender_column,
                target_column,
                target,
                f'Column "{gender_column}" [{target}]',
                {
                    "SP": exploded_statistical_parity,
                    "RD": exploded_representation_disparity,
                },
            )


if __name__ == "__main__":
    main()
