from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from src.envs import GERMAN_DATASET


def load_dataset(path=GERMAN_DATASET, split_seed=None):
    return GermanDataset.load(path, split_seed=split_seed)


class GermanDataset:
    def __init__(self, data: pd.DataFrame, test: pd.DataFrame = None):
        self.data = data
        self.test = test

    def load(path, categorical=True, split_seed=None):
        path = Path(path)
        data_path = path / ("german.data" if categorical else "german.data-numeric")

        names = [f"Attribute {n + 1}" for n in range(20)] + ["risk"]

        data = pd.read_csv(
            data_path,
            sep="\s+",
            header=None,
            names=names,
            engine="python",
            index_col=False,
            dtype={"risk": object},
        )

        if split_seed:
            train, test = train_test_split(
                data, test_size=0.3, random_state=split_seed, shuffle=True
            )
        else:
            train, test = data, pd.DataFrame(columns=data.columns).astype(data.dtypes)

        return GermanDataset(data=train, test=test)
