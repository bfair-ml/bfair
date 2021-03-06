from pathlib import Path

import pandas as pd
from bfair.envs import ADULT_DATASET

from .base import Dataset


def load_dataset(path=ADULT_DATASET):
    return AdultDataset.load(path)


class AdultDataset(Dataset):
    @classmethod
    def load(cls, path):
        path = Path(path)
        data_path = path / "adult.data"
        test_path = path / "adult.test"

        names_path = path / "adult.names"
        lines = names_path.read_text().strip().splitlines()[-14:]
        names = [line.split(":")[0] for line in lines] + ["income"]

        data = pd.read_csv(
            data_path, sep=", ", header=None, names=names, engine="python"
        )
        test = pd.read_csv(
            test_path, sep=", ", header=None, names=names, engine="python", skiprows=1
        )
        test["income"] = test["income"].str.slice(0, -1)

        return AdultDataset(data=data, test=test)
