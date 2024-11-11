from pathlib import Path

import pandas as pd
from bfair.envs import VILLANOS_DATASET

from .base import Dataset

TEXT_COLUMN = "Text"
LABEL_COLUMN = "Violent"

TARGET_VALUES = ["yes", "no"]
POSITIVE_VALUE = "no"

_TEXT_COLUMN = "text"
_LABEL_COLUMN = "label"


def load_dataset(path=VILLANOS_DATASET, **kwargs):
    return VillanosDataset.load(
        path,
    )


class VillanosDataset(Dataset):
    @classmethod
    def load(cls, path):
        path = Path(path)

        collections = {}
        for split in ["all"]:
            df = pd.read_csv(path / f"{split}.tsv", sep="\t")
            violent_list = df[_LABEL_COLUMN].apply(
                (
                    lambda row: "yes"
                    if row == "VIOLENTO"
                    else "no"
                    if row == "NOVIOLENTO"
                    else None
                ),
            )
            data = pd.concat(
                [
                    df[_TEXT_COLUMN].rename(TEXT_COLUMN),
                    violent_list.rename(LABEL_COLUMN),
                ],
                join="inner",
                axis=1,
            )
            collections[split] = data

        return VillanosDataset(
            data=collections["all"],
        )
