from pathlib import Path

import pandas as pd
from bfair.envs import TOXICITY_DATASET

from .base import Dataset

MALE_VALUE = "Male"
FEMALE_VALUE = "Female"
GENDER_VALUES = [MALE_VALUE, FEMALE_VALUE]

TEXT_COLUMN = "Text"
GENDER_COLUMN = "Gender"

_TEXT_COLUMN = "comment_text"
_MALE_COLUMN = "male"
_FEMALE_COLUMN = "female"
_GENDER_COLUMNS = [_MALE_COLUMN, _FEMALE_COLUMN]


def load_dataset(path=TOXICITY_DATASET, threshold=0):
    return ToxicityDataset.load(path, threshold=threshold)


class ToxicityDataset(Dataset):
    @classmethod
    def load(cls, path, threshold=0):
        path = Path(path)

        collections = {}
        for split in ["train"]:
            df = pd.read_csv(path / f"{split}.csv", engine="python")
            gender_list = (
                df[_GENDER_COLUMNS]
                .dropna()
                .apply(
                    lambda row: [
                        gender.title()
                        for gender, score in zip(_GENDER_COLUMNS, row)
                        if score > threshold
                    ],
                    axis=1,
                )
            )
            data = pd.concat(
                [
                    df[_TEXT_COLUMN].rename(TEXT_COLUMN),
                    gender_list.rename(GENDER_COLUMN),
                ],
                axis=1,
            )
            collections[split] = data

        return ToxicityDataset(
            data=collections["train"],
        )
