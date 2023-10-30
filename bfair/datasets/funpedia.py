from .base import Dataset

import pandas as pd
import datasets as db

MALE_VALUE = "Male"
FEMALE_VALUE = "Female"
NEUTRAL_VALUE = "Gender-Neutral"
GENDER_VALUES = [MALE_VALUE, FEMALE_VALUE]

TEXT_COLUMN = "Text"
GENDER_COLUMN = "Gender"

_TEXT_COLUMN = "text"
_GENDER_COLUMN = "gender"


def load_dataset(**kwargs):
    return FunpediaDataset.load()


class FunpediaDataset(Dataset):
    @classmethod
    def load(cls):
        source = db.load_dataset("md_gender_bias", "funpedia")

        collections = {}
        for split in source:
            df = source[split].to_pandas()
            data = pd.concat(
                [
                    df[_TEXT_COLUMN].rename(TEXT_COLUMN),
                    df[_GENDER_COLUMN].apply(str.title).rename(GENDER_COLUMN),
                ],
                axis=1,
            )
            collections[split] = data

        return FunpediaDataset(
            data=collections["train"],
            validation=collections["validation"],
            test=collections["test"],
        )
