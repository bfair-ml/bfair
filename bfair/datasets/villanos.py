from pathlib import Path

import pandas as pd
from bfair.envs import VILLANOS_DATASET

from .base import Dataset

TEXT_COLUMN = "Text"
LABEL_COLUMN = "Violent"
GENDER_COLUMN_A = MENTIONS = "Mentions"
GENDER_COLUMN_B = TARGETS = "Targets"

MALE_VALUE = "male"
FEMALE_VALUE = "female"
GENDER_VALUES = [MALE_VALUE, FEMALE_VALUE]

LABEL_VALUES = ["yes", "no"]
POSITIVE_VALUE = "no"

_TEXT_COLUMN = "text"
_LABEL_COLUMN = "label"

_TEXT_COLUMN_GENDERED = "TEXTO"
_LABEL_COLUMN_GENDERED = "VIOLENCIA"

_GENDER_ORDER = [MALE_VALUE, FEMALE_VALUE]

_MENTIONS_MALE_COLUMN = "Mención HOMBRE"
_MENTIONS_FEMALE_COLUMN = "Mención MUJER"
_MENTIONS_GENDER_COLUMNS = [_MENTIONS_MALE_COLUMN, _MENTIONS_FEMALE_COLUMN]

_TARGETS_MALE_COLUMN = "Dirigido HOMBRE"
_TARGETS_FEMALE_COLUMN = "Dirigido MUJER"
_TARGETS_GENDER_COLUMNS = [_TARGETS_MALE_COLUMN, _TARGETS_FEMALE_COLUMN]


def load_dataset(path=VILLANOS_DATASET, gendered=False, **kwargs):
    return (
        VillanosDataset.load(path)
        if not gendered
        else VillanosGenderedDataset.load(path)
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


class VillanosGenderedDataset(Dataset):
    @classmethod
    def load(cls, path):
        path = Path(path)

        collections = {}
        for split in ["manual"]:
            df = pd.read_csv(path / f"{split}.csv", sep=";")
            violent_list = df[_LABEL_COLUMN_GENDERED].apply(
                (lambda row: "yes" if row > 0 else "no" if row == 0 else None),
            )
            mentions_list = (
                df[_MENTIONS_GENDER_COLUMNS]
                .dropna()
                .apply(
                    lambda row: [
                        gender.lower()
                        for gender, active in zip(_GENDER_ORDER, row)
                        if active > 0
                    ],
                    axis=1,
                )
            )
            targets_list = (
                df[_TARGETS_GENDER_COLUMNS]
                .dropna()
                .apply(
                    lambda row: [
                        gender.lower()
                        for gender, active in zip(_GENDER_ORDER, row)
                        if active > 0
                    ],
                    axis=1,
                )
            )
            data = pd.concat(
                [
                    df[_TEXT_COLUMN_GENDERED].rename(TEXT_COLUMN),
                    violent_list.rename(LABEL_COLUMN),
                    mentions_list.rename(GENDER_COLUMN_A),
                    targets_list.rename(GENDER_COLUMN_B),
                ],
                join="inner",
                axis=1,
            )
            collections[split] = data

        return VillanosGenderedDataset(
            data=collections["manual"],
        )
