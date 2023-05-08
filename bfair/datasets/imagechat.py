from .base import Dataset

import pandas as pd
import datasets as db

MALE_VALUE = "Male"
FEMALE_VALUE = "Female"
GENDER_VALUES = [MALE_VALUE, FEMALE_VALUE]

TEXT_COLUMN = "Text"
GENDER_COLUMN = "Gender"

_CAPTION_COLUMN = "caption"
_MALE_COLUMN = "male"
_FEMALE_COLUMN = "female"
_GENDER_COLUMNS = [_MALE_COLUMN, _FEMALE_COLUMN]


def load_dataset(**kwargs):
    return ImageChatDataset.load()


class ImageChatDataset(Dataset):
    @classmethod
    def load(cls):
        source = db.load_dataset("md_gender_bias", "image_chat")

        collections = {}
        for split in source:
            df = source[split].to_pandas()
            gender_list = df[_GENDER_COLUMNS].apply(
                lambda row: [
                    gender.title()
                    for gender, include in zip(_GENDER_COLUMNS, row)
                    if include
                ],
                axis=1,
            )
            data = pd.concat(
                [
                    df[_CAPTION_COLUMN].rename(TEXT_COLUMN),
                    gender_list.rename(GENDER_COLUMN),
                ],
                axis=1,
            )
            collections[split] = data

        return ImageChatDataset(
            data=collections["train"],
            validation=collections["validation"],
            test=collections["test"],
        )
