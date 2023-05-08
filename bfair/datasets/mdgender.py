from .base import Dataset

import pandas as pd
import datasets as db

TEXT_COLUMN = "Text"
GENDER_COLUMN = "Gender"
CONFIDENCE_COLUMN = "Confidence"
TYPE_COLUMN = "Annotation Type"

MALE_VALUE = "Male"
FEMALE_VALUE = "Female"
GENDER_VALUES = [MALE_VALUE, FEMALE_VALUE]

_TEXT_COLUMN = "text"
_LABELS_COLUMN = "labels"
_CLASS_COLUMN = "class_type"
_CONFIDENCE_COLUMN = "confidence"

_ABOUT_VALUE = 0
_CERTAIN_VALUE = "certain"

_GENDER_MAP = {
    0: FEMALE_VALUE,
    1: MALE_VALUE,
    2: FEMALE_VALUE,
    3: MALE_VALUE,
    4: FEMALE_VALUE,
    5: MALE_VALUE,
}


def load_dataset(split_seed=None, **kwargs):
    return MDGenderDataset.load(split_seed=split_seed)


class MDGenderDataset(Dataset):
    @classmethod
    def load(cls, just_about=True, just_certain=True, split_seed=None):
        source = db.load_dataset("md_gender_bias", "new_data", split="train")

        df = pd.DataFrame.from_dict(source)
        if just_about:
            # for other `class_types` there are sentences that mention X gender but it is not annotated because it is not speaker or recipient.
            df = df[df[_CLASS_COLUMN] == _ABOUT_VALUE]

        if just_certain:
            df = df[df[_CONFIDENCE_COLUMN] == _CERTAIN_VALUE]

        gender = df[_LABELS_COLUMN].apply(lambda x: [_GENDER_MAP[key] for key in x])
        data = pd.concat(
            [
                df[_TEXT_COLUMN].rename(TEXT_COLUMN),
                df[_CONFIDENCE_COLUMN].rename(CONFIDENCE_COLUMN),
                df[_CLASS_COLUMN].rename(TYPE_COLUMN),
                gender.rename(GENDER_COLUMN),
            ],
            axis=1,
        )

        return MDGenderDataset(
            data=data,
            split_seed=split_seed,
            stratify_by=GENDER_COLUMN,
        )
