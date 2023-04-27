import pandas as pd

from pathlib import Path
from bfair.envs import IMDB_REVIEWS_DATASET

from .base import Dataset

REVIEW_COLUMN = "Review"
GENDER_COLUMN = "Gender"
SENTIMENT_COLUMN = "Sentiment"

GENDER_VALUES = ["Male", "Female"]
SENTIMENT_VALUES = ["negative", "positive"]


def load_dataset(path=IMDB_REVIEWS_DATASET, split_seed=None):
    return ReviewsDataset.load(path, split_seed=split_seed)


class ReviewsDataset(Dataset):
    @classmethod
    def load(cls, path, split_seed=None):
        path = Path(path)
        data_path = path / "reviews.csv"

        data = pd.read_csv(
            data_path,
            engine="python",
            converters={GENDER_COLUMN: cls._parse_gender},
        )
        return ReviewsDataset(data, split_seed=split_seed, stratify_by=GENDER_COLUMN)

    @classmethod
    def _parse_gender(cls, cell):
        return tuple(
            sorted(
                clean
                for raw in str.split(cell, ",")
                for clean in (raw.strip(),)
                if clean
            )
        )
