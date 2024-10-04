import pandas as pd
from pathlib import Path

from .base import Dataset
from bfair.utils import md2text

PROMPT = "prompt"
OUTPUT = "output"


def load_dataset(**kargs):
    return Victoria.load(**kargs)


class Victoria(Dataset):
    @classmethod
    def load(
        cls,
        model,
        leading: bool,
        path="datasets/victoria/LLMs",
        split_seed=None,
        stratify_by=None,
    ):
        path = Path(path)
        data_path = path / ("Leading" if leading else "No_leading") / f"{model}.csv"
        data = pd.read_csv(data_path, sep=";")
        data[OUTPUT] = data[OUTPUT].apply(md2text)
        return Victoria(data=data, split_seed=split_seed, stratify_by=stratify_by)

    @staticmethod
    def language():
        return "spanish"
