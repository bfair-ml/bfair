import pandas as pd
from pathlib import Path

from .base import Dataset


PROMPT = "prompt"
OUTPUT = "output"


def load_dataset(**kargs):
    return Victoria.load(**kargs)


class Victoria(Dataset):
    @classmethod
    def load(cls, model, path="datasets/victoria", split_seed = None, stratify_by = None):
        path = Path(path)
        data_path = path / "02_Respuestas" / f"{model}.csv"
        data = pd.read_csv(data_path, sep=";")
        return Victoria(data=data, split_seed=split_seed, stratify_by=stratify_by)

    @staticmethod
    def language():
        return "spanish"
