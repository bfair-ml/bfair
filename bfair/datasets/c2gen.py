from .base import Dataset

import datasets as db


CONTEXT = "context"
KEYWORDS = "keywords"


def load_dataset(split_seed=None):
    return C2Gen.load(split_seed=split_seed)


class C2Gen(Dataset):
    @classmethod
    def load(cls, split_seed=None):
        source = db.load_dataset("Non-Residual-Prompting/C2Gen", "c2gen", split="test")
        data = source.to_pandas()
        return C2Gen(data=data, split_seed=split_seed)
    
    @staticmethod
    def language():
        return "english"
