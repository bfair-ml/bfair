from pathlib import Path
import pandas as pd

from .base import Dataset


def load_from_path(path, split_seed=None):
    with Path(path).open() as fd:
        return load_from_file(fd, split_seed)


def load_from_file(fd, split_seed=None):
    return CustomDataset.load(fd, split_seed)


class CustomDataset(Dataset):
    def load(file, split_seed=None):
        data = pd.read_csv(file)
        return CustomDataset(data=data, split_seed=split_seed)
