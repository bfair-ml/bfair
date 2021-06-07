from pathlib import Path

import pandas as pd
from bfair.envs import GERMAN_DATASET

from .base import Dataset

ATTRIBUTES_NAMES = [
    "status of existing checking account",
    "duration in month",
    "credit history",
    "purpose",
    "credit amount",
    "savings account/bonds",
    "present employment since",
    "installment rate in percentage of disposable income",
    "personal status and sex",
    "other debtors / guarantors",
    "present residence since",
    "property",
    "age in years",
    "other installment plans",
    "housing",
    "number of existing credits at this bank",
    "job",
    "number of people being liable to provide maintenance for",
    "telephone",
    "foreign worker",
]

ATTRIBUTES_MAPPER = {
    "A11": "... < 0 DM",
    "A12": "0 <= ... < 200 DM",
    "A13": "... >= 200 DM / salary assignments for at least 1 year",
    "A14": "no checking account",
    "A30": "no credits taken/ all credits paid back duly",
    "A31": "all credits at this bank paid back duly",
    "A32": "existing credits paid back duly till now",
    "A33": "delay in paying off in the past",
    "A34": "critical account/ other credits existing (not at this bank)",
    "A40": "car (new)",
    "A41": "car (used)",
    "A42": "furniture/equipment",
    "A43": "radio/television",
    "A44": "domestic appliances",
    "A45": "repairs",
    "A46": "education",
    "A47": "(vacation - does not exist?)",
    "A48": "retraining",
    "A49": "business",
    "A410": "others",
    "A61": "... < 100 DM",
    "A62": "100 <= ... < 500 DM",
    "A63": "500 <= ... < 1000 DM",
    "A64": ".. >= 1000 DM",
    "A65": "unknown/ no savings account",
    "A71": "unemployed",
    "A72": "... < 1 year",
    "A73": "1 <= ... < 4 years",
    "A74": "4 <= ... < 7 years",
    "A75": ".. >= 7 years",
    "A91": "male : divorced/separated",
    "A92": "female : divorced/separated/married",
    "A93": "male : single",
    "A94": "male : married/widowed",
    "A95": "female : single",
    "A101": "none",
    "A102": "co-applicant",
    "A103": "guarantor",
    "A121": "real estate",
    "A122": "if not A121 : building society savings agreement/ life insurance",
    "A123": "if not A121/A122 : car or other, not in attribute 6",
    "A124": "unknown / no property",
    "A141": "bank",
    "A142": "stores",
    "A143": "none",
    "A151": "rent",
    "A152": "own",
    "A153": "for free",
    "A171": "unemployed/ unskilled - non-resident",
    "A172": "unskilled - resident",
    "A173": "skilled employee / official",
    "A174": "management/ self-employed/",
    "A191": "none",
    "A192": "yes, registered under the customers name",
    "A201": "yes",
    "A202": "no",
}


def load_dataset(path=GERMAN_DATASET, split_seed=None):
    return GermanDataset.load(path, split_seed=split_seed)


class GermanDataset(Dataset):
    @classmethod
    def load(cls, path, categorical=True, split_seed=None):
        path = Path(path)
        data_path = path / ("german.data" if categorical else "german.data-numeric")

        names = ATTRIBUTES_NAMES + ["risk"]

        data = pd.read_csv(
            data_path,
            sep="\s+",
            header=None,
            names=names,
            engine="python",
            index_col=False,
            dtype={"risk": object},
        )

        for feature in ATTRIBUTES_NAMES:
            if data.dtypes[feature] != object:
                continue
            data.loc[:, feature] = data[feature].apply(ATTRIBUTES_MAPPER.get)

        data.loc[:, "risk"] = data["risk"].apply({"1": "good", "2": "bad"}.get)

        return GermanDataset(data, split_seed=split_seed)

    @staticmethod
    def cost_matrix(gold, prediction):
        if gold == prediction:
            return 0
        return 5 if gold == "bad" else 1
