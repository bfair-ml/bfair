import argparse
import pandas as pd
from collections import defaultdict
from bfair.datasets import load_villanos
from bfair.datasets.villanos import (
    _MENTIONS_FEMALE_COLUMN,
    _MENTIONS_MALE_COLUMN,
    _TARGETS_FEMALE_COLUMN,
    _TARGETS_MALE_COLUMN,
    _TEXT_COLUMN_GENDERED,
    _LABEL_COLUMN_GENDERED,
    TARGETS,
    MENTIONS,
    TEXT_COLUMN,
    LABEL_COLUMN,
    FEMALE_VALUE,
    MALE_VALUE,
)

parser = argparse.ArgumentParser()
parser.add_argument("--objective", type=str, required=True, choices=[MENTIONS, TARGETS])
args = parser.parse_args()

OBJECTIVE = args.objective

ds = load_villanos(gendered=True)
ds.data[OBJECTIVE] = ds.data[OBJECTIVE].map(tuple)

sizes = defaultdict(lambda: float("inf"))
for violent, group_by_violent in ds.data.groupby(LABEL_COLUMN):
    for category, count in group_by_violent[OBJECTIVE].value_counts().items():
        sizes[len(category)] = min(sizes[len(category)], count)
sizes[0] = min(sizes[0], sizes[1] + sizes[2])
sizes = dict(sizes)
print(sizes)

selected = []
for violent, group_by_violent in ds.data.groupby(LABEL_COLUMN):
    print(violent, len(group_by_violent))

    for objective, group_by_objective in group_by_violent.groupby(OBJECTIVE):
        print(list(objective), len(group_by_objective), "->", sizes[len(objective)])
        selected.append(group_by_objective.sample(sizes[len(objective)]))

selected_df = pd.concat(selected)
selected_df.rename(
    columns={LABEL_COLUMN: _LABEL_COLUMN_GENDERED, TEXT_COLUMN: _TEXT_COLUMN_GENDERED},
    inplace=True,
)
selected_df[_LABEL_COLUMN_GENDERED] = selected_df[_LABEL_COLUMN_GENDERED].apply(
    lambda x: int(x == "yes")
)
selected_df[_MENTIONS_FEMALE_COLUMN] = selected_df[MENTIONS].apply(
    lambda x: int(FEMALE_VALUE in x)
)
selected_df[_MENTIONS_MALE_COLUMN] = selected_df[MENTIONS].apply(
    lambda x: int(MALE_VALUE in x)
)
selected_df[_TARGETS_FEMALE_COLUMN] = selected_df[TARGETS].apply(
    lambda x: int(FEMALE_VALUE in x)
)
selected_df[_TARGETS_MALE_COLUMN] = selected_df[TARGETS].apply(
    lambda x: int(MALE_VALUE in x)
)
selected_df.drop(columns=[MENTIONS, TARGETS], inplace=True)
selected_df.to_csv(
    f"villanos_balanced_by_{OBJECTIVE.lower()}.csv", sep=";", index=False
)
