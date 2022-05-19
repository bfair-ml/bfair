from functools import partial

from autogoal import datasets
from autogoal.datasets import (
    abalone,
    cars,
    dorothea,
    german_credit,
    gisette,
    shuttle,
    yeast,
)
from autogoal.kb import MatrixContinuousDense
from bfair.utils.autogoal import succeeds_in_training_and_testing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from experiments.core import run, setup

valid_datasets = [
    "abalone",
    "cars",
    "dorothea",
    "gisette",
    "shuttle",
    "yeast",
    "german_credit",
]


def get_dataset_and_loader(name):
    data = getattr(datasets, name).load()

    if len(data) == 4:
        X_train, y_train, X_test, y_test = data
    else:
        X, y = data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    def load_dataset(max_examples=None):
        if max_examples is None:
            return X_train, y_train, X_test, y_test
        else:
            return (
                X_train[:max_examples],
                y_train[:max_examples],
                X_test[:max_examples],
                y_test[:max_examples],
            )

    return (X_train, X_test, y_train, y_test), load_dataset


def main():
    args = setup()
    selected_datasets = [args.title] if args.title in valid_datasets else valid_datasets
    for name in selected_datasets:
        (X_train, X_test, y_train, _), load_dataset = get_dataset_and_loader(name)
        run(
            load_dataset=load_dataset,
            input_type=MatrixContinuousDense,
            score_metric=accuracy_score,
            maximize=True,
            args=args,
            title=name,
            diversifier_run_kwargs=dict(
                constraint=succeeds_in_training_and_testing(X_train, y_train, X_test)
            ),
        )


if __name__ == "__main__":
    main()
