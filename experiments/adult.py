from functools import partial

from autogoal.kb import MatrixContinuousDense
from bfair.datasets import load_adult
from bfair.utils import encode_features
from bfair.utils.autogoal import succeeds_in_training_and_testing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from experiments.core import run, setup


def encode_dataset(dataset, target_attribute="income"):
    X_train, y_train, encoders = encode_features(dataset.data, target=target_attribute)
    X_test, y_test, _ = encode_features(
        dataset.test, target=target_attribute, source_encoders=encoders
    )
    return (X_train, X_test, y_train, y_test), encoders


def load_dataset(data, max_examples=None):
    if len(data) == 4:
        X_train, X_test, y_train, y_test = data
    else:
        X, y = data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return (
        X_train[:max_examples],
        y_train[:max_examples],
        X_test[:max_examples],
        y_test[:max_examples],
    )


def sensor(X, indexes):
    return X[:, indexes]


def main():
    dataset = load_adult()

    target_attribute = "income"
    protected_attributes = ["sex"]  # ["race", "sex", "marital-status", ...]

    data, encoders = encode_dataset(dataset, target_attribute=target_attribute)
    X_train, X_test, y_train, y_test = data

    positive_target = encoders[target_attribute].transform([">50K"])[0]
    protected_indexes = [dataset.columns.get_loc(attr) for attr in protected_attributes]

    args = setup()
    run(
        load_dataset=partial(load_dataset, data=data),
        input_type=MatrixContinuousDense,
        score_metric=accuracy_score,
        maximize=True,
        args=args,
        title="adult",
        protected_attributes=protected_attributes,
        target_attribute=target_attribute,
        positive_target=positive_target,
        sensor=partial(sensor, indexes=protected_indexes),
        diversifier_run_kwargs=dict(
            constraint=succeeds_in_training_and_testing(X_train, y_train, X_test)
        ),
    )


if __name__ == "__main__":
    main()
