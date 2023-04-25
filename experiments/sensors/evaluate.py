import sys
import pandas as pd

from bfair.sensors import P_GENDER
from bfair.sensors.optimization import load, compute_errors, compute_scores
from bfair.datasets import load_review
from bfair.datasets.reviews import REVIEW_COLUMN, GENDER_COLUMN

GENDER_VALUES = ["Male", "Female"]


def main():
    if len(sys.argv) != 2:
        print("Wrong number of arguments!")
        exit()

    config_str = sys.argv[1]

    try:
        config = eval(config_str)
    except Exception as e:
        print(f"Invalid handler configuration. {e}")
        exit()

    config = list(config.items())

    generated = load(config)
    handler = generated.model
    print("Loaded!")

    dataset = load_review(split_seed=0)
    X = dataset.data[REVIEW_COLUMN]
    y = dataset.data[GENDER_COLUMN]

    predictions = [handler(review, GENDER_VALUES, P_GENDER) for review in X]

    errors = compute_errors(y, predictions, GENDER_VALUES)
    print(errors)

    scores = compute_scores(errors)
    print(scores)

    results = pd.concat(
        (
            X,
            y.str.join(" & "),
            pd.Series(predictions, name="Predicted").str.join(" & "),
        ),
        axis=1,
    )
    print(results)


if __name__ == "__main__":
    main()
