from collections import Counter, defaultdict
from functools import partial
from typing import List

import numpy as np
from bfair.utils.autogoal import ClassifierWrapper

from sklearn.metrics import accuracy_score


def stack_predictions(X, estimators):
    predictions = [model.predict(X) for model in estimators]
    return np.column_stack(predictions)


def get_most_common(items, prefered=None):
    counter = Counter(items)
    best, count = counter.most_common(1)[0]
    return best if prefered is None or counter.get(prefered, -1) < count else prefered


# TODO: per label scores


def get_most_common_with_score(items, scores, maximize=True):
    select_best = max if maximize else min
    worst_score = float("-inf") if maximize else float("+inf")

    counter = {}
    default = (0, worst_score)
    for item, score in zip(items, scores):
        count, best_score = counter.get(item, default)
        counter[item] = (count + 1, select_best(score, best_score))

    best = None
    best_count = -1
    best_score = worst_score
    for item, (count, score) in counter.items():
        if (
            count > best_count
            or count == best_count
            and select_best(score, best_score) != best_score
        ):
            best = item
            best_count = count
            best_score = score

    return best


class VotingClassifier:
    def __init__(self, estimators, scores=None, maximize=True):
        self.estimators = estimators
        self.scores = scores
        self.maximize = maximize

    @property
    def fitted(self):
        return True

    def fit_estimators(self, X, y):
        for estimator in self.estimators:
            estimator.fit(X, y)

    def fit(self, X, y, on_predictions=False, selection=None):
        pass

    def predict(self, X, on_predictions=False, selection=None):
        predictions = self._get_predictions(X, on_predictions, selection)
        y = self._forward_predictions(predictions)
        return y

    def _get_predictions(self, X, on_predictions=False, selection=None):
        if not on_predictions and selection is not None:
            raise ValueError(
                "If `selection` is provided, `on_predictions` must be `True`"
            )  # this is enforced by design. I don't want partially specified ensemblers

        return (
            self._stack_predictions(X)
            if not on_predictions
            else X[:, selection]
            if selection is not None
            else X
        )

    def _stack_predictions(self, X):
        return stack_predictions(X, self.estimators)

    def _forward_predictions(self, predictions):
        select = (
            get_most_common
            if self.scores is None
            else partial(
                get_most_common_with_score,
                scores=self.scores,
                maximize=self.maximize,
            )
        )
        most_commons = [select(sample) for sample in predictions]
        return np.asarray(most_commons)


class MLVotingClassifier(VotingClassifier):
    def __init__(
        self, estimators, scores=None, maximize=True, *, model=None, model_init=None
    ):
        if model is None == model_init is None:
            raise ValueError(
                "One and only one between `model` and `model_init` should be supplied"
            )
        super().__init__(estimators, scores, maximize)
        self.model = model_init() if model is None else model

    @property
    def fitted(self):
        return self.model.fitted

    def fit(self, X, y, on_predictions=False, selection=None):
        predictions = self._get_predictions(X, on_predictions, selection)
        self.model.fit(predictions, y)

    def _forward_predictions(self, predictions):
        return self.model.predict(predictions)


class OverfittedVotingClassifier(VotingClassifier):
    def __init__(self, estimators, scores, maximize=True):
        super().__init__(estimators, scores, maximize)
        self.best_index = np.argmax(scores) if maximize else np.argmin(scores)
        self.oracle = None

    @property
    def fitted(self):
        return self.oracle is not None

    def fit(self, X, y, on_predictions=False, selection=None):
        predictions = self._get_predictions(X, on_predictions, selection)
        self.oracle = self._build_oracle(predictions, y)

    @staticmethod
    def _build_oracle(predictions, y):
        group = defaultdict(list)
        for pred, gold in zip(predictions, y):
            key = tuple(pred)
            group[key].append(gold)
        return {
            key: Counter(value).most_common(1)[0][0] for key, value in group.items()
        }

    def _forward_predictions(self, predictions):
        return np.asarray(
            [
                self.oracle.get(tuple(pred), pred[self.best_index])
                for pred in predictions
            ]
        )


def optimistic_oracle(X, y, score_metric, estimators: List[ClassifierWrapper]):
    predictions = stack_predictions(X, estimators)
    y_pred = np.asarray(
        [gold if gold in pred else pred[0] for pred, gold in zip(predictions, y)]
    )
    return score_metric(y, y_pred)


def optimistic_oracle_coverage(X, y, estimators: List[ClassifierWrapper]):
    predictions = stack_predictions(X, estimators)
    y = y[np.newaxis].T  # row array to column array
    grid = predictions == y
    found = grid.any(axis=-1)
    correct = found.sum()
    return correct / len(y)


def overfitted_oracle(X, y, score_metric, estimators: List[ClassifierWrapper]):
    predictions = stack_predictions(X, estimators)
    oracle = OverfittedVotingClassifier._build_oracle(predictions, y)
    y_pred = np.asarray([oracle[tuple(pred)] for pred in predictions])
    return score_metric(y, y_pred)


def overfitted_oracle_coverage(X, y, estimators: List[ClassifierWrapper]):
    return overfitted_oracle(X, y, accuracy_score, estimators)
