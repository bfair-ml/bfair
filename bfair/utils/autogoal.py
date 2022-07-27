import datetime
import time
from pathlib import Path

import numpy as np

from autogoal.search import Logger


class ClassifierWrapper:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self._fitted = False

    @property
    def fitted(self):
        return self._fitted

    def fit(self, X, y):
        self.pipeline.send("train")
        self.pipeline.run(X, y)
        self._fitted = True
        return self

    def predict(self, X):
        self.pipeline.send("eval")
        y_pred = self.pipeline.run(X, None)
        return y_pred

    @staticmethod
    def wrap(pipelines):
        return [ClassifierWrapper(p) for p in pipelines]

    @classmethod
    def wrap_and_fit(cls, pipelines, X, y):
        classifiers = cls.wrap(pipelines)
        for wrapped in classifiers:
            wrapped.fit(X, y)
        return classifiers

    def __str__(self) -> str:
        return str(self.pipeline)

    def __repr__(self) -> str:
        return repr(self.pipeline)


class FileLogger(Logger):
    def __init__(self, output_path: str):
        super().__init__()
        self.output_path = Path(output_path)
        self.start_time = None

    def begin(self, generations, pop_size):
        self.start_time = time.time()

    def end(self, best, best_fn):
        current_time = time.time()
        elapsed = int(current_time - self.start_time)
        elapsed = datetime.timedelta(seconds=elapsed)

        with self.output_path.open("a") as fd:
            fd.writelines(
                (
                    f"Search completed: best_fn={self.format_fitness(best_fn)}, best=\n{repr(best)}\n",
                    f"Time spent: elapsed={elapsed}\n",
                    "\n",
                )
            )


def split_input(X, y, validation_split=0.3):
    len_x = len(X) if isinstance(X, list) else X.shape[0]
    indices = np.arange(0, len_x)
    np.random.shuffle(indices)
    split_index = int(validation_split * len(indices))

    if split_index == 0:
        X_test, y_test = (X, y)
    else:
        train_indices = indices[:-split_index]
        test_indices = indices[-split_index:]

        if isinstance(X, list):
            X, y, X_test, y_test = (
                [X[i] for i in train_indices],
                y[train_indices],
                [X[i] for i in test_indices],
                y[test_indices],
            )
        else:
            X, y, X_test, y_test = (
                X[train_indices],
                y[train_indices],
                X[test_indices],
                y[test_indices],
            )

    return X, y, (X_test, y_test)


def join_input(X_train, y_train, X_test, y_test):
    if isinstance(X_train, list):
        X = X_train + X_test
        y = y_train + y_test
    else:
        X = np.concatenate((X_train, X_test))
        y = np.concatenate((y_train, y_test))
    return X, y


def succeeds_in_training_and_testing(X_train, y_train, X_test):
    def constraint(solution, fn):
        try:
            solution.send("train")
            solution.run(X_train, y_train)
            solution.send("eval")
            y_pred = solution.run(X_test, None)
            return True
        except:
            return False

    return constraint
