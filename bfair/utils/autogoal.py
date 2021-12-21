import datetime
import time
from pathlib import Path

from autogoal.search import Logger


class ClassifierWrapper:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def fit(self, X, y):
        self.pipeline.send("train")
        self.pipeline.run(X, y)
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
                    "Search completed: best_fn=%.3f, best=\n%r\n" % (best_fn, best),
                    f"Time spent: elapsed={elapsed}\n",
                    "\n",
                )
            )
