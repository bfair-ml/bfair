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

    def __str__(self) -> str:
        return str(self.pipeline)

    def __repr__(self) -> str:
        return repr(self.pipeline)
