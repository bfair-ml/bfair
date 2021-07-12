from typing import List, Tuple

from autogoal.sampling import Sampler
from autogoal.search import PESearch
from bfair.utils import ClassifierWrapper

from .sampling import LogSampler, SampleModel


class AutoGoalEnsembler:
    def __init__(
        self, *, score_metric, errors="warn", allow_duplicates=False, **search_kwargs
    ):
        self.score_metric = score_metric
        self.search_kwargs = search_kwargs
        self.search_kwargs["errors"] = errors
        self.search_kwargs["allow_duplicates"] = allow_duplicates

    def __call__(
        self,
        X,
        y,
        classifiers: List[ClassifierWrapper],
        scores: List[float],
        **run_kwargs
    ):
        return self._optimize_sampler_fn(X, y, classifiers, scores, **run_kwargs)

    def _optimize_sampler_fn(
        self, X, y, classifiers, scores, **run_kwargs
    ) -> Tuple[SampleModel, float]:
        generator, fn = self._build_generator_and_fn(X, y, classifiers, scores)
        search = PESearch(
            generator_fn=generator,
            fitness_fn=fn,
            maximize=self.maximize,
            **self.search_kwargs,
        )

        best, best_fn = search.run(**run_kwargs)
        return best, best_fn

    def _build_generator_and_fn(self, X, y, classifiers, scores):
        def generator(sampler: Sampler, log=True):
            sampler = LogSampler(sampler) if log else sampler
            ensembler = ...  # TODO: use sampler to build ensembler
            return SampleModel(sampler, ensembler)

        def fn(generated: SampleModel):
            ensembler = generated.model
            y_pred = ensembler.predict(X)
            score = self.score_metric(y, y_pred)
            return score

        return generator, fn
