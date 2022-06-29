from typing import Any, Callable, List, Tuple, Union

import numpy as np
from autogoal.kb import MatrixCategorical, Supervised, VectorCategorical
from autogoal.ml import AutoML
from autogoal.sampling import Sampler
from autogoal.search import PESearch
from bfair.methods.voting import (
    MLVotingClassifier,
    OverfittedVotingClassifier,
    VotingClassifier,
    stack_predictions,
)
from bfair.utils import ClassifierWrapper
from bfair.utils.autogoal import split_input

from .sampling import LogSampler, SampleModel


class AutoGoalEnsembler:
    def __init__(
        self,
        *,
        score_metric: Callable[[Any, Any, Any], Union[float, List[float]]],
        maximize: Union[bool, List[bool]],
        validation_split=0.3,
        errors="warn",
        allow_duplicates=False,
        include_filter=".*",
        exclude_filter=None,
        registry=None,
        search_algorithm=PESearch,
        **search_kwargs,
    ):
        self.score_metric = score_metric
        self.validation_split = validation_split
        self.maximize = maximize
        self.search_algorithm = search_algorithm
        self.search_kwargs = search_kwargs
        self.search_kwargs["errors"] = errors
        self.search_kwargs["allow_duplicates"] = allow_duplicates

        self._automl_scaffold = AutoML(
            input=(MatrixCategorical, Supervised[VectorCategorical]),
            output=VectorCategorical,
            include_filter=include_filter,
            exclude_filter=exclude_filter,
            registry=registry,
        )
        self._pipeline_space = None

    def _init_space(self):
        if self._pipeline_space is None:
            self._pipeline_space = self._automl_scaffold.make_pipeline_builder()
        return self._pipeline_space

    def __call__(
        self,
        X,
        y,
        classifiers: List[ClassifierWrapper],
        scores: List[float],
        maximized: bool,
        *,
        test_on=None,
        pre_caching=True,
        **run_kwargs,
    ):
        self._init_space()
        return self._optimize_sampler_fn(
            X,
            y,
            classifiers,
            scores,
            maximized,
            test_on=test_on,
            pre_caching=pre_caching,
            **run_kwargs,
        )

    def _optimize_sampler_fn(
        self,
        X,
        y,
        classifiers,
        scores,
        maximized,
        *,
        test_on=None,
        pre_caching=True,
        **run_kwargs,
    ) -> Tuple[SampleModel, float]:

        if test_on is None:
            X, y, (X_test, y_test) = split_input(X, y, self.validation_split)
        else:
            X_test, y_test = test_on
        generator, fn = self._build_generator_and_fn(
            X,
            y,
            X_test,
            y_test,
            classifiers,
            scores,
            pre_caching,
            maximized,
        )
        search = self.search_algorithm(
            generator_fn=generator,
            fitness_fn=fn,
            maximize=self.maximize,
            **self.search_kwargs,
        )

        best, best_fn = search.run(**run_kwargs)
        return best, best_fn

    def _build_generator_and_fn(
        self, X, y, X_test, y_test, classifiers, scores, maximized, pre_caching
    ):
        named_classifiers = {str(c): c for c in classifiers}
        classifier2index = {str(c): i for i, c in enumerate(classifiers)}
        ensembler_types = ["voting", "learning", "overfit"]

        e_input = stack_predictions(X, classifiers) if pre_caching else X
        e_input_test = stack_predictions(X_test, classifiers) if pre_caching else X_test

        model_generator = self._pipeline_space

        def build_ensembler(sampler: LogSampler):
            n_classifiers = sampler.discrete(
                min=min(2, len(classifiers)),
                max=len(classifiers),
                handle="n_classifiers",
            )
            names = sampler.multichoice(
                options=named_classifiers,
                k=n_classifiers,
                handle="selected_classifiers",
            )
            selected_classifiers = [named_classifiers[n] for n in names]
            indexes = [classifier2index[n] for n in names]
            selected_scores = [scores[i] for i in indexes]

            ensembler_type = sampler.choice(ensembler_types, handle="ensembler_type")
            if ensembler_type == "voting":
                ensembler = VotingClassifier(
                    selected_classifiers, selected_scores, maximized
                )
            elif ensembler_type == "learning":
                pipeline = model_generator(sampler)
                model = ClassifierWrapper(pipeline)
                ensembler = MLVotingClassifier(
                    selected_classifiers, selected_scores, maximized, model=model
                )
            elif ensembler_type == "overfit":
                ensembler = OverfittedVotingClassifier(
                    selected_classifiers, selected_scores, maximized
                )
            else:
                raise Exception(f"Unknown ensembler_type: {ensembler_type}")
            return ensembler, indexes

        def generator(sampler: Sampler):
            sampler = LogSampler(sampler)
            ensembler, indexes = build_ensembler(sampler)
            ensembler.fit(
                e_input,
                y,
                on_predictions=pre_caching,
                selection=indexes,
            )
            return SampleModel(sampler, ensembler, indexes=indexes)

        def fn(generated: SampleModel):
            ensembler = generated.model
            indexes = generated.info["indexes"]
            y_pred = ensembler.predict(
                e_input_test,
                on_predictions=pre_caching,
                selection=indexes,
            )
            score = self.score_metric(X_test, y_test, y_pred)
            return score

        return generator, fn
