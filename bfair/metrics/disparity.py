from typing import Callable, List, Union

import pandas as pd
from pandas import DataFrame, Series

DIFFERENCE = "difference"
RATIO = "ratio"


class MetricHandler:
    def __init__(self, *args, named_metrics: dict = {}, **kwargs):
        self.metrics = {}
        for metric in args:
            self.metrics[metric.__name__] = metric
        self.metrics.update(named_metrics)
        self.metrics.update(kwargs)

    def __call__(
        self,
        *,
        data: DataFrame,
        protected_attributes: Union[List[str], str],
        target_attribute: str,
        target_predictions: Series,
        positive_target,
        mode: Union[str, Callable[..., float]] = DIFFERENCE,
        return_probs=False,
    ):
        """
        ## Parameters

        `data`: DataFrame with all features (including the protected and target attributes)

        `protected_attributes`: A single attribute name or a list of them.

        `target_attribute`: Name of the attribute used to discriminate the items.

        `target_predictions`: Series of predicted values for the discrimination attribute.

        `positive_target`: Value of the `target_attribute` that indicates a positive discrimination.

        `mode`:
            - `difference`: compute the difference between the highest scored and least scored groups.
            - `ratio`: compute the ratio between the highest scored and least scored groups.
            - `Callable`: compute a custom score between the highest scored and least scored groups.

        `return_probs`: Whether to return the computed probabilities in addition the score.

        ## Returns

        `out`: `(value, probs)` if `return_probs` else `value`
        """
        return {
            key: metric(
                data=data,
                protected_attributes=protected_attributes,
                target_attribute=target_attribute,
                target_predictions=target_predictions,
                positive_target=positive_target,
                mode=mode,
                return_probs=return_probs,
            )
            for key, metric in self.metrics.items()
        }

    @staticmethod
    def to_df(computed_metrics):
        data = {}
        for metric, info in computed_metrics.items():
            if isinstance(info, (tuple, list)):
                value, probs = info
                row = {"value": value}
                row.update(probs)
            else:
                row = {"value": info}
            data[metric] = row
        return DataFrame.from_dict(data, orient="index")


def base_metric(
    *,
    data: DataFrame,
    protected_attributes: Union[List[str], str],
    target_attribute: str,
    target_predictions: Series,
    positive_target,
    mode: Union[str, Callable[..., float]] = DIFFERENCE,
    return_probs: bool = False,
    **kwargs,
) -> float:
    """
    ## Parameters

    `data`: DataFrame with all features (including the protected and target attributes)

    `protected_attributes`: A single attribute name or a list of them.

    `target_attribute`: Name of the attribute used to discriminate the items.

    `target_predictions`: Series of predicted values for the discrimination attribute.

    `positive_target`: Value of the `target_attribute` that indicates a positive discrimination.

    `mode`:
        - `difference`: compute the difference between the highest scored and least scored groups.
        - `ratio`: compute the ratio between the highest scored and least scored groups.
        - `Callable`: compute a custom score between the highest scored and least scored groups.

    `return_probs`: Whether to return the computed probabilities in addition the score.

    **kwargs: Additional parameters to ignore.

    ## Returns

    `out`: `(value, probs)` if `return_probs` else `value`
    """
    pass


def disparity_metric(func: Callable) -> base_metric:
    def metric(*args, mode=DIFFERENCE, return_probs=False, **kwargs):
        probs = func(*args, mode=mode, **kwargs)
        min_group = min(probs.values())
        max_group = max(probs.values())

        value = (
            max_group - min_group
            if mode == DIFFERENCE
            else (1 - min_group / max_group if max_group != 0 else 0)
            if mode == RATIO
            else mode(min_group=min_group, max_group=max_group)
        )

        return (value, probs) if return_probs else value

    metric.__name__ = func.__name__
    return metric


def accuracy(
    *,
    data: DataFrame,
    target_attribute: str,
    target_predictions: Series,
    **kwargs,
):
    gold_target = data[target_attribute]
    equals = gold_target == target_predictions
    return sum(equals) / len(equals)


@disparity_metric
def accuracy_disparity(
    *,
    data: DataFrame,
    protected_attributes: Union[List[str], str],
    target_attribute: str,
    target_predictions: Series,
    **kwargs,
):
    all_data = pd.concat(
        (data, Series(target_predictions, name="__prediction__")), axis=1
    )

    scores = {
        key: sum(equals) / len(equals)
        for key, group in all_data.groupby(protected_attributes)
        for equals in (group[target_attribute] == group["__prediction__"],)
    }

    return scores


@disparity_metric
def statistical_parity(
    *,
    data: DataFrame,
    protected_attributes: Union[List[str], str],
    target_attribute: str,
    positive_target,
    target_predictions: Series = None,
    **kwargs,
):
    if target_predictions is None:
        target_predictions = data[target_attribute]

    positives = target_predictions[target_predictions == positive_target]

    probs = {
        key: len(group.index & positives.index) / len(group)
        for key, group in data.groupby(protected_attributes)
    }

    return probs


@disparity_metric
def equal_opportunity(
    *,
    data: DataFrame,
    protected_attributes: Union[List[str], str],
    target_attribute: str,
    target_predictions: Series,
    positive_target,
    **kwargs,
):
    positives = target_predictions[target_predictions == positive_target]
    true_positives = data[data[target_attribute] == positive_target]

    probs = {
        key: len(group.index & positives.index) / len(group)
        for key, group in true_positives.groupby(protected_attributes)
    }

    return probs


@disparity_metric
def false_positive_rate(
    *,
    data: DataFrame,
    protected_attributes: Union[List[str], str],
    target_attribute: str,
    target_predictions: Series,
    positive_target,
    **kwargs,
):
    positives = target_predictions[target_predictions == positive_target]
    true_negatives = data[data[target_attribute] != positive_target]

    probs = {
        key: len(group.index & positives.index) / len(group)
        for key, group in true_negatives.groupby(protected_attributes)
    }

    return probs


def equalized_odds(
    *,
    data: DataFrame,
    protected_attributes: Union[List[str], str],
    target_attribute: str,
    target_predictions: Series,
    positive_target,
    mode: Union[str, Callable[..., float]] = DIFFERENCE,
    return_probs: bool = False,
    **kwargs,
):
    def f1(prob1, prob2):
        try:
            return 2 * prob1 * prob2 / (prob1 + prob2)
        except ZeroDivisionError:
            return 0

    eo_value, eo_probs = equal_opportunity(
        data=data,
        protected_attributes=protected_attributes,
        target_attribute=target_attribute,
        target_predictions=target_predictions,
        positive_target=positive_target,
        mode=mode,
        return_probs=True,
    )
    fpr_value, fpr_probs = false_positive_rate(
        data=data,
        protected_attributes=protected_attributes,
        target_attribute=target_attribute,
        target_predictions=target_predictions,
        positive_target=positive_target,
        mode=mode,
        return_probs=True,
    )

    value = 1 - f1(1 - eo_value, 1 - fpr_value)

    probs = {
        key: f1(eo_probs.get(key, 0), fpr_probs.get(key, 0))
        for key in eo_probs.keys() | fpr_probs.keys()
    }

    return (value, probs) if return_probs else value
