from typing import List, Union

from pandas import DataFrame, Series


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
        return_probs=False,
    ):
        return {
            key: metric(
                data=data,
                protected_attributes=protected_attributes,
                target_attribute=target_attribute,
                target_predictions=target_predictions,
                positive_target=positive_target,
                return_probs=return_probs,
            )
            for key, metric in self.metrics.items()
        }


def statistical_parity(
    *,
    data: DataFrame,
    protected_attributes: Union[List[str], str],
    target_attribute: str,
    positive_target,
    target_predictions: Series = None,
    return_probs=False,
):
    if target_predictions is None:
        target_predictions = data[target_attribute]

    positives = target_predictions == positive_target

    probs = {
        key: len(group[positives]) / len(group)
        for key, group in data.groupby(protected_attributes)
    }

    min_group = min(probs.values())
    max_group = max(probs.values())
    if return_probs:
        return max_group - min_group, probs
    else:
        return max_group - min_group


def equal_opportunity(
    *,
    data: DataFrame,
    protected_attributes: Union[List[str], str],
    target_attribute: str,
    target_predictions: Series,
    positive_target,
    return_probs=False,
):
    positives = target_predictions == positive_target
    true_positives = data[data[target_attribute] == positive_target]

    probs = {
        key: len(group[positives]) / len(group)
        for key, group in true_positives.groupby(protected_attributes)
    }

    min_group = min(probs.values())
    max_group = max(probs.values())
    if return_probs:
        return max_group - min_group, probs
    else:
        return max_group - min_group
