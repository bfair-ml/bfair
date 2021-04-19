from typing import List, Union

from pandas import DataFrame, Series


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
