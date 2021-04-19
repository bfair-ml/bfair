from operator import pos
from pandas import DataFrame, Series


def statistical_parity(
    data: DataFrame,
    protected_attributes,
    target_attribute,
    positive_target,
    *,
    return_probs=False,
):
    if isinstance(target_attribute, Series):
        positives = target_attribute == positive_target
    elif isinstance(target_attribute, str):
        positives = data[target_attribute] == positive_target
    else:
        raise TypeError()

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
