import numpy as np
from numpy import ndarray
from sklearn.metrics import classification_report


def build_oracle_output_matrix(
    gold: ndarray, predictions: ndarray, weights: ndarray = None
) -> ndarray:
    """
    ## Parameters

    `gold`: ndarray (Nj,)

    `predictions`: ndarray (Ni, Nj)
        - `i`: classifier.
        - `j`: training sample.

    `weights`: ndarray (Ni,)
        - `sum(weights) == 1`
        - `all(weights > 0)`

    ## Returns

    `out`: ndarray (Nj, Ni)
        - `out_ji == 1` if training sample `x_j` is classified correctly by classifier `h_i`.
        - `out_ji == -1` otherwise.
    """
    gold = np.asarray(gold)
    predictions = np.asarray(predictions)

    if weights is not None:
        weights = np.asarray(weights)

        if weights.sum() != 1:
            raise ValueError("Weights must add up to 1.")
        if (weights < 0).any():
            raise ValueError("All weights must be positive.")

    def correct(predictions):
        return predictions == gold

    matrix = np.apply_along_axis(correct, axis=1, arr=predictions)
    signed = matrix.transpose() * 2 - 1
    weighted = signed if weights is None else signed * weights
    return weighted


def disagreement(oracle_matrix: ndarray) -> ndarray:
    """
    The diversity increases with the value of the disagreement measure.

    ## Parameters

    `oracle_matrix`: ndarray (Ni, Nj)
        - `i`: training sample.
        - `j`: classifier.

    ## Returns

    `out`: ndarray (Ni, Ni)
    """
    n_samples = oracle_matrix.shape[0]

    def measure(classifier):
        diff = np.not_equal(oracle_matrix.transpose(), classifier)
        count = np.count_nonzero(diff, axis=1)
        return count / n_samples

    return np.apply_along_axis(measure, axis=0, arr=oracle_matrix)


def double_fault(oracle_matrix: ndarray) -> ndarray:
    """
    The diversity decreases when the value of the double-fault measure increases.

    ## Parameters

    `oracle_matrix`: ndarray (Ni, Nj)
        - `i`: training sample.
        - `j`: classifier.

    ## Returns

    `out`: ndarray (Ni, Ni)
    """

    n_samples = oracle_matrix.shape[0]

    def measure(classifier):
        mistake = classifier < 0
        same = oracle_matrix.transpose() == classifier
        count = np.count_nonzero(same & mistake, axis=1)
        return count / n_samples

    return np.apply_along_axis(measure, axis=0, arr=oracle_matrix)


def double_fault_inverse(oracle_matrix: ndarray) -> ndarray:
    """
    The diversity increases when the value of the double-fault measure increases.

    ## Parameters

    `oracle_matrix`: ndarray (Ni, Nj)
        - `i`: training sample.
        - `j`: classifier.

    ## Returns

    `out`: ndarray (Ni, Ni)
    """

    return 1 - double_fault(oracle_matrix)


def kohavi_wolpert_variance(oracle_matrix: ndarray) -> ndarray:
    """
    The diversity increases as the value of the variance increases


    ## Parameters

    `oracle_matrix`: ndarray (Ni, Nj)
        - `i`: training sample.
        - `j`: classifier.

    ## Returns

    `out`: ndarray (Ni,)
    """

    # This is computed this way because we are assuming equally weighted classsifiers
    probs = np.apply_along_axis(lambda sample: (sample < 0) / 2, axis=1, arr=oracle_matrix)

    n_classifiers = oracle_matrix.shape[1]
    kw_variance = np.zeros((n_classifiers, n_classifiers))
    for i in range(n_classifiers):
        for j in range(n_classifiers):
            classifiers_probs = np.sum(probs[:, (i, j)], axis=1)
            kw_variance[i, j] = np.sum(classifiers_probs * (1-classifiers_probs))

    return kw_variance
