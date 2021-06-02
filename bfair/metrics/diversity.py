import numpy as np
from numpy import ndarray


def build_oracle_output_matrix(
    gold: ndarray, predictions: ndarray, weights: ndarray = None
):
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
        return np.equal(predictions, gold)

    matrix = np.apply_along_axis(correct, axis=1, arr=predictions)
    signed = matrix.transpose() * 2 - 1
    weighted = signed if weights is None else signed * weights
    return weighted


def disagreement(oracle_matrix: ndarray):
    """
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
