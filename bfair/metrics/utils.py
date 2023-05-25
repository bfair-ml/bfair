def delta_percent(estimated_fairness, true_fairness, scale=0.01):
    if (
        not (0 <= estimated_fairness <= 1)
        or not (0 <= true_fairness <= 1)
        or not (0 <= scale <= 1)
    ):
        raise ValueError()
    return int(abs(estimated_fairness - true_fairness) / scale) * scale * 100
