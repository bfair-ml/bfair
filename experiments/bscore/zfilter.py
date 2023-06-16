import argparse
import pandas as pd

from pathlib import Path
from scipy.stats import zscore
from bfair.metrics.lm.bscore import BiasScore


def main(args):
    path = Path(args.path)
    scores_per_word = pd.read_csv(path, index_col="words")
    log_scores = scores_per_word[BiasScore.S_LOG]
    zscores = zscore(log_scores)
    selected = scores_per_word[
        (zscores < -args.zscore_threshold) | (zscores > args.zscore_threshold)
    ]
    selected.to_csv(
        path.with_name(f"{path.stem}-z{args.zscore_threshold}filtered{path.suffix}")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    parser.add_argument("--zscore-threshold", type=int, default=3)
    args = parser.parse_args()
    main(args)
