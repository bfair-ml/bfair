import argparse
import pandas as pd

import pandas as pd
from pathlib import Path
from bfair.metrics.lm.bscore import BiasScore


def main(args):
    path = Path(args.path)
    scores_per_word = pd.read_csv(path, index_col="words")
    disparity_scores = scores_per_word[BiasScore.S_RATIO]
    selected = scores_per_word[disparity_scores.abs() >= args.disparity_threshold]
    selected.to_csv(
        path.with_name(
            f"{path.stem}-disparity-filtered-({args.disparity_threshold}){path.suffix}"
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    parser.add_argument("--disparity-threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
