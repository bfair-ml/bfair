import math
import argparse
import pandas as pd
import json

from pathlib import Path
from statistics import mean, stdev
from bfair.metrics.lm.bscore import BiasScore

def main(args):
    path = Path(args.path)
    scores_per_word = pd.read_csv(path)
    results = {}
    
    for scoring_mode in [BiasScore.S_RATIO, BiasScore.S_LOG]:
        column = scores_per_word[scoring_mode]
        not_infinity = [s for s in column if not math.isinf(s)]
        results[scoring_mode] = {
            "mean": mean(not_infinity),
            "standard_deviation": stdev(not_infinity)
        }
    
    print(json.dumps({ path.name: results }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    args = parser.parse_args()
    main(args)
