import argparse
import spacy
import pandas as pd

from pathlib import Path
from bfair.metrics.lm.bscore import BiasScore


def main(args):
    path = Path(args.path)
    scores_per_word = pd.read_csv(path)

    if scores_per_word.empty:
        print(f"WARNING: Empty! ... skipping ({path})")
        return

    model_name = BiasScore.LANGUAGE2MODEL[args.language]
    nlp = spacy.load(model_name)
    with_pos = scores_per_word["words"].apply(lambda x: nlp(x)[0].pos_ == args.pos)
    selected = scores_per_word[with_pos].set_index("words")
    selected.to_csv(path.with_name(f"{path.stem}-filtered-by-{args.pos}{path.suffix}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    parser.add_argument("--language")
    parser.add_argument("--pos", default="ADJ")
    args = parser.parse_args()
    main(args)
