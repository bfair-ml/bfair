import argparse
import spacy
import pandas as pd

from pathlib import Path


def main(args):
    path = Path(args.path)
    scores_per_word = pd.read_csv(path)

    nlp = spacy.load("en_core_web_sm")
    with_pos = scores_per_word["words"].apply(lambda x: nlp(x)[0].pos_ == args.pos)
    selected = scores_per_word[with_pos].set_index("words")
    selected.to_csv(path.with_name(f"{path.stem}-filtered-by-{args.pos}{path.suffix}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    parser.add_argument("--pos", default="ADJ")
    args = parser.parse_args()
    main(args)
