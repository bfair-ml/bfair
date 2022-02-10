from autogoal.datasets import haha
from autogoal.kb import Sentence, Seq
from sklearn.metrics import f1_score

from experiments.core import run, setup


def main():
    args = setup()
    run(
        load_dataset=haha.load,
        input_type=Seq[Sentence],
        score_metric=f1_score,
        maximize=True,
        args=args,
    )


if __name__ == "__main__":
    main()
