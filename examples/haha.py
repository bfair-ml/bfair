from autogoal.datasets import haha
from autogoal.kb import MatrixContinuousDense, Sentence, Seq
from autogoal.logging import logger
from autogoal.search import ConsoleLogger, ProgressLogger, RichLogger

from bfair.methods.autogoal import AutoGoalMitigator
from bfair.methods.autogoal.ensembling.ensembler import AutoGoalEnsembler

# ==================================================================================
# LOAD DATASET
# ==================================================================================
#

X_train, y_train, X_test, y_test = haha.load()

#
# ==================================================================================


# ==================================================================================
# FIND SOLUTION
# ==================================================================================
#

mitigator = AutoGoalMitigator.build(
    input=Seq[Sentence],
    n_classifiers=5,
    detriment=20,
    pop_size=10,
    search_iterations=1,
)
model = mitigator(X_train, y_train, logger=[ProgressLogger(), ConsoleLogger()])

#
# ==================================================================================
