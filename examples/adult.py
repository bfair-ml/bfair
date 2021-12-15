from autogoal.datasets import haha
from autogoal.kb import MatrixContinuousDense, Sentence, Seq
from autogoal.logging import logger
from autogoal.search import ConsoleLogger, ProgressLogger, RichLogger

from bfair.datasets import load_adult
from bfair.methods.autogoal import AutoGoalMitigator
from bfair.methods.autogoal.ensembling.ensembler import AutoGoalEnsembler
from bfair.metrics import statistical_parity
from bfair.utils import encode_features

# ==================================================================================
# LOAD DATASET
# ==================================================================================
#

dataset = load_adult()
data = dataset.data
feature_names = data.columns[:-1]
target_name = data.columns[-1]
X_train, y_train, encoders = encode_features(
    data, target=target_name, feature_names=feature_names
)

#
# ==================================================================================


# ==================================================================================
# FIND SOLUTION
# ==================================================================================
#

protected_attributes = ["sex"]
attr_indexes = [data.columns.get_loc(attrib) for attrib in protected_attributes]
positive_target = encoders[target_name].transform([">50K"]).item()

mitigator = AutoGoalMitigator.build(
    input=MatrixContinuousDense,
    n_classifiers=5,
    detriment=20,
    pop_size=10,
    search_iterations=1,
    protected_attributes=protected_attributes,
    fairness_metric=statistical_parity,
    target_attribute=target_name,
    positive_target=">50K",
    sensor=(lambda X: X[:, attr_indexes]),
)
model = mitigator(X_train, y_train, logger=[ProgressLogger(), ConsoleLogger()])

#
# ==================================================================================
