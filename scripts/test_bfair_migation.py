from functools import partial
from autogoal.kb import MatrixContinuousDense
from autogoal.search import PESearch, ConsoleLogger
from sklearn.metrics import accuracy_score
from bfair.methods import AutoGoalMitigator
from bfair.metrics import DIFFERENCE, double_fault_inverse, statistical_parity
from experiments.adult import encode_dataset, load_adult, sensor

dataset = load_adult()
data, encoders = encode_dataset(dataset, target_attribute="income")
positive_target = encoders["income"].transform([">50K"])[0]
protected_indexes = [dataset.columns.get_loc(attr) for attr in ["sex"]]
mitigator = AutoGoalMitigator.build(
    input=MatrixContinuousDense,
    n_classifiers=5,
    detriment=20,
    score_metric=accuracy_score,
    diversity_metric=double_fault_inverse,
    fairness_metrics=statistical_parity,
    ranking_fn=None,
    maximize=True,
    maximize_fmetric=False,
    protected_attributes=["sex"],
    target_attribute="income",
    positive_target=positive_target,
    sensor=partial(sensor, indexes=protected_indexes),
    metric_kwargs=dict(
        mode=DIFFERENCE,
    ),
    # [start] AutoML args [start]
    #
    search_algorithm=PESearch,
    pop_size=10,
    search_iterations=10,
    search_timeout=60 * 60,
    errors="warn",
    #
    # [ end ] AutoML args [ end ]
)
X_train, X_test, y_train, y_test = data
pipelines, scores = mitigator.diversify(
    X_train,
    y_train,
    logger=[ConsoleLogger()],
)
model, score = mitigator.ensemble(
    pipelines,
    scores,
    X_train,
    y_train,
    logger=[ConsoleLogger()],
)
