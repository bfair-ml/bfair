import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from streamlit.config import _server_enable_xsrf_protection

from bfair.datasets import load_adult, load_german
from bfair.datasets.custom import load_from_file
from bfair.methods import SklearnMitigator
from bfair.metrics import (
    DIFFERENCE,
    RATIO,
    MetricHandler,
    accuracy,
    accuracy_disparity,
    equal_opportunity,
    equalized_odds,
    false_positive_rate,
    statistical_parity,
)
from bfair.utils import encode_features


# = DATASET =================================================
@st.cache
def load_dataset(name):
    if name == "adult":
        return load_adult()
    if name == "german":
        return load_german()
    else:
        raise ValueError(f"Unknown dataset: {name}")


dataset_name = st.sidebar.selectbox("name", ["custom", "adult", "german"])
if dataset_name == "custom":
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt"])
    if uploaded_file is None:
        st.stop()
    dataset = load_from_file(uploaded_file)
else:
    dataset = load_dataset(dataset_name)

df: pd.DataFrame = dataset.data
df_test: pd.DataFrame = dataset.test

f"# Dataset: _{dataset_name}_"
with st.beta_expander("Explore data"):
    "**Training set**"
    df
    "**Test set**"
    df_test

# = FEATURES =================================================
feature_names = df.columns[:-1]
X = df[feature_names]

with st.beta_expander("Features"):
    "#### All features"
    X
    for feature in feature_names:
        f"#### {feature}"
        values = df[feature].unique()
        values.sort()
        values
        f"**Total:** {len(values)}"

# = TARGET =================================================
target_name = df.columns[-1]
y = df[target_name]

with st.beta_expander("Target"):
    y

labels = set(y.unique())
assert len(labels) == 2
positive_target = st.sidebar.selectbox("positive_target", list(labels))
negative_target = (labels - {positive_target}).pop()

f"**Positive target:** {positive_target}"
f"**Negative target:** {negative_target}"

# = CLASSIFIER =================================================

X, y, encoders = encode_features(df, target=target_name, feature_names=feature_names)
encoder = encoders[target_name]

X_test, y_test, _ = encode_features(
    df_test, target=target_name, feature_names=feature_names, source_encoders=encoders
)

use_test = not df_test.empty and st.sidebar.checkbox("Test")
evaluation_df = df_test if use_test else df
evaluation_X = X_test if use_test else X
evaluation_y = y_test if use_test else y

if st.sidebar.checkbox("Custom Params"):
    max_depth = st.sidebar.number_input("max_depth (see range 8..10)", 1)
else:
    max_depth = None

# = PROTECTED ATTRIBUTES =================================================

metric_mode = st.sidebar.selectbox("Disparity Mode", [DIFFERENCE, RATIO])
default_metrics = [
    accuracy_disparity,
    statistical_parity,
    equal_opportunity,
    false_positive_rate,
    equalized_odds,
]
all_metrics = [accuracy] + default_metrics
selected_metrics = st.sidebar.multiselect(
    "Metrics",
    all_metrics,
    default=default_metrics,
    format_func=lambda x: x.__name__,
)

protected_attributes = st.sidebar.selectbox(
    "Protected Attributes", [None] + list(feature_names)
)

if protected_attributes:
    "## Disparity Metrics"

    metrics = MetricHandler(*selected_metrics)

    selected_algoritms = st.sidebar.multiselect(
        "Algorithms",
        ["Only one", "DecisionTreeClassifier", "LogisticRegression", "SVC", "Ensemble"],
    )

    with st.beta_expander("Gold"):
        measure = metrics(
            data=evaluation_df,
            protected_attributes=protected_attributes,
            target_attribute=target_name,
            target_predictions=evaluation_df[target_name],
            positive_target=positive_target,
            mode=metric_mode,
            return_probs=True,
        )
        measure = metrics.to_df(measure)
        measure

    if "Only one" in selected_algoritms:
        with st.beta_expander("Only one"):
            with st.spinner("Training ..."):
                winner = st.sidebar.selectbox(
                    "Winner", df[protected_attributes].unique()
                )
                predictor = lambda x: (
                    positive_target
                    if x[protected_attributes] == winner
                    else negative_target
                )

                predicted = evaluation_df.apply(predictor, axis=1)
                measure = metrics(
                    data=evaluation_df,
                    protected_attributes=protected_attributes,
                    target_attribute=target_name,
                    target_predictions=predicted,
                    positive_target=positive_target,
                    mode=metric_mode,
                    return_probs=True,
                )
                measure = metrics.to_df(measure)
                measure

    if "DecisionTreeClassifier" in selected_algoritms:
        with st.beta_expander("DecisionTreeClassifier"):
            with st.spinner("Training ..."):
                classifier = DecisionTreeClassifier(max_depth=max_depth)
                classifier.fit(X, y)

                predicted = classifier.predict(evaluation_X)
                predicted = encoder.inverse_transform(predicted)
                predicted = pd.Series(predicted, evaluation_df.index)

                measure = metrics(
                    data=evaluation_df,
                    protected_attributes=protected_attributes,
                    target_attribute=target_name,
                    target_predictions=predicted,
                    positive_target=positive_target,
                    mode=metric_mode,
                    return_probs=True,
                )
                measure = metrics.to_df(measure)
                measure

    if "LogisticRegression" in selected_algoritms:
        with st.beta_expander("LogisticRegression"):
            with st.spinner("Training ..."):
                classifier = LogisticRegression()
                classifier.fit(X, y)

                predicted = classifier.predict(evaluation_X)
                predicted = encoder.inverse_transform(predicted)
                predicted = pd.Series(predicted, evaluation_df.index)

                measure = metrics(
                    data=evaluation_df,
                    protected_attributes=protected_attributes,
                    target_attribute=target_name,
                    target_predictions=predicted,
                    positive_target=positive_target,
                    mode=metric_mode,
                    return_probs=True,
                )
                measure = metrics.to_df(measure)
                measure

    if "SVC" in selected_algoritms:
        with st.beta_expander("SVC"):
            with st.spinner("Training ..."):
                classifier = SVC()
                classifier.fit(X, y)

                predicted = classifier.predict(evaluation_X)
                predicted = encoder.inverse_transform(predicted)
                predicted = pd.Series(predicted, evaluation_df.index)

                measure = metrics(
                    data=evaluation_df,
                    protected_attributes=protected_attributes,
                    target_attribute=target_name,
                    target_predictions=predicted,
                    positive_target=positive_target,
                    mode=metric_mode,
                    return_probs=True,
                )
                measure = metrics.to_df(measure)
                measure

    if "Ensemble" in selected_algoritms:
        with st.beta_expander("Ensemble"):
            with st.spinner("Training ..."):
                mitigator = SklearnMitigator(
                    dataset=dataset,
                    target=target_name,
                    metrics=metrics,
                    encoders=encoders,
                )
                ensemble = mitigator(
                    DecisionTreeClassifier(), LogisticRegression(), SVC()
                )

                predicted = ensemble.predict(evaluation_X)
                predicted = mitigator.encoders[target_name].inverse_transform(predicted)
                predicted = pd.Series(predicted, evaluation_df.index)

                measure = metrics(
                    data=evaluation_df,
                    protected_attributes=protected_attributes,
                    target_attribute=target_name,
                    target_predictions=predicted,
                    positive_target=positive_target,
                    mode=metric_mode,
                    return_probs=True,
                )
                measure = metrics.to_df(measure)
                measure
