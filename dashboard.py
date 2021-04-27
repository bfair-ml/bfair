import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.datasets import load_adult
from src.methods import SklearnMitigator
from src.metrics import (
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
from src.utils import encode_features


# = DATASET =================================================
@st.cache
def load_dataset(name):
    if name == "adult":
        return load_adult()
    else:
        raise ValueError(f"Unknown dataset: {name}")


dataset_name = st.sidebar.selectbox("name", ["adult"])
dataset = load_dataset(dataset_name)
df: pd.DataFrame = dataset.data
df_test: pd.DataFrame = dataset.test

f"# Dataset: _{dataset_name}_"
if st.sidebar.checkbox("Show dataset"):
    df

# = FEATURES =================================================
feature_names = df.columns[:-1]
X = df[feature_names]

if st.sidebar.checkbox("Show Features"):
    "## Features"
    X
    for feature in feature_names:
        f"### {feature}"
        values = df[feature].unique()
        values.sort()
        values
        f"**Total:** {len(values)}"

# = TARGET =================================================
target_name = df.columns[-1]
y = df[target_name]

if st.sidebar.checkbox("Show Target"):
    "## Target"
    y

# = CLASSIFIER =================================================

X, y, encoders = encode_features(df, target=target_name, feature_names=feature_names)
encoder = encoders[target_name]

X_test, y_test, _ = encode_features(
    df_test, target=target_name, feature_names=feature_names, source_encoders=encoders
)

use_test = st.sidebar.checkbox("Test")
evaluation_df = df_test if use_test else df
evaluation_X = X_test if use_test else X
evaluation_y = y_test if use_test else y

if st.sidebar.checkbox("Custom Params"):
    max_depth = st.sidebar.number_input("max_depth (see range 8..10)", 1)
else:
    max_depth = None

# = PROTECTED ATTRIBUTES =================================================

protected_attributes = st.sidebar.selectbox(
    "Protected Attributes", [None] + list(feature_names)
)
metric_mode = st.sidebar.selectbox("Disparity Mode", [DIFFERENCE, RATIO])

if protected_attributes:
    "## Disparity Metrics"

    metrics = MetricHandler(
        accuracy,
        accuracy_disparity,
        statistical_parity,
        equal_opportunity,
        false_positive_rate,
        equalized_odds,
    )

    "### Gold"

    measure = metrics(
        data=evaluation_df,
        protected_attributes=protected_attributes,
        target_attribute=target_name,
        target_predictions=evaluation_df[target_name],
        positive_target=">50K",
        mode=metric_mode,
        return_probs=True,
    )
    measure = metrics.to_df(measure)
    measure

    "### Only one"
    winner = st.sidebar.selectbox("Winner", df[protected_attributes].unique())
    predictor = lambda x: (">50K" if x[protected_attributes] == winner else "<=50K")

    predicted = evaluation_df.apply(predictor, axis=1)
    measure = metrics(
        data=evaluation_df,
        protected_attributes=protected_attributes,
        target_attribute=target_name,
        target_predictions=predicted,
        positive_target=">50K",
        mode=metric_mode,
        return_probs=True,
    )
    measure = metrics.to_df(measure)
    measure

    if st.sidebar.checkbox("DecisionTreeClassifier"):
        "### DecisionTreeClassifier"
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
            positive_target=">50K",
            mode=metric_mode,
            return_probs=True,
        )
        measure = metrics.to_df(measure)
        measure

    if st.sidebar.checkbox("LogisticRegression"):
        "### LogisticRegression"
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
            positive_target=">50K",
            mode=metric_mode,
            return_probs=True,
        )
        measure = metrics.to_df(measure)
        measure

    if st.sidebar.checkbox("SVC"):
        "### SVC"
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
            positive_target=">50K",
            mode=metric_mode,
            return_probs=True,
        )
        measure = metrics.to_df(measure)
        measure

    if st.sidebar.checkbox("Ensemble"):
        "### Ensemble"
        mitigator = SklearnMitigator(
            dataset=dataset, target=target_name, metrics=metrics, encoders=encoders
        )
        ensemble = mitigator(DecisionTreeClassifier(), LogisticRegression(), SVC())

        predicted = ensemble.predict(evaluation_X)
        predicted = mitigator.encoders[target_name].inverse_transform(predicted)
        predicted = pd.Series(predicted, evaluation_df.index)

        measure = metrics(
            data=evaluation_df,
            protected_attributes=protected_attributes,
            target_attribute=target_name,
            target_predictions=predicted,
            positive_target=">50K",
            mode=metric_mode,
            return_probs=True,
        )
        measure = metrics.to_df(measure)
        measure

