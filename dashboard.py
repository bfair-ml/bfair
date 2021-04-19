import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from src.datasets.adult import load_dataset as load_adult
from src.metrics import (
    MetricHandler,
    accuracy,
    accuracy_disparity,
    equal_opportunity,
    statistical_parity,
)


@st.cache
def load_dataset(name):
    if name == "adult":
        return load_adult()
    else:
        raise ValueError(f"Unknown dataset: {name}")


# = DATASET =================================================
dataset_name = st.sidebar.selectbox("name", ["adult"])
dataset = load_dataset(dataset_name)
df: pd.DataFrame = dataset.data

f"# Dataset: _{dataset_name}_"
if st.sidebar.checkbox("Show dataset"):
    df
    # df.dtypes


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
encoders = {}
for feature in X.columns:
    if X.dtypes[feature] != object:
        continue
    encoder = encoders[feature] = LabelEncoder()
    X.loc[:, feature] = encoder.fit_transform(X[feature])

encoder = encoders[target_name] = LabelEncoder()
y = encoder.fit_transform(y)

X = X.to_numpy()

classifier = DecisionTreeClassifier(
    max_depth=st.sidebar.number_input("max_depth (see range 8..10)", 1)
)
classifier.fit(X, y)

score = classifier.score(X, y)

"## `DecisionTreeClassifier` Score"
score

# = PROTECTED ATTRIBUTES =================================================

protected_attributes = st.sidebar.selectbox(
    "Protected Attributes", [None] + list(feature_names)
)

if protected_attributes:
    "## Disparity Metrics"

    metrics = MetricHandler(
        accuracy, accuracy_disparity, statistical_parity, equal_opportunity
    )

    "### Gold"

    measure = metrics(
        data=df,
        protected_attributes=protected_attributes,
        target_attribute=target_name,
        target_predictions=df[target_name],
        positive_target=">50K",
        return_probs=True,
    )
    measure

    "### Only one"
    winner = st.sidebar.selectbox("Winner", df[protected_attributes].unique())
    predictor = lambda x: (">50K" if x[protected_attributes] == winner else "<=50K")
    predicted = df.apply(predictor, axis=1)

    measure = metrics(
        data=df,
        protected_attributes=protected_attributes,
        target_attribute=target_name,
        target_predictions=predicted,
        positive_target=">50K",
        return_probs=True,
    )
    measure

    "### Classifier"
    predicted = classifier.predict(X)
    predicted = encoder.inverse_transform(predicted)
    predicted = pd.Series(predicted, df.index)

    measure = metrics(
        data=df,
        protected_attributes=protected_attributes,
        target_attribute=target_name,
        target_predictions=predicted,
        positive_target=">50K",
        return_probs=True,
    )
    measure

    "#### Score"
    score

