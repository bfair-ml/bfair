import pandas as pd
import streamlit as st

from collections import OrderedDict

from autogoal.search import ConsoleLogger
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from bfair.datasets import load_adult, load_german, load_review
from bfair.datasets.reviews import REVIEW_COLUMN, GENDER_COLUMN, GENDER_VALUES
from bfair.datasets.custom import load_from_file
from bfair.methods import AutoGoalDiversifier, SklearnMitigator, VotingClassifier
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
from bfair.utils import ClassifierWrapper, encode_features
from bfair.sensors import SensorHandler, CoreferenceNERSensor, DBPediaSensor, P_GENDER
from bfair.sensors.optimization import (
    load as load_from_config,
    get_embedding_based_sensor,
)


def tab(section):
    """
    #### Summary

    Creates a streamlit select box that groups all decorated functions that have the same key.

    #### Example:
    ```
    @tab('my tabs')
    def tab1(x):
        st.show("Im in TAB 1", x)

    @tab('my tabs')
    def tab2(x):
        st.show("Im in TAB 2", x)

    tab.run('my tabs', x=42)
    ```
    """

    import collections

    if not hasattr(tab, "__tabs__"):
        tab.__tabs__ = collections.defaultdict(dict)

        def run(sec, *args, **kwargs):
            func = st.sidebar.selectbox(sec, list(tab.__tabs__[sec]))
            func = tab.__tabs__[sec][func]
            func(*args, **kwargs)

        tab.run = run

    def wrapper(func):
        name = " ".join(s.title() for s in func.__name__.split("_"))
        tab.__tabs__[section][name] = func
        return func

    return wrapper


# = DATASET =================================================
@st.cache
def load_dataset(name):
    if name == "adult":
        return load_adult()
    if name == "german":
        return load_german()
    else:
        raise ValueError(f"Unknown dataset: {name}")


@tab("Tasks")
def mitigation():

    dataset_name = st.sidebar.selectbox("Dataset", ["custom", "adult", "german"])
    if dataset_name == "custom":
        uploaded_file = st.file_uploader("Dataset: Choose a file", type=["csv", "txt"])
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

    X, y, encoders = encode_features(
        df, target=target_name, feature_names=feature_names
    )
    encoder = encoders[target_name]

    X_test, y_test, _ = encode_features(
        df_test,
        target=target_name,
        feature_names=feature_names,
        source_encoders=encoders,
    )

    use_test = not df_test.empty and st.sidebar.checkbox("Test")
    evaluation_df = df_test if use_test else df
    evaluation_X = X_test if use_test else X
    evaluation_y = y_test if use_test else y

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

    if not protected_attributes:
        return

    "## Disparity Metrics"

    metrics = MetricHandler(*selected_metrics)

    selected_algoritms = st.sidebar.multiselect(
        "Algorithms",
        [
            "Custom",
            "Only one",
            "DecisionTreeClassifier",
            "LogisticRegression",
            "SVC",
            "Ensemble",
            "AutoGoalMitigator",
        ],
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

    if "Custom" in selected_algoritms:
        with st.beta_expander("Custom"):
            uploaded_file = st.file_uploader(
                "Prediction: Choose a file", type=["csv", "txt"]
            )
            if uploaded_file is not None:
                predicted = pd.read_csv(uploaded_file)[target_name]
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

    if "Only one" in selected_algoritms:
        with st.beta_expander("Only one"):
            with st.spinner("Training ..."):
                winner = st.selectbox("Winner", df[protected_attributes].unique())
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
            if st.checkbox("Limit depths"):
                max_depth = st.number_input("max_depth (see range 8..10)", 1)
            else:
                max_depth = None

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

    if "AutoGoalMitigator" in selected_algoritms:
        with st.beta_expander("AutoGoalMitigator"):

            col1, col2, col3 = st.beta_columns(3)

            n_classifiers = col1.number_input(
                "Number of classifiers",
                min_value=1,
                max_value=20,
                value=5,
            )
            search_iterations = col2.number_input(
                "Search iterations",
                min_value=1,
                value=2,
            )
            population_size = col3.number_input(
                "Population size",
                min_value=max(n_classifiers // search_iterations, 1),
                value=20,
            )

            logger = ConsoleLogger() if st.checkbox("Log to console") else None

            with st.spinner("Training ..."):
                from autogoal.kb import MatrixContinuousDense

                diversifier = AutoGoalDiversifier(
                    input=MatrixContinuousDense,
                    n_classifiers=n_classifiers,
                    pop_size=population_size,
                    search_iterations=search_iterations,
                )

                with st.spinner("Searching pipelines ..."):
                    pipelines, scores = diversifier(X, y, logger=logger)

                for p, s in zip(pipelines, scores):
                    "**Pipeline ...**"
                    p
                    f"... with score: {s}"

                classifiers = [ClassifierWrapper(p) for p in pipelines]

                voting = VotingClassifier(classifiers)

                predicted = voting.predict(evaluation_X)
                predicted = encoders[target_name].inverse_transform(predicted)
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


@st.cache(allow_output_mutation=True)
def load_sensors(language, config=None):
    if config is None:
        ensemble_based_handler_configuration = {
            "plain_mode": True,
            "embedding-sensor.remove-stopwords": False,
            "embedding-sensor.filter-filter": "NonNeutralWordsFilter",
            "embedding-sensor.filter-neutral-relative-threshold": 0.6390311558106935,
            "embedding-sensor.filter-neutral-norm-threshold": 0.1556742604492079,
            "embedding-sensor.aggretator-0": "UnionAggregator",
            "embedding-sensor.embedding-source": "word2vec-debiased",
        }
        embedding_sensor = load_from_config(
            list(ensemble_based_handler_configuration.items()),
            language=language,
            root=get_embedding_based_sensor,
        )
        sensors = [
            embedding_sensor,
            CoreferenceNERSensor.build(language=language, threshold=0),
            DBPediaSensor.build(language=language),
        ]
        return SensorHandler(sensors)
    else:
        generated = load_from_config(config, language=language)
        handler = generated.model
        return handler


@tab("Tasks")
def protected_attributes_extraction():
    dataset_name = st.sidebar.selectbox("Dataset", ["custom", "reviews"])

    if dataset_name == "reviews":
        dataset = load_review()
        X = dataset.data[REVIEW_COLUMN]
        y = dataset.data[GENDER_COLUMN]
    else:
        text = st.text_input("Text")
        if not text:
            st.stop()

        X = pd.Series([text], name=REVIEW_COLUMN)
        y = pd.Series([("CUSTOM SENTENCE",)], name=GENDER_COLUMN)

    language = st.sidebar.selectbox("Language", ["english"])

    handler_type = st.sidebar.selectbox(
        "Sensors",
        ["Not selected", "Default", "Custom"],
    )

    if handler_type == "Default":
        config = None
    elif handler_type == "Custom":
        config_str = st.sidebar.text_area("Custom Handler Configuration")
        try:
            config = eval(config_str)
        except Exception as e:
            st.error(f"Invalid handler configuration. {e}")
            st.stop()

        st.write(config)
        config = list(config.items())
    else:
        st.stop()

    handler = load_sensors(language, config)
    sensors = handler.sensors + [handler]

    for sensor in sensors:
        f"## Sensor: {type(sensor).__name__}"
        predictions = [sensor(review, GENDER_VALUES, P_GENDER) for review in X]
        results = pd.concat(
            (
                X,
                y.str.join(" & "),
                pd.Series(predictions, name="Predicted", index=X.index).str.join(" & "),
            ),
            axis=1,
        )
        st.write(results)


tab.run("Tasks")
