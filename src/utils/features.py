from typing import List, Union

import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder


def encode_features(
    data: DataFrame,
    *,
    target: Union[int, str],
    feature_names: List[str] = None,
    source_encoders=None
):
    columns = data.columns
    target_name = columns[target] if isinstance(target, int) else target
    if feature_names is None:
        feature_names = columns[columns != target_name]

    # INIT ENCODERS
    if source_encoders is None:
        encoders = {}
    else:
        encoders = source_encoders

    # INIT DATA
    X = data[feature_names]
    y = data[target_name]

    # ENCODE X
    for feature in feature_names:
        if data.dtypes[feature] != object:
            continue
        if source_encoders is None:
            encoder = encoders[feature] = LabelEncoder()
            encoder.fit(X[feature])
        else:
            encoder = encoders[feature]
        X.loc[:, feature] = encoder.transform(X[feature])
    X = X.to_numpy()

    # ENCODE Y
    if source_encoders is None:
        encoder = encoders[target_name] = LabelEncoder()
        encoder.fit(y)
    else:
        encoder = encoders[target_name]
    y = encoder.transform(y)

    return X, y, encoders
