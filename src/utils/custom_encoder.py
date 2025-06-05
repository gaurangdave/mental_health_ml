from sklearn.base import BaseEstimator, TransformerMixin, check_array, check_is_fitted
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import numpy as np
import pandas as pd


class CustomEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoding="onehot", categories="auto"):
        self.encoding = encoding
        self.categories = categories

    def fit(self, X, y=None):
        X_checked = check_array(X, dtype=["object", np.float64, np.int64])

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.to_list()
        else:
            self.feature_names_in_ = [f"ce_{i}" for i in range(X.shape[1])]

        if self.encoding == "ordinal":
            self.encoder_ = OrdinalEncoder(
                categories=self.categories, handle_unknown="use_encoded_value", unknown_value=-1)
            self.encoder_ = self.encoder_.fit(X_checked)
        else:
            self.encoder_ = OneHotEncoder(
                sparse_output=False, handle_unknown="ignore")
            self.encoder_ = self.encoder_.fit(X_checked)
        return self

    def transform(self, X, y=None):
        X_checked = check_array(X, dtype=["object", np.float64, np.int64
                                          ])
        check_is_fitted(self, ['encoder_'])
        transformed_data = self.encoder_.transform(X_checked)
        return transformed_data

    def get_feature_names_out(self, input_features=None):
        return self.encoder_.get_feature_names_out(input_features=input_features)
