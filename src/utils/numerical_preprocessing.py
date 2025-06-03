import pandas as pd
import numpy as np
import matplotlib as plt

from pathlib import Path

import unicodedata
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.pipeline import Pipeline
from pandas.api.types import is_string_dtype
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import rbf_kernel

# Numerical Pipeline

default_num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

default_num_pipeline.set_output(transform="pandas")

# Age Pipeline
# helper function to map and create age bins


def map_age_bins_feature_names(function_transformer, feature_names_in):
    features_out = feature_names_in.tolist()
    features_out.extend(["age_range"])
    return features_out


def map_age_bins_fn(df):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("map_age_bins_fn : Input must be a pandas DataFrame")
    df_copy = df.copy()
    # default value
    df_copy["age_range"] = "gte_33"
    df_copy.loc[df_copy["age"].between(
        18, 23, "left"), "age_range"] = "18_to_23"
    df_copy.loc[df_copy["age"].between(
        23, 28, "left"), "age_range"] = "23_to_28"
    df_copy.loc[df_copy["age"].between(
        28, 33, "left"), "age_range"] = "28_to_33"
    return df_copy


map_age_bins = FunctionTransformer(
    map_age_bins_fn, feature_names_out=map_age_bins_feature_names)


def make_age_encoder_fn(encoding="onehot"):
    encoder = OneHotEncoder(
        sparse_output=False, handle_unknown="ignore")

    if encoding == "ordinal":
        encoder = OrdinalEncoder(categories=[[
            "18_to_23", "23_to_28", "28_to_33", "gte_33"
        ]], handle_unknown="use_encoded_value", unknown_value=-1)

    return ColumnTransformer(
        [("encode_age_range", encoder, ["age_range"])]
    )


age_pipeline = Pipeline(
    [("default_num_pipeline", default_num_pipeline),
     ("map_age_bins", map_age_bins),
     ("encode_age", make_age_encoder_fn(encoding="onehot"))]
)

# CGPA pipeline
# helper function to map and create cgpa bins


def map_cgpa_bins_feature_names(function_transformer, feature_names_in):
    features_out = feature_names_in.tolist()
    features_out.extend(["cgpa_range"])
    return features_out


def map_cgpa_bins_fn(df):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("map_cgpa_bins_fn : Input must be a pandas DataFrame")
    df_copy = df.copy()
    # default value
    df_copy["cgpa_range"] = "gte_7"
    df_copy.loc[df_copy["cgpa"].between(
        0, 4, "left"), "cgpa_range"] = "lt_4"
    df_copy.loc[df_copy["cgpa"].between(
        4, 7, "left"), "cgpa_range"] = "4_to_7"
    return df_copy


map_cgpa_bins = FunctionTransformer(
    map_cgpa_bins_fn, feature_names_out=map_cgpa_bins_feature_names)


def make_cgpa_encoder_fn(encoding="onehot"):
    encoder = OneHotEncoder(
        sparse_output=False, handle_unknown="ignore")

    if encoding == "ordinal":
        encoder = OrdinalEncoder(categories=[[
            "lt_4", "4_to_7", "gte_7"
        ]], handle_unknown="use_encoded_value", unknown_value=-1)

    return ColumnTransformer(
        [("encode_cgpa_range", encoder, ["cgpa_range"])]
    )


cgpa_pipeline = Pipeline(
    [("default_num_pipeline", default_num_pipeline),
     ("map_cgpa_bins", map_cgpa_bins),
     ("encode_cgpa", make_cgpa_encoder_fn(encoding="onehot"))]
)

# Study hours pipeline
# helper function to map and create hours bins


def map_hours_bins_feature_names(function_transformer, feature_names_in):
    features_out = feature_names_in.tolist()
    features_out.extend(["hours_range"])
    return features_out


def map_hours_bins_fn(df):
    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            "map_hours_bins_fn : Input must be a pandas DataFrame")
    df_copy = df.copy()
    # default value
    df_copy["hours_range"] = "gte_8"
    df_copy.loc[df_copy["work_study_hours"].between(
        0, 4, "left"), "hours_range"] = "lt_4"
    df_copy.loc[df_copy["work_study_hours"].between(
        4, 8, "left"), "hours_range"] = "4_to_8"
    return df_copy


map_hours_bins = FunctionTransformer(
    map_hours_bins_fn, feature_names_out=map_hours_bins_feature_names)


def make_hours_encoder_fn(encoding="onehot"):
    encoder = OneHotEncoder(
        sparse_output=False, handle_unknown="ignore")

    if encoding == "ordinal":
        encoder = OrdinalEncoder(categories=[[
            "lt_4", "4_to_8", "gte_8"
        ]], handle_unknown="use_encoded_value", unknown_value=-1)

    return ColumnTransformer(
        [("encode_hours_range", encoder, ["hours_range"])]
    )


hours_pipeline = Pipeline(
    [("default_num_pipeline", default_num_pipeline),
     ("map_hours_bins", map_hours_bins),
     ("encode_hours", make_hours_encoder_fn(encoding="onehot"))]
)

# ratings pipeline
# TODO: move the scaler to tunable parameter
rating_columns_pipeline = Pipeline([
    ("default_num_pipeline", default_num_pipeline),
    ("scale_rating_columns", StandardScaler())
])
rating_columns_pipeline.set_output(transform="pandas")

## work pressure and job satisfaction pipeline
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop)
    
    def get_feature_names_out(self, input_features=None):
        input_features = (
            input_features if input_features is not None
            else getattr(self, "feature_names_in_", None)
        )
        if input_features is None:
            raise ValueError("Input features not available for get_feature_names_out")

        return [col for col in input_features if col not in self.columns_to_drop]

# Use inside your top-level pipeline
drop_useless_cols = DropColumns(columns_to_drop=["work_pressure", "job_satisfaction"])

drop_column_pipeline = Pipeline([
    ("drop_useless", drop_useless_cols)
])