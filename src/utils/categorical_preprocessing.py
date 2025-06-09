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

from src.utils.custom_encoder import CustomEncoder


data_dir = Path("..", "data")

# creating functional transformers

# fill na with Unknown


def fill_empty_strings_fn(df, columns=None):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    df_copy = df.copy()
    for col in df_copy.columns:
        # TODO : Confirm if comparing with type object is correct.
        if is_string_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].fillna("unknown")
    return df_copy

# remove spaces


def strip_spaces_fn(df, colmns=None):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    df_copy = df.copy()
    for col in df_copy.columns:
        # TODO : Confirm if comparing with type object is correct.
        if is_string_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].str.strip()
    return df_copy


def to_lower_case_fn(df):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    df_copy = df.copy()
    for col in df_copy.columns:
        # TODO : Confirm if comparing with type object is correct.
        if is_string_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].str.lower()
    return df_copy


def normalize_unicode_fn(df):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    df_copy = df.copy()
    for col in df_copy.columns:
        # TODO : Confirm if comparing with type object is correct.
        if is_string_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].map(lambda ct: unicodedata.normalize(
                "NFKD", ct).encode("ascii", "ignore").decode())
    return df_copy


fill_empty_strings = FunctionTransformer(
    fill_empty_strings_fn, feature_names_out="one-to-one")
strip_spaces = FunctionTransformer(
    strip_spaces_fn, feature_names_out="one-to-one")
to_lower_case = FunctionTransformer(
    to_lower_case_fn, feature_names_out="one-to-one")
normalize_unicode = FunctionTransformer(
    normalize_unicode_fn, feature_names_out="one-to-one")

# in our use case pipeline would make more sense as we need to use output of one transformer in another.
default_cat_pipeline = Pipeline([
    ("fill_empty_strings", fill_empty_strings),
    ("strip_spaces", strip_spaces),
    ("to_lower_case", to_lower_case),
    ("normalize_unicode", normalize_unicode)
])

# gender pipeline
gender_pipeline = Pipeline([
    ("default_cat_pipeline", default_cat_pipeline),
    ("encode_gender", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
])

# profession pipeline


def map_working_profession_fn(df):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    df_copy = df.copy()
    df_copy.loc[df_copy["profession"] != 'student'] = 'working'
    return df_copy


map_working_profession = FunctionTransformer(
    map_working_profession_fn, feature_names_out="one-to-one")

profession_pipeline = Pipeline([
    ("default_cat_pipeline", default_cat_pipeline),
    ("map_profession", map_working_profession),
    ("encode_profession", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
])

# sleep duration pipeline


def clean_sleep_duration_fn(df):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    df_copy = df.copy()
    df_copy["sleep_duration"] = df["sleep_duration"].str.strip("'")
    return df_copy


clean_sleep_duration = FunctionTransformer(
    clean_sleep_duration_fn, feature_names_out="one-to-one")


def map_sleep_duration_fn(df):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    df_copy = df.copy()
    # map 'less than 5 hours' to lt_5
    df_copy.loc[df_copy["sleep_duration"] ==
                'less than 5 hours', 'sleep_duration'] = 'lt_5'
    # map '5-6 hours' to bt_5_6
    df_copy.loc[df_copy["sleep_duration"] ==
                '5-6 hours', 'sleep_duration'] = 'bt_5_6'
    # map '7-8 hours' to bt_7_8
    df_copy.loc[df_copy["sleep_duration"] ==
                '7-8 hours', 'sleep_duration'] = 'bt_7_8'
    # more than 8 hours to gt_8
    df_copy.loc[df_copy["sleep_duration"] ==
                'more than 8 hours', 'sleep_duration'] = 'gt_8'
    # more than others to gt_8
    df_copy.loc[df_copy["sleep_duration"] ==
                'others', 'sleep_duration'] = 'gt_8'
    return df_copy


map_sleep_duration = FunctionTransformer(
    map_sleep_duration_fn, feature_names_out="one-to-one")


sleep_duration_pipeline = Pipeline([("default_cat_pipeline", default_cat_pipeline),
                                    ("sleep_duration_clean", clean_sleep_duration),
                                    ("sleep_duration_mapping", map_sleep_duration),
                                    ("sleep_duration_encoding", CustomEncoder(categories=[[
                                        "lt_5", "bt_5_6", "bt_7_8", "gt_8"
                                    ]]))
                                    ])
# dietary habits pipeline


def map_dietary_habits_fn(df):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    df_copy = df.copy()
    df_copy.loc[df_copy["dietary_habits"] ==
                "others", "dietary_habits"] = "unhealthy"
    return df_copy


map_dietary_habits = FunctionTransformer(
    map_dietary_habits_fn, feature_names_out="one-to-one")


dietary_habits_pipeline = Pipeline([("default_cat_pipeline", default_cat_pipeline),
                                    ("dietary_habits_mapping", map_dietary_habits),
                                    ("dietary_habits_encoding", CustomEncoder(categories=[[
                                        "unhealthy", "moderate", "healthy"
                                    ]]))
                                    ])
# degree pipeline
degree_mapping_dict = {
    "class 12":     {"field": "school",      "level": "high_school"},

    # Commerce & Business
    "b.com":        {"field": "commerce",    "level": "bachelor"},
    "m.com":        {"field": "commerce",    "level": "master"},
    "bba":          {"field": "business",    "level": "bachelor"},
    "mba":          {"field": "business",    "level": "master"},

    # Engineering & Tech
    "b.tech":       {"field": "engineering", "level": "bachelor"},
    "be":           {"field": "engineering", "level": "bachelor"},
    "b.arch":       {"field": "architecture", "level": "bachelor"},
    "me":           {"field": "engineering", "level": "master"},
    "m.tech":       {"field": "engineering", "level": "master"},

    # Science & CS
    "bsc":          {"field": "science",     "level": "bachelor"},
    "msc":          {"field": "science",     "level": "master"},
    "bca":          {"field": "computer_app", "level": "bachelor"},
    "mca":          {"field": "computer_app", "level": "master"},

    # Education
    "b.ed":         {"field": "education",   "level": "bachelor"},
    "m.ed":         {"field": "education",   "level": "master"},

    # Medical
    "mbbs":         {"field": "medical",     "level": "bachelor"},
    # Technically PG, but aligned here
    "md":           {"field": "medical",     "level": "master"},
    "b.pharm":      {"field": "pharmacy",    "level": "bachelor"},
    "m.pharm":      {"field": "pharmacy",    "level": "master"},

    # Law
    "llb":          {"field": "law",         "level": "bachelor"},
    "llm":          {"field": "law",         "level": "master"},

    # Hospitality
    "bhm":          {"field": "hospitality", "level": "bachelor"},
    "mhm":          {"field": "hospitality", "level": "master"},

    # Arts
    "ba":           {"field": "arts",        "level": "bachelor"},
    "ma":           {"field": "arts",        "level": "master"},

    # Research
    "phd":          {"field": "research",    "level": "doctorate"},

    # Other
    "others":       {"field": "unknown",     "level": "unknown"}
}
# helper function to clean up degree column


def clean_degree_fn(df):
    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            "degree_clean_fn : Input must be a pandas DataFrame")
    df_copy = df.copy()
    df_copy["degree"] = df_copy["degree"].str.strip("'")
    return df_copy


clean_degree = FunctionTransformer(
    clean_degree_fn, feature_names_out="one-to-one")


def map_degree_feature_names(function_transformer, feature_names_in):
    features_out = feature_names_in.tolist()
    features_out.extend(["degree_field", "degree_level"])
    return features_out

# helper funtion to map degree to degree_field and degree_level


def map_field(val):
    return degree_mapping_dict.get(val, {}).get("field", "unknown")


def map_level(val):
    return degree_mapping_dict.get(val, {}).get("level", "unknown")


def map_degree_fn(df):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("degree_mapping_fn: Input must be a pandas DataFrame")
    df_copy = df.copy()
    df_copy["degree_field"] = df_copy["degree"].map(map_field)
    df_copy["degree_level"] = df_copy["degree"].map(map_level)
    return df_copy


map_degree = FunctionTransformer(
    map_degree_fn, feature_names_out=map_degree_feature_names)

# helper function to create a column transformer to encode degree_field and degree_level fields.


# testing basic pipeline
degree_pipeline = Pipeline([
    ("default_cat_pipeline", default_cat_pipeline),
    ("clean", clean_degree),
    ("mapping", map_degree),
    ("degree_encoding", ColumnTransformer([
        ("degree_field_encoding", OneHotEncoder(
            handle_unknown="ignore", sparse_output=False), ["degree_field"]),
        ("degree_level_encoding", CustomEncoder(categories=[[
            "unknown", "high_school", "bachelor", "master", "doctorate"
        ]]), ["degree_level"])
    ]))
])

# suicidal thoughts pipeline
suicidal_thoughts_pipeline = Pipeline([
    ("default_cat_pipeline", default_cat_pipeline),
    ("suididal_thoughts_encoding", OneHotEncoder(
        handle_unknown="ignore", sparse_output=False))
])

# family history pipeline
family_history_pipeline = Pipeline([
    ("default_cat_pipeline", default_cat_pipeline),
    ("family_history_encoding", OneHotEncoder(
        handle_unknown="ignore", sparse_output=False))
])

# city pipeline
master_city_list = pd.read_csv(Path(data_dir, "detailed_in.csv"))

# helper function to clean up city column and remove special characters


def clean_city_fn(df):
    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            "degree_clean_fn : Input must be a pandas DataFrame")
    df_copy = df.copy()
    df_copy["city"] = df_copy["city"].str.strip("'")
    return df_copy


clean_city = FunctionTransformer(clean_city_fn, feature_names_out="one-to-one")

# helper function to rename old city names to new ones


def rename_city_fn(df):
    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            "degree_clean_fn : Input must be a pandas DataFrame")
    df_copy = df.copy()
    df_copy.loc[df_copy["city"] == "vasai-virar", "city"] = "virar"
    df_copy.loc[df_copy["city"] == "bangalore", "city"] = "bengaluru"
    return df_copy


rename_city = FunctionTransformer(
    rename_city_fn, feature_names_out="one-to-one")

# helper function that maps city names to default values of is_valid_city = 0, lat/long of Nagpur


def map_city_feature_names(function_transformer, feature_names_in):
    features_out = feature_names_in.tolist()
    features_out.extend(["is_valid_city", "lat", "long"])
    return features_out


def map_city_fn(df):
    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            "degree_clean_fn : Input must be a pandas DataFrame")
    df_copy = df.copy()
    df_copy["is_valid_city"] = 0
    df_copy["lat"] = 21.122615
    df_copy["long"] = 79.041124
    return df_copy


map_city = FunctionTransformer(
    map_city_fn, feature_names_out=map_city_feature_names)

# helper function that compares and verifies the city name and if its a valid city then updates the lat/long value


def map_city_data(city_name):
    # search for city name in master city lsit
    city_data = master_city_list.loc[master_city_list["name"] == city_name]
    # if city exists then return valid info
    if city_data.shape[0] > 0:
        return (1, city_data["lat"].values[0], city_data["long"].values[0])
    # if city doesn't exist the mark it as invalid and return default info
    return 0, 21.122615, 79.041124


def verify_city_fn(df):
    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            "degree_clean_fn : Input must be a pandas DataFrame")
    df_copy = df.copy()
    unique_cities = df_copy["city"].unique()
    for unique_city in unique_cities:
        is_valid_city, lat, long = map_city_data(unique_city)
        df_copy.loc[df_copy["city"] == unique_city,
                    "is_valid_city"] = is_valid_city
        df_copy.loc[df_copy["city"] == unique_city, "lat"] = lat
        df_copy.loc[df_copy["city"] == unique_city, "long"] = long
    return df_copy


verify_city = FunctionTransformer(
    verify_city_fn, feature_names_out="one-to-one")

# city depression ratio
"""
For this transformer we'll assume that we'll get clean city names
"""


class MapCityDepressionRatio(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.output_colums = ["city_depression_ratio"]
        pass

    """
        In the fit method, group by city names and create a city depression ratio map.
        see implementation below
    """

    def fit(self, X, y=None):
        y_df = y
        if not isinstance(y, pd.DataFrame):
            y_df = pd.DataFrame(y, columns=["depression"])
        # calculate the city depression ratios
        combined_data = pd.concat([X, y_df], axis=1)
        # only  calculate the depression ratios for valid cities.
        combined_aggregation = combined_data.loc[combined_data["is_valid_city"] == 1].groupby(["lat", "long", "city"], as_index=False).agg(
            # temp workaround to avoid using `city` column.
            total_instances=("city", "count"),
            # ideally we should drop this column in previous step of the pipeline.
            total_depression_count=("depression", "sum")
        )
        combined_aggregation["total_non_depression_count"] = combined_aggregation["total_instances"] - \
            combined_aggregation["total_depression_count"]
        combined_aggregation["depression_ratio"] = combined_aggregation["total_depression_count"] / \
            combined_aggregation["total_instances"]
        # set the mapping to use it during transformation
        self.city_depression_ratio_map = combined_aggregation.set_index("city")[
            "depression_ratio"]

        # save the median value to imputing
        # TODO: set it as parameter for future experimentation
        self.city_depression_ratio_median = combined_aggregation["depression_ratio"].median(
        )
        return self

    """
        create following fields in X_copy
        is_valid_city = default it to 0
        lat = default it to nagpur lat
        long = default it to nagpur long
        city_depression_ratio

        After the values are set, fill in the values with median for now.
    """

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy["city_depression_ratio"] = np.nan
        X_copy["city_depression_ratio"] = X_copy["city"].map(
            self.city_depression_ratio_map)
        # fill the city depression ration missing values with median
        X_copy["city_depression_ratio"] = X_copy["city_depression_ratio"].fillna(
            self.city_depression_ratio_median)
        # drop city column from X_copy
        # errors='ignore' is useful
        X_copy = X_copy.drop(columns=['city'], errors='ignore')
        return pd.DataFrame(X_copy)

    def get_feature_names_out(self, input_features=None):
        # find index of city from input feature list
        city_val_index = np.where(input_features == "city")[0][0]
        # drop the value at the index
        required_features = np.delete(input_features, city_val_index)
        # list(input_features).extend(self.output_colums)
        output_features = np.append(required_features, self.output_colums)
        return output_features


map_city_depression_ratio = MapCityDepressionRatio()

# cluster similarity


class ClusterSimilarityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=5, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
        self.similarity_columns = [
            f"similarity_to_cluster_{i}" for i in range(self.n_clusters)]

    def fit(self, X, y=None, sample_weight=None):
        X_copy = X.copy()

        self.n_features_in_ = X_copy.shape[1]
        if hasattr(X_copy, 'columns'):
            self.feature_names_in_ = np.array(X_copy.columns, dtype=object)
        else:
            self.feature_names_in_ = np.array(
                [f"input_feature_{i}" for i in range(self.n_features_in_)], dtype=object)

        # only find similarity based on lat and long
        X_lat_long = X_copy.loc[:, ["lat", "long"]]
        self.kmeans_ = KMeans(self.n_clusters, n_init=10,
                              random_state=self.random_state)
        self.kmeans_.fit(X_lat_long, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        check_is_fitted(self, ['kmeans_'])

        X_copy = X.copy()
        # only find similarity based on lat and long
        X_lat_long = X_copy.loc[:, ["lat", "long"]]
        similarity_values = rbf_kernel(
            X_lat_long, self.kmeans_.cluster_centers_, gamma=self.gamma)

        similarity_values_df = pd.DataFrame(
            similarity_values, columns=self.similarity_columns, index=X_copy.index)

        similarity_df = pd.concat(
            [X_copy, similarity_values_df], axis=1)
        return similarity_df

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ['kmeans_', 'feature_names_in_'])
        actual_input_column_names = []
        if input_features is not None:
            # If input_features are provided by the caller, use them
            actual_input_column_names = list(input_features)
        elif hasattr(self, 'feature_names_in_'):
            # Fallback to names captured during fit
            actual_input_column_names = list(self.feature_names_in_)

        # list(input_features).extend(self.output_colums)
        output_features = np.append(
            actual_input_column_names, self.similarity_columns)
        return output_features


cluster_similarity = ClusterSimilarityTransformer()


class ConditionalDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None, active=True):
        self.columns_to_drop = columns_to_drop  # columns to drop
        self.active = active  # flag to make this dropper active/inactive

    def fit(self, X, y=None):
        X_copy = X.copy()
        self.cols_to_drop_present_ = [
            col for col in self.columns_to_drop if col in X_copy.columns]
        return self

    def transform(self, X):
        check_is_fitted(self, ["cols_to_drop_present_"])
        X_copy = X.copy()
        if self.active and self.columns_to_drop:
            # Ensure columns exist before trying to drop
            # cols_to_drop_present = [
            #     col for col in self.columns_to_drop if col in X_copy.columns]
            if self.cols_to_drop_present_:
                return X_copy.drop(columns=self.cols_to_drop_present_)
        return X_copy

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ["cols_to_drop_present_"])
        if self.active and self.cols_to_drop_present_:
            return [f for f in input_features if f not in self.cols_to_drop_present_]
        return input_features


# by default this is active
dropper_pipeline = ConditionalDropper(columns_to_drop=["city", "lat", "long"])

city_pipeline = Pipeline([
    ("default_cat_pipeline", default_cat_pipeline),
    ("city_clean", clean_city),
    ("rename_city", rename_city),
    ("city_mapping", map_city),
    ("city_verification", verify_city),
    ("city_depression_ratio", map_city_depression_ratio),
    ("city_cluster_similarity", cluster_similarity),
    ("drop_lat_long", dropper_pipeline)
])
