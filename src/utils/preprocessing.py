from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

from src.utils import categorical_preprocessing, numerical_preprocessing


class ShapeLoggerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, step_identifier="default_id"):
        self.step_identifier = step_identifier
        pass

    def fit(self, X, y=None, sample_weight=None):
        shape_in_fit = X.shape
        print(
            f"Step : {self.step_identifier}, Method : fit, Shape : {shape_in_fit}")
        return self  # always return self!

    def transform(self, X):
        shape_in_transform = X.shape
        print(
            f"Step : {self.step_identifier}, Method : transform, Shape : {shape_in_transform}")
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features


pipeline = ColumnTransformer([
    ("preprocess_gender", categorical_preprocessing.gender_pipeline, [
        "gender"]),

    ("preprocess_profession",
     categorical_preprocessing.profession_pipeline, ["profession"]),

    ("sleep_duration_pipeline", categorical_preprocessing.sleep_duration_pipeline, [
        "sleep_duration"]),

    ("dietary_habits_pipeline",
     categorical_preprocessing.dietary_habits_pipeline, ["dietary_habits"]),

    ("degree_pipeline",
     categorical_preprocessing.degree_pipeline, ["degree"]),

    (
        "suicidal_thoughts_pipeline", categorical_preprocessing.suicidal_thoughts_pipeline, [
            "suicidal_thoughts"]
    ),

    (
        "family_history_pipeline", categorical_preprocessing.family_history_pipeline, [
            "family_history"]
    ),


    ("city_pipeline", categorical_preprocessing.city_pipeline, ["city"]),

    ("age_pipeline", numerical_preprocessing.age_pipeline, ["age"]),

    ("cgpa_pipeline", numerical_preprocessing.cgpa_pipeline, ["cgpa"]),

    ("hours_pipeline", numerical_preprocessing.hours_pipeline,
     ["work_study_hours"]),

    ("ratings_column_pipeline", numerical_preprocessing.rating_columns_pipeline, [
        "academic_pressure", "study_satisfaction", "financial_stress"]),
])
