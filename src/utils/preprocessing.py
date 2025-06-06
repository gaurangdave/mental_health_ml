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
    # ("step_0_logger", ShapeLoggerTransformer(step_identifier=0), ["gender"]),
    ("preprocess_gender", categorical_preprocessing.gender_pipeline, [
        "gender"]),

    # ("step_1_logger", ShapeLoggerTransformer(step_identifier=1), ["profession"]),
    ("preprocess_profession",
     categorical_preprocessing.profession_pipeline, ["profession"]),

    # ("step_2_logger", ShapeLoggerTransformer(step_identifier=2), ["sleep_duration"]),
    ("sleep_duration_pipeline", categorical_preprocessing.sleep_duration_pipeline, [
        "sleep_duration"]),

    # ("step_3_logger", ShapeLoggerTransformer(step_identifier=3), ["dietary_habits"]),
    ("dietary_habits_pipeline",
     categorical_preprocessing.dietary_habits_pipeline, ["dietary_habits"]),

    # ("step_4_logger", ShapeLoggerTransformer(step_identifier=4), ["degree"]),
    ("degree_pipeline",
     categorical_preprocessing.degree_pipeline, ["degree"]),

    # ("step_5_logger", ShapeLoggerTransformer(step_identifier=5), ["suicidal_thoughts"]),
    (
        "suicidal_thoughts_pipeline", categorical_preprocessing.suicidal_thoughts_pipeline, [
            "suicidal_thoughts"]
    ),

    # ("step_6_logger", ShapeLoggerTransformer(step_identifier=6), ["family_history"]),
    (
        "family_history_pipeline", categorical_preprocessing.family_history_pipeline, [
            "family_history"]
    ),


    # ("step_7_logger", ShapeLoggerTransformer(step_identifier=7), ["city"]),
    ("city_pipeline", categorical_preprocessing.city_pipeline, ["city"]),

    # ("step_8_logger", ShapeLoggerTransformer(step_identifier=8), ["age"]),
    ("age_pipeline", numerical_preprocessing.age_pipeline, ["age"]),

    # ("step_9_logger", ShapeLoggerTransformer(step_identifier=9), ["cgpa"]),
    ("cgpa_pipeline", numerical_preprocessing.cgpa_pipeline, ["cgpa"]),

    # ("step_10_logger", ShapeLoggerTransformer(step_identifier=10), ["work_study_hours"]),
    ("hours_pipeline", numerical_preprocessing.hours_pipeline,
     ["work_study_hours"]),

    # ("step_11_logger", ShapeLoggerTransformer(step_identifier=11), [
    #     "academic_pressure", "study_satisfaction", "financial_stress"]),
    ("ratings_column_pipeline", numerical_preprocessing.rating_columns_pipeline, [
        "academic_pressure", "study_satisfaction", "financial_stress"]),

    # ("step_12_logger", ShapeLoggerTransformer(step_identifier=12), [
    #     "academic_pressure", "study_satisfaction", "financial_stress"]),
])
