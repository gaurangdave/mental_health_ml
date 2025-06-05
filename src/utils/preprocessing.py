from sklearn.compose import ColumnTransformer

from src.utils import categorical_preprocessing, numerical_preprocessing


pipeline = ColumnTransformer([("preprocess_gender", categorical_preprocessing.gender_pipeline, ["gender"]),
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
        "academic_pressure", "study_satisfaction", "financial_stress"])
])
