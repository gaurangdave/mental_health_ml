import json
from pathlib import Path
import os
import pandas as pd
import numpy as np
from joblib import dump

util_dir = os.path.dirname(os.path.abspath(
    __file__))  # This is your Project Root
root_dir = Path(util_dir, "..", "..")
models_dir = Path(root_dir, "models/")
data_dir = Path(root_dir, "data/")


def update_models_metrics(model_name, version, recall, precision, f1, file_name="N/A"):
    # create models directory
    os.makedirs(data_dir, exist_ok=True)
    # metrics file
    file_path = data_dir / "model_metrics.csv"

    # create new row for metrics
    new_row = {
        "model": model_name,
        "version": version,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "file": file_name
    }
    metrics_data = pd.DataFrame([new_row])

    # read existing data from file if it exists
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        # append to metrics data
        metrics_data = pd.concat([data, metrics_data], ignore_index=True)

    metrics_data.to_csv(file_path, index=False)
    return metrics_data

def update_model_params(model_name, version, params):
    # create models directory
    os.makedirs(data_dir, exist_ok=True)
    # metrics file
    file_path = data_dir / "model_params.json"

    # create new params object
    new_params = [
        {
            "name": model_name,
            "version": version,
            "params": params
        }
    ]

    # read existing params from file
    existing_params = []
    if os.path.exists(file_path):
        with open(file_path, "r") as fp:
            existing_params = json.load(fp)

    final_params = existing_params + new_params
    with open(file_path, "w") as fp:
        json.dump(final_params, fp)
    return final_params

def save_model(model_name, version, estimator):
    # create models directory
    os.makedirs(models_dir, exist_ok=True)

    file_name = f"{"_".join(model_name.lower().split(" "))}_{version}.joblib"
    model_path = Path(models_dir, file_name)
    return dump(estimator, str(model_path)), file_name

def calculate_mean_from_cv(cv_scores):
    mean_recall = np.mean(cv_scores["test_recall"])
    mean_precision = np.mean(cv_scores["test_recall"])
    mean_f1 = np.mean(cv_scores["test_f1"])
    print(f"Mean Recall: {mean_recall}, Mean Precision: {mean_precision},Mean F1: {mean_f1}")
    return mean_recall,mean_precision,mean_f1


def read_best_mean_grid_search_metrics(cv_results, cv_results_index):
    best_mean_recall = cv_results["mean_test_recall"][cv_results_index]
    best_mean_precision = cv_results["mean_test_precision"][cv_results_index]    
    best_mean_f1 = cv_results["mean_test_f1"][cv_results_index]
    print(f"Mean Recall: {best_mean_recall}, Mean Precision: {best_mean_precision},Mean F1: {best_mean_f1}")
    return best_mean_recall,best_mean_precision,best_mean_f1