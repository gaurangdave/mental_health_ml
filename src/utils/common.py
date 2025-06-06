from pathlib import Path
import os
import pandas as pd
from joblib import dump

util_dir = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root
root_dir = Path(util_dir,"..","..")
models_dir = Path(root_dir, "models/")
data_dir = Path(root_dir, "data/")


def update_models_metrics(model_name, version, recall, precision, f1, file_name = "N/A"):
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
    pass


def save_model(model_name, version, estimator):
    # create models directory
    os.makedirs(models_dir, exist_ok=True)

    file_name = f"{"_".join(model_name.lower().split(" "))}_{version}.joblib"
    model_path = Path(models_dir, file_name)
    return dump(estimator, str(model_path)), file_name
