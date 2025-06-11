# 🧠 Student Depression Risk Analysis

This project explores and models mental health trends among students using the Student Depression Dataset. The goal is to identify key lifestyle, academic, and personal factors contributing to student depression risk, and to build an interpretable machine learning model that can predict and explain those risks.

The project is part of a personal initiative to build meaningful, end-to-end machine learning applications through 2-week sprints.

## 🚀 Key Objectives
* Clean and preprocess the dataset for analysis and modeling.
* Explore correlations between student lifestyle factors and depression symptoms.
* Build a classifier to predict high-risk individuals.
* Use explainability tools (e.g., SHAP) to identify important features.
* Optionally, deploy an interactive web app to visualize predictions and insights.

## 🛠 Tech Stack
* Python (Pandas, scikit-learn, matplotlib/seaborn, SHAP)
* Jupyter Notebook for data exploration and modeling

## Notebooks
- [00_data_access.ipynb](./notebooks/00_data_access.ipynb) - Formats Geo Names data from `geonames.org` and saves it to CSV.
- [01_data_exploration.ipynb](./notebooks/01_data_exploration.ipynb) - Performs initial data exploration, cleaning, EDA, feature engineering, and train/test split.
- [02_data_preparation.ipynb](./notebooks/02_data_preparation.ipynb) - Preprocesses categorical and numerical data, creating scikit-learn pipelines for cleaning and transformation.
- [03_0_training_evaluation.ipynb](./notebooks/03_0_training_evaluation.ipynb) - Creates baseline models and explores preprocessed features for correlation.
- [03_1_logistic_regression.ipynb](./notebooks/03_1_logistic_regression.ipynb) - Trains and tunes a Logistic Regression model.
- [03_2_linear_svc.ipynb](./notebooks/03_2_linear_svc.ipynb) - Trains and tunes a Linear SVC model.
- [03_3_random_forest.ipynb](./notebooks/03_3_random_forest.ipynb) - Trains and tunes a Random Forest Classifier model.
- [03_4_svc.ipynb](./notebooks/03_4_svc.ipynb) - Trains and tunes an SVC model, focusing on non-linear kernels.
- [03_5_knn.ipynb](./notebooks/03_5_knn.ipynb) - Trains and tunes a K-Nearest Neighbors (KNN) model.
- [03_6_stacking_classifier.ipynb](./notebooks/03_6_stacking_classifier.ipynb) - Explores StackingClassifier and pivots to VotingClassifier for model ensembling.
- [04_test_prediction.ipynb](./notebooks/04_test_prediction.ipynb) - Evaluates trained models against the test dataset and visualizes performance.
- [05_huggingface_integration.ipynb](./notebooks/05_huggingface_integration.ipynb) - Integrates with Hugging Face Hub to upload/download datasets and models.
- [06_scratch_pad.ipynb](./notebooks/06_scratch_pad.ipynb) - A space for quick code experiments and API testing.

## 📁 Folder Structure

```
mental_health_ml/
│
├── .env
├── .git/
├── .gitignore
├── data/                 # Raw dataset and processed CSVs
├── models/               # Trained machine learning models
├── notebooks/            # Jupyter notebooks for EDA, modeling, etc.
├── requirements.txt      # Project dependencies
├── requirements_pc.txt   # Project dependencies for personal computer
├── src/                  # Source code for utility functions and modules
├── venv/                 # Virtual environment files
├── visualizations/       # Saved visualizations and plots
└── README.md             # This file
```


## 📊 Sample Insights (To Be Added)
* 💡 [Placeholder for an insight or visualization preview]
* 📉 [Placeholder for model performance summary]


## 📌 Future Enhancements
* Add SHAP-based explainability dashboard
* Explore clustering of students by behavior/risk groups
* Deploy model as a web app for interactive predictions
* Write a blog post or LinkedIn write-up with insights

