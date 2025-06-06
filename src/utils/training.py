import pandas as pd

# defining custom refit logic
def refit_strategy(cv_results):
    cv_results_df = pd.DataFrame(cv_results)

    # 98% of max score
    excellence_zone = 0.98

    # step 1: find highest mean test score
    highest_mean_recall = cv_results_df["mean_test_recall"].max()

    # step 2: zone of excellence
    recall_excellence_zone = cv_results_df[cv_results_df["mean_test_recall"] >= (
        excellence_zone * highest_mean_recall)]

    # select highest F1 score from excellence zone
    best_param_index = recall_excellence_zone["mean_test_f1"].idxmax()

    return best_param_index