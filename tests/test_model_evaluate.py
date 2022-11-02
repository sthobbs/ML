from utils import non_empty_file
import pytest


# 1) check that performance evaluation plots are generated
@pytest.mark.parametrize("file_name", [
    "det_other.png",
    "det_test.png",
    "det_train.png",
    "det_validation.png",
    "n_estimators_vs_auc.png",
    "n_estimators_vs_aucpr.png",
    "n_estimators_vs_error.png",
    "n_estimators_vs_logloss.png",
    "precision_recall_vs_threshold_other.png",
    "precision_recall_vs_threshold_test.png",
    "precision_recall_vs_threshold_train.png",
    "precision_recall_vs_threshold_validation.png",
    "precision_vs_recall_other.png",
    "precision_vs_recall_test.png",
    "precision_vs_recall_train.png",
    "precision_vs_recall_validation.png",
    "roc_other.png",
    "roc_test.png",
    "roc_train.png",
    "roc_validation.png",
    "score_histogram_other.png",
    "score_histogram_test.png",
    "score_histogram_train.png",
    "score_histogram_validation.png",
])
def test_evaluation_plots(file_name, xgb_exp_evaluated):
    path = xgb_exp_evaluated.performance_dir / "plots" / file_name
    assert non_empty_file(path)

# 2) check that performance evaluation tables are generated
@pytest.mark.parametrize("file_name", [
    "ks_statistics.csv",
    "metrics_other.csv",
    "metrics_test.csv",
    "metrics_train.csv",
    "metrics_validation.csv",
    "n_estimators_vs_metrics_other.csv",
    "n_estimators_vs_metrics_test.csv",
    "n_estimators_vs_metrics_train.csv",
    "n_estimators_vs_metrics_validation.csv",
    "optimal_n_estimators.csv",
    "threshold_vs_metrics_other.csv",
    "threshold_vs_metrics_test.csv",
    "threshold_vs_metrics_train.csv",
    "threshold_vs_metrics_validation.csv",
])
def test_evaluation_tables(file_name, xgb_exp_evaluated):
    path = xgb_exp_evaluated.performance_dir / "tables" / file_name
    assert non_empty_file(path)
