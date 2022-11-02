from utils import non_empty_file
import pytest


calibration_files = [
    "calibration_curves_other.png",
    "calibration_curves_test.png",
    "calibration_curves_train.png",
    "calibration_curves_validation.png",
    "compare_other.png",
    "compare_test.png",
    "compare_train.png",
    "compare_validation.png",
    "mapping_table_other.csv",
    "mapping_table_test.csv",
    "mapping_table_train.csv",
    "mapping_table_validation.csv",
    "score_mapping_other.png",
    "score_mapping_test.png",
    "score_mapping_train.png",
    "score_mapping_validation.png",
]

# 1) check that logistic model calibration plots and tables are generated
@pytest.mark.parametrize("file_name", calibration_files)
def test_calibration_logistic(file_name, xgb_exp_calibration_logistic):
    path = xgb_exp_calibration_logistic.calibration_dir / "comparison" / file_name
    assert non_empty_file(path)

# 2) check that isotonic model calibration plots and tables are generated
@pytest.mark.parametrize("file_name", calibration_files)
def test_calibration_isotonic(file_name, xgb_exp_calibration_isotonic):
    path = xgb_exp_calibration_isotonic.calibration_dir / "comparison" / file_name
    assert non_empty_file(path)

# 3) check that logistic model calibration model object is saved
def test_calibration_logistic(xgb_exp_calibration_logistic):
    path = xgb_exp_calibration_logistic.calibration_dir / "model/calibration_model.pkl"
    assert non_empty_file(path)

# 4) check that isotonic model calibration model object is saved
def test_calibration_logistic(xgb_exp_calibration_isotonic):
    path = xgb_exp_calibration_isotonic.calibration_dir / "model/calibration_model.pkl"
    assert non_empty_file(path)
