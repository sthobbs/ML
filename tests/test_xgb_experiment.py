import pytest
from experiment.experiment_xgb import XGBExperiment
from sklearn.base import BaseEstimator
from utils import non_empty_file


# 1) check that the model is trained
def test_train_xgb(xgb_exp_train):
    assert xgb_exp_train.model.__sklearn_is_fitted__()


# 2) check that the config is copied
def test_config_copy(xgb_exp_train):
    path = xgb_exp_train.output_dir / "config.yaml"
    assert non_empty_file(path)


# 3) check that log file is being written to
def test_log_write(xgb_exp_train):
    path = xgb_exp_train.log_dir / "experiment.log"
    assert non_empty_file(path)


# 4) check that the model is saved to pickle and binary
@pytest.mark.parametrize("file_name", [
    "model.pkl",
    "model.bin"
])
def test_save_model(file_name, xgb_exp_save_model):
    path = xgb_exp_save_model.model_dir / file_name
    assert non_empty_file(path)


# 5) load model from path
def test_load_model_from_path():
    config_path = "./tests/objects/configs/xgb_load_model_from_path.yaml"
    exp = XGBExperiment(config_path)
    assert getattr(exp, "model", None) is None
    model_path = "./tests/objects/models/xgb_model.pkl"
    exp.load_model(path=model_path)
    assert isinstance(exp.model, BaseEstimator)


# 6) load model from path in config
def test_load_model_from_path_in_config():
    config_path = "./tests/objects/configs/xgb_load_model_from_path.yaml"
    exp = XGBExperiment(config_path)
    assert getattr(exp, "model", None) is None
    exp.load_model()
    assert isinstance(exp.model, BaseEstimator)


# 7) load model from sklearn object
def test_load_model_from_obj(xgb_exp_train):
    config_path = "./tests/objects/configs/xgb_load_model_from_obj.yaml"
    exp = XGBExperiment(config_path)
    assert getattr(exp, "model", None) is None
    exp.load_model(model_obj=xgb_exp_train.model)
    assert isinstance(exp.model, BaseEstimator)


# 8) check that scores tables are generated
@pytest.mark.parametrize("file_name", [
    "other_scores.csv",
    "test_scores.csv",
    "train_scores.csv",
    "validation_scores.csv",
])
def test_scores_gen(file_name, xgb_exp_scores):
    path = xgb_exp_scores.score_dir / file_name
    assert non_empty_file(path)


# ---------------------------
# Test Hyperparameter Tuning
# ---------------------------

# returns true if hyperparamter tuning was successful
def check_hp_tuning(config_path: str):
    exp = XGBExperiment(config_path)
    exp.setup()
    initial_params = exp.hyperparameters.copy()
    exp.train()
    tuned_params = {k: v for k, v in exp.model.get_params().items() if k in initial_params}
    print(initial_params, tuned_params)
    return initial_params != tuned_params


# 1) test atpe hyperparameter tuning (optimize auc)
def test_hptune_atpe():
    config_path = "./tests/objects/configs/hptune_atpe.yaml"
    assert check_hp_tuning(config_path)


# 2) test grid search hyperparameter tuning (with cross validation, optimize aucpr)
def test_hptune_gridsearch_cv():
    config_path = "./tests/objects/configs/hptune_gridsearch_cv.yaml"
    assert check_hp_tuning(config_path)


# 3) test grid search hyperparameter tuning (optimize brier_loss)
def test_hptune_gridsearch():
    config_path = "./tests/objects/configs/hptune_gridsearch.yaml"
    assert check_hp_tuning(config_path)


# 4) test random search hyperparameter tuning (optimize log_loss)
def test_hptune_randomsearch():
    config_path = "./tests/objects/configs/hptune_randomsearch.yaml"
    assert check_hp_tuning(config_path)


# 5) test tpe hyperparameter tuning (with cross validation, optimize aucpr)
def test_hptune_tpe():
    config_path = "./tests/objects/configs/hptune_tpe.yaml"
    assert check_hp_tuning(config_path)
