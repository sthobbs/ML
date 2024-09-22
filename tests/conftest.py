import pytest
from experiment.experiment_xgb import XGBExperiment


# load data and train model
@pytest.fixture(scope="session")
def xgb_exp_train():
    config_path = "./tests/objects/configs/xgb.yaml"
    exp = XGBExperiment(config_path)
    exp.setup()
    exp.train()
    return exp


# 2nd experiment for additional tests (mainly involvding quantile bins)
@pytest.fixture(scope="session")
def xgb_exp_train_quantiles():
    config_path = "./tests/objects/configs/xgb_quantiles.yaml"
    exp = XGBExperiment(config_path)
    exp.setup()
    exp.train()
    return exp


# save model
@pytest.fixture(scope="session")
def xgb_exp_save_model(xgb_exp_train):
    xgb_exp_train.save_model()
    return xgb_exp_train


# generate performance evaluation
@pytest.fixture(scope="session")
def xgb_exp_evaluated(xgb_exp_train):
    xgb_exp_train.evaluate()
    return xgb_exp_train


# generate model explanitory artifacts
@pytest.fixture(scope="session")
def xgb_exp_explained(xgb_exp_train):
    xgb_exp_train.explain()
    return xgb_exp_train


# generate model explanitory artifacts using quantile bins
@pytest.fixture(scope="session")
def xgb_exp_explained_quantiles(xgb_exp_train_quantiles):
    xgb_exp_train_quantiles.explain()
    return xgb_exp_train_quantiles


# generate scores
@pytest.fixture(scope="session")
def xgb_exp_scores(xgb_exp_train):
    xgb_exp_train.gen_scores()
    return xgb_exp_train


# calibrate model with logistic regression
@pytest.fixture(scope="session")
def xgb_exp_calibration_logistic(xgb_exp_train):
    xgb_exp_train.calibrate(calibration_type='logistic', bin_type='uniform', n_bins=5)
    return xgb_exp_train


# calibrate model with isotonic regression
@pytest.fixture(scope="session")
def xgb_exp_calibration_isotonic(xgb_exp_train_quantiles):
    xgb_exp_train_quantiles.calibrate(calibration_type='isotonic', bin_type='quantile', n_bins=5)
    return xgb_exp_train_quantiles
