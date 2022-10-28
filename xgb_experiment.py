
import pandas as pd
import xgboost as xgb
import os
pwd = r"C:\Users\hobbs\Documents\Programming\ML"
os.chdir(pwd)
from experiment import Experiment, ConfigError
import logging


class XGBExperiment(Experiment):
    """Class for training and evaluating XGBoost models"""

    def __init__(self, config_path):
        """
        Constructs attributes from a config file

        Parameters
        ----------
            config_path : str
                path to yaml config file
        """

        super().__init__(config_path)

    def validate_config(self):
        """Ensure that the config file is valid."""

        super().validate_config()
        valid_model_types = {'XGBClassifier', 'XGBRegressor'}
        if self.config["model_type"] not in valid_model_types:
            raise ConfigError(f"model_type must be in {valid_model_types}")
        if not self.config["supervised"]:
            raise ConfigError(f"supervised must be True")

    def load_model(self, model_obj=None, path=None):
        """
        Loads a model object from a parameter or file path, or instantiates a 
        new model object.

        Parameters
        ----------
            model_obj : str, optional
                scikit-learn model object with a .predict_proba() method
                (default is None)
            path : str, optional
                file path to scikit-learn model object with a .predict_proba()
                method (default is None)
        """

        super().load_model(model_obj, path)
        assert isinstance(self.model, xgb.XGBModel), "self.model must be an XGBoost model"

    def train(self, **kwargs):
        """
        tune hyperparameters, then train a final XGBoost model with
        the tuned hyperparmeters.
        """

        kwargs['verbose'] = self.verbose
        kwargs['eval_set'] = [(self.data[n]['X'], self.data[n]['y']) for n in self.dataset_names]
        super().train(**kwargs)

    def save_model(self):
        """Save the XGBoost model object as both .pkl and .bin files."""

        # save pickle version
        super().save_model()
        # save binary version
        self.model.save_model(self.model_dir/'model.bin')

    def evaluate(self, increment=0.01):
        """Evaluate XGboost model and generate performance charts."""

        # generate general metrics
        super().evaluate(increment)
        # generate XGBoost metrics
        self.model_eval.xgb_evaluate(self.dataset_names)






### Next Steps
# test explain code
# relax required config variables
# (M) comments
    # Returns --------, Examples --------- >>> (?)
# (M) refactor explain code into seperate class
# read hyperopt papers
# read shap paper
# Calibrate score (for binary classification)
    # plots
    # calibration model object
    # isotonic vs other types (as input parameter)
    # (M) search for optimal number of splits
    # calibrate based on validation? (maybe use test if there is no validation)
    # I think I'll make another folder for it


# GCP integration
    # move to and from gcs
    # read and write
    # gcp logging
    # BQ integration
# GCP storage class
    # move to and from GCS
    # ...
# GCP BQ class
    # query parallel
    # ...
# GCP logging class
    # ...
# unit tests (pytest)
# use isinstance() in more places?
# Curriculum learning sequences
# (M) Some sort of performance overlap analysis of two models


# have to test
    # load model (in different ways)
    # reproducability
    # other model types
# package up environment (in docker container?) for reproducability


# (M) xgboost feature importance plot
# (PN) add **kwargs if users want to pass stuff into my methods (into sklearn methods)



# EDA class
    # pandas profiler
    # null %, constant fields
    # feature values over time (need date or timestamp field)
        # might want a new dataset before implementing this
# Transform class
    # normalize (should normalize test/validation based on train parameters) 
    # synthetic data (SMOTE, other algorithms)
    # train/val/test split
# Code to prep data
    # e.g. train/val/test split
    # get list of features
    # imputation
    # encode variables (ordinal, one-hot, binary)
    # downsample
# (M) Mock data generation

# probably not going to do:
    # combine metrics tables into one
    # add annotation of min(metric values) for metric vs n_estimators plots (one for each dataset?)
    # add performance vs epoch for other model types later



