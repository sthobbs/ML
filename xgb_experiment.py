

import pandas as pd
import xgboost as xgb
import os
pwd = r"C:\Users\hobbs\Documents\Programming\ML"
os.chdir(pwd)
from experiment import Experiment, ConfigError
import logging
# import warnings
# import yaml
# warnings.filterwarnings('ignore')



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

    def explain(self):
        """
        Generate model explanitory charts including feature importance
        and shap values.
        """

        # Generate permutation feature importance and shapely value charts
        super().explain()

        
        # Get XGBoost feature importance
        self.logger.info(f"----- Generating XGBoost Feature Importances -----")
        imp_types = ['gain', 'total_gain', 'weight', 'cover', 'total_cover'] # importance types
        bstr = self.model.get_booster()
        imps = [pd.Series(bstr.get_score(importance_type=t), name=t) for t in imp_types]
        df = pd.concat(imps, axis=1) # dataframe of importances
        df = df.apply(lambda x: x / x.sum(), axis=0) # normalize so each column sums to 1
        df.sort_values('gain', ascending=False, inplace=True)
        self.explain_dir.mkdir(exist_ok=True)
        df.to_csv(f'{self.explain_dir}/xgb_feature_importance.csv')



if __name__ == "__main__":
    config_path = "C:/Users/hobbs/Documents/Programming/ML/config.yaml"
    exp = XGBExperiment(config_path)
    exp.run()


# config_path = "C:/Users/hobbs/Documents/Programming/ML/config.yaml"
# with open(config_path, "r") as f:
#     config = yaml.safe_load(f)



### Next Steps
# fix train by just calling super method
# put feature importance in subfolder (feature_importance/)
# relax required config variables
# (M) comments
    # Returns --------, Examples --------- >>> (?)
# (M) refactor explain code into seperate class
# read hyperopt papers
# read shap paper
# add README.md with examples
# change prints to logs and save logs to file?
# __repr__() and other dunder methods

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
# run linter over code
# unit tests (pytest)
# use isinstance() in more places?
# Curriculum learning sequences
# (M) Some sort of performance overlap analysis of two models



# more explainability artifacts? (do a quick google search)
    # LIME
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
    # encode variables (ordinal, one-hot, binary)
    # normalize (should normalize test/validation based on train parameters) 
    # synthetic data (SMOTE, other algorithms)
    # train/val/test split
# Code to prep data
    # e.g. train/val/test split
    # get list of features
    # imputation
    # downsample
# (M) Mock data generation

# probably not going to do:
    # combine metrics tables into one
    # add annotation of min(metric values) for metric vs n_estimators plots (one for each dataset?)
    # add performance vs epoch for other model types later



