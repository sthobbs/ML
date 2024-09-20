import xgboost as xgb
from sklearn import ensemble, tree, neural_network, neighbors, linear_model, cluster
from sklearn.base import BaseEstimator
from sklearn.utils import shuffle
from sklearn.model_selection import ParameterGrid, cross_val_score, GridSearchCV, PredefinedSplit
from sklearn.metrics import make_scorer
from pathlib import Path
from datetime import datetime
from shutil import copyfile
from hyperopt import fmin, rand, tpe, atpe, hp, STATUS_OK, Trials, pyll
from dask.diagnostics import ProgressBar
import dask.dataframe as dd
import pandas as pd
import numpy as np
import yaml
import pickle
import time
import random
import logging
from typing import Union, Optional, Dict, Tuple
from experiment.model_evaluate import ModelEvaluate, metric_score
from experiment.model_explain import ModelExplain
from experiment.model_calibrate import ModelCalibrate
import warnings
ProgressBar().register()

# workaround for hyperopt/numpy incompatability
if not hasattr(np, "warnings"):
    setattr(np, "warnings", warnings)


class ConfigError(Exception):
    """Exception for issues with a configuration file."""

    pass


class Experiment():
    """
    Class for running ML model experiments

    Author:
       Steve Hobbs
       github.com/sthobbs
    """

    def __init__(self, config_path: str) -> None:
        """
        Constructs attributes from a config file

        Parameters
        ----------
            config_path : str
                path to yaml config file
        """

        print(f"----- Initializing {self.__class__.__name__} -----")

        # Load and validate config file
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            raise ConfigError(f"error loading config file: {e}")
        self.validate_config()

        # Set variables
        self.config_path = Path(config_path)

        # ------ Meta Config -------
        self.version = self.config["version"]
        self.description = self.config["description"]

        # ------ Input Config -------
        self.data_dir = Path(self.config["data_dir"])
        self.data_file_patterns = self.config["data_file_patterns"]
        self.input_model_path = self.config.get("input_model_path")

        # ------ Output Config -------
        self.experiment_dir = Path(self.config["experiment_dir"])
        now = datetime.now().strftime("%Y%m%d-%H%M%S")  # current datetime
        self.output_dir = self.experiment_dir / f"{self.version}-{now}"
        self.performance_dir = self.output_dir / self.config["performance_dir"]
        self.model_dir = self.output_dir / self.config["model_dir"]
        self.explain_dir = self.output_dir / self.config["explain_dir"]
        self.save_scores = self.config["save_scores"]
        if self.save_scores:
            self.score_dir = self.output_dir / self.config["score_dir"]
        self.log_dir = self.output_dir / self.config.get("log_dir", "logs")
        self.calibration_dir = self.output_dir / self.config.get("calibration_dir", "calibration")
        self.misc_dir = self.output_dir / self.config.get("misc_dir", "misc")
        self.performance_increment = float(self.config.get("performance_increment", 0.01))

        # ------ Job Config -------
        self.model_type = self.config["model_type"]
        self.supervised = self.config["supervised"]
        self.binary_classification = self.config["binary_classification"]
        self.label = None
        if self.supervised:
            self.label = self.config["label"]
        self.features = self.config["features"]
        self.aux_fields = self.config.get("aux_fields")
        if self.aux_fields is None:
            self.aux_fields = []
        elif isinstance(self.aux_fields, str):
            self.aux_fields = [self.aux_fields]
        assert isinstance(self.aux_fields, list), \
            f"self.aux_fields must be a list or string, not {type(self.aux_fields)}"
        self.seed = int(self.config["seed"])
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.verbose = int(self.config.get("verbose", 10))

        # ------ Hyperparameters -------
        self.hyperparameters = self.config["hyperparameters"]
        self.hyperparameters["random_state"] = self.seed
        self.hyperparameter_tuning = self.config.get("hyperparameter_tuning", False)
        self.hyperparameter_eval_metric = self.config.get("hyperparameter_eval_metric", "log_loss")
        self.cross_validation = self.config.get("cross_validation", False)
        self.cv_folds = int(self.config.get("cv_folds", 5))
        self.tuning_algorithm = self.config.get("tuning_algorithm")
        self.grid_search_n_jobs = int(self.config.get("grid_search_n_jobs", 1))
        self.tuning_iterations = self.config.get("tuning_iterations")
        if self.tuning_iterations is not None:
            self.tuning_iterations = int(self.tuning_iterations)
        self.tuning_parameters = self.config.get("tuning_parameters")

        # ------ Model Explainability -------
        # Permutation Importance
        self.permutation_importance = self.config.get("permutation_importance", False)
        self.perm_imp_metrics = self.config.get("perm_imp_metrics", "neg_log_loss")
        if isinstance(self.perm_imp_metrics, str):
            self.perm_imp_metrics = [self.perm_imp_metrics]
        self.perm_imp_n_repeats = int(self.config.get("perm_imp_n_repeats", 10))
        # Shapely Values
        self.shap = self.config.get("shap", False)
        self.shap_sample = self.config.get("shap_sample")
        if self.shap_sample is not None:
            self.shap_sample = int(self.shap_sample)
        # Feature Distribution
        self.feature_distribution = self.config.get("feature_distribution", False)
        self.exclude_outliers = self.config.get("exclude_outliers", False)
        # Feature vs Time
        self.feature_vs_time = self.config.get("feature_vs_time", False)
        self.dt_field = self.config.get("dt_field", "timestamp")
        self.dt_format = self.config.get("dt_format", "mixed")
        self.feature_vs_time_n_bins = int(self.config.get("feature_vs_time_n_bins", 15))
        # Population Stability Index
        self.psi = self.config.get("psi", False)
        self.psi_bin_type = self.config.get("psi_bin_type", "fixed")
        self.psi_n_bins = int(self.config.get("psi_n_bins", 10))
        # Characteristic Stability Index
        self.csi = self.config.get("csi", False)
        self.csi_bin_type = self.config.get("csi_bin_type", "fixed")
        self.csi_n_bins = int(self.config.get("csi_n_bins", 10))
        # Variance Inflation Factor
        self.vif = self.config.get("vif", False)
        # WOE/IV
        self.woe_iv = self.config.get("woe_iv", False)
        self.woe_bin_type = self.config.get("woe_bin_type", "quantiles")
        self.woe_n_bins = int(self.config.get("woe_n_bins", 10))
        # Correlation
        self.correlation = self.config.get("correlation", False)
        self.corr_max_features = int(self.config.get("corr_max_features", 100))
        # Summary Statistics
        self.summary_stats = self.config.get("summary_stats", False)
        self.quantiles = self.config.get("quantiles")
        self.top_n_value_counts = int(self.config.get("top_n_value_counts", 5))
        # Binary Splits
        self.binary_splits = self.config.get("binary_splits", False)
        self.n_splits = int(self.config.get("n_splits", 10))

        # ------ Model Calibration -------
        self.model_calibration = self.config.get("model_calibration", False)
        self.calibration_type = self.config.get("calibration_type", "logistic")
        self.calibration_train_dataset_name = self.config.get("calibration_train_dataset_name", "validation")

        # ------ Other -------
        self.data: Dict[str, Dict[str, Union[pd.core.frame.DataFrame, pd.core.series.Series]]] = {}  # where data will be stored
        self.aux_data: Dict[str, Union[pd.core.frame.DataFrame, pd.core.series.Series]] = {}  # where auxiliary fields will be stored
        self.dataset_names = list(self.data_file_patterns)  # names of datasets

        # ------ Logging -------
        # make output dirs
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

        # create logger
        self.logger = logging.getLogger(__name__).getChild(self.__class__.__name__).getChild(str(id(self)))
        self.logger.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # create and add handlers for console output
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # create and add handlers for file output
        fh = logging.FileHandler(self.log_dir/"experiment.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def validate_config(self) -> None:
        """Ensure that the config file is valid."""

        # specify which keys and values are required/valid
        required_keys = {
            'version', 'description', 'data_dir', 'data_file_patterns',
            'experiment_dir', 'performance_dir',
            'model_dir', 'explain_dir', 'save_scores', 'model_type',
            'supervised', 'binary_classification', 'features',
            'seed', 'hyperparameters',
        }
        other_valid_keys = {
            'input_model_path', 'score_dir', 'log_dir', 'calibration_dir',
            'misc_dir', 'performance_increment', 'label', 'aux_fields',
            'verbose', 'hyperparameter_tuning',  'hyperparameter_eval_metric',
            'cross_validation', 'cv_folds', 'tuning_algorithm',
            'grid_search_n_jobs',  'tuning_iterations', 'tuning_parameters',
            'permutation_importance', 'perm_imp_metrics', 'perm_imp_n_repeats',
            'shap', 'shap_sample', 'feature_distribution', 'exclude_outliers',
            'feature_vs_time', 'dt_field', 'dt_format', 'feature_vs_time_n_bins',
            'psi', 'psi_bin_type', 'psi_n_bins', 'csi', 'csi_bin_type',
            'csi_n_bins', 'vif', 'woe_iv', 'woe_bin_type', 'woe_n_bins',
            'correlation', 'corr_max_features', 'summary_stats', 'quantiles',
            'top_n_value_counts', 'binary_splits', 'n_splits', 'model_calibration',
            'calibration_type', 'calibration_train_dataset_name'
        }
        valid_keys = required_keys.union(other_valid_keys)
        keys_with_required_vals = {
            'version', 'description', 'data_dir', 'data_file_patterns',
            'experiment_dir', 'performance_dir', 'model_dir', 'explain_dir',
            'save_scores', 'model_type', 'supervised', 'features',
            'seed'
        }
        keys = set(self.config.keys())
        keys_with_vals = {k for k, v in self.config.items() if v is not None}

        # check for missing required keys
        missing_required_keys = required_keys.difference(keys)
        if missing_required_keys:
            msg = f"missing key(s) in config file: {', '.join(missing_required_keys)}"
            raise ConfigError(msg)

        # check for non-valid keys
        unexpected_keys = keys.difference(valid_keys)
        if unexpected_keys:
            msg = f"unexpected key(s) in config file: {', '.join(unexpected_keys)}"
            raise ConfigError(msg)

        # check for keys with missing required values
        keys_missing_required_vals = keys_with_required_vals.difference(keys_with_vals)
        if keys_missing_required_vals:
            msg = f"missing value(s) in config file for: {', '.join(keys_missing_required_vals)}"
            raise ConfigError(msg)

        # check for existence of train
        keys_with_vals = {k for k, v in self.config["data_file_patterns"].items() if v}
        if 'train' not in keys_with_vals:
            raise ConfigError("missing train key or value within data_file_patterns")
        # need test or validation if HP-tuning without CV
        if self.config.get("hyperparameter_tuning", False) \
                and not self.config.get("cross_validation", False) \
                and not ('validation' in keys_with_vals or 'test' in keys_with_vals):
            msg = "either 'validation' or 'test' must have a path in data_file_patterns"
            raise ConfigError(msg)

        # check for score_dir value if we want to save model scores
        if self.config["save_scores"]:
            if not self.config.get("score_dir"):
                raise ConfigError("missing score_dir key or value")

        # check performance_increment (either no key, None or castable to float between 0 and 1)
        performance_increment = self.config.get("performance_increment")
        if performance_increment is not None:
            try:
                performance_increment = float(performance_increment)
            except Exception as e:
                raise ConfigError(f"shap_sample exception converting to int: {e}")
            if not 0 < performance_increment < 1:
                raise ConfigError("performance_increment must be between 0 and 1")

        # check that features is a list of length >= 1
        if type(self.config["features"]) is not list or len(self.config["features"]) < 1:
            raise ConfigError("features must be a list with len >= 1")

        # specify valid supervised and unsupervised models
        supervised_models = {
          'XGBClassifier', 'XGBRegressor',
          'RandomForestClassifier', 'RandomForestRegressor',
          'DecisionTreeClassifier', 'DecisionTreeRegressor',
          'MLPClassifier', 'MLPRegressor',
          'KNeighborsClassifier', 'KNeighborsRegressor',
          'LogisticRegression', 'LinearRegression'
        }
        unsupervised_models = {'KMeans', 'DBSCAN', 'IsolationForest'}
        valid_model_types = supervised_models | unsupervised_models | {'Other'}

        # check that model_type is valid
        model_type = self.config["model_type"]
        if self.config["model_type"] not in valid_model_types:
            raise ConfigError(f"invalid model_type: {model_type}")

        # check supervised is consistent with model_type
        if self.config["model_type"] in supervised_models and not self.config["supervised"]:
            raise ConfigError(f"supervised should be True when model_type = {model_type}")
        if self.config["model_type"] in unsupervised_models and self.config["supervised"]:
            raise ConfigError(f"supervised should be False when model_type = {model_type}")

        # check label is consistent with supervised
        if self.config["supervised"] and not self.config.get("label"):
            raise ConfigError("need label when supervised = True")

        # check eval_metric
        if "eval_metric" in self.config.get("hyperparameters", []):
            if not isinstance(self.config["hyperparameters"]["eval_metric"], list):
                if not isinstance(self.config["hyperparameters"]["eval_metric"], str):
                    raise ConfigError("eval_metric should be a string or list")
                eval_metric = [self.config["hyperparameters"]["eval_metric"]]
            else:
                eval_metric = self.config["hyperparameters"]["eval_metric"]
            valid_metrics = {
                'rmse', 'rmsle', 'mae', 'mape', 'mphe', 'logloss', 'error', 'merror',
                'mlogloss', 'poisson-nloglik', 'gamma-nloglik', 'cox-nloglik',
                'gamma-deviance', 'tweedie-nloglik', 'aft-nloglik', 'auc', 'aucpr'
            }
            for m in eval_metric:
                if m not in valid_metrics:
                    raise ConfigError(f"{m} is not a valid eval_metric")

        # check that hyperparamter tuning algorithm is valid
        if self.config.get("hyperparameter_tuning", False):
            if self.config.get("tuning_algorithm") not in {"grid", "random", "tpe", "atpe"}:
                msg = f'tuning_algorithm value must be in {"grid", "random", "tpe", "atpe"}'
                raise ConfigError(msg)
            # check grid_search_n_jobs
            if self.config["tuning_algorithm"] == "grid":
                feature = self.config.get("grid_search_n_jobs", 1)
                try:
                    feature = int(feature)
                except Exception as e:
                    raise ConfigError(f"{feature} exception converting to int: {e}")
                if not (feature == -1 or feature >= 1):
                    raise ConfigError("invalid grid_search_n_jobs value")
            # check tuning_iterations
            if self.config["tuning_algorithm"] in {"random", "tpe", "atpe"}:
                if not self.config.get("tuning_iterations"):
                    msg = "must specify tuning_iterations for the chosen tuning_algorithm"
                    raise ConfigError(msg)
            valid_metrics = {'average_precision', 'aucpr', 'auc', 'log_loss', 'brier_loss'}
            if self.config.get("hyperparameter_eval_metric", "log_loss") not in valid_metrics:
                raise ConfigError("invalid hyperparameter_eval_metric value")

        # check that tuning_parameters is valid
        if self.config.get("hyperparameter_tuning", False):
            # check tuning_parameters has a value
            if not self.config["tuning_parameters"]:
                msg = "when hyperparameter_tuning is True, tuning_parameters must be specified"
                raise ConfigError(msg)
            # check tuning_parameters is a dictionary
            if not isinstance(self.config["tuning_parameters"], dict):
                msg = "when hyperparameter_tuning is True, tuning_parameters must be a dictionary"
                raise ConfigError(msg)
            # for grid search, check that tuning_parameters specifies lists of possible values
            if self.config["tuning_algorithm"] == "grid":
                for k, v in self.config["tuning_parameters"].items():
                    if not isinstance(v, list):
                        raise ConfigError(f"The tuning_parameters value of {k} must be a list")
            # for hyperopt search, check that tuning_parameters specifies valid values
            if self.config["tuning_algorithm"] in {"random", "tpe", "atpe"}:
                for hyperparameter, distribution in self.config["tuning_parameters"].items():
                    # check that both the function and params are specified
                    if set(distribution.keys()) != {'function', 'params'}:
                        msg = (f"tuning_parameters.{hyperparameter} must contain"
                               " 'function' and 'params' keys, and no others")
                        raise ConfigError(msg)
                    func = distribution['function']
                    # check that all and only all valid params are included
                    func_to_params = {
                        "choice": {'options'},
                        "uniform": {'low', 'high'},
                        "quniform": {'low', 'high', 'q'},
                        "normal": {'mu', 'sigma'}
                    }
                    if set(distribution['params'].keys()) != func_to_params[func]:
                        msg = (f"tuning_parameters.{hyperparameter}.params must contain"
                               f" the following keys and no others: {func_to_params[func]}.")
                        raise ConfigError(msg)
                    # check that all params have valid values
                    param_to_datatype: Dict[str, Union[type, Tuple[type, type]]] = {
                        "options": list,
                        "low": (int, float),
                        "high": (int, float),
                        "q": (int, float),
                        "mu": (int, float),
                        "sigma": (int, float)
                    }
                    for param, value in distribution['params'].items():
                        valid_type = param_to_datatype[param]
                        if not isinstance(value, valid_type):
                            msg = (f"tuning_parameters.{hyperparameter}.params.{param} must"
                                   f" be of type: {valid_type}")
                            raise ConfigError(msg)
                        if param == "sigma" and value <= 0:
                            msg = f"tuning_parameters.{hyperparameter}.params.{param} must be > 0"
                            raise ConfigError(msg)

        # Note: Not checking perm_imp_metrics since there are many possible values that work

        # check perm_imp_n_repeats
        try:
            int(self.config.get("perm_imp_n_repeats", 10))
        except Exception as e:
            raise ConfigError(f"perm_imp_n_repeats exception converting to int: {e}")

        # check shap_sample (either no key, None or castable to int)
        shap_sample = self.config.get("shap_sample")
        if shap_sample is not None:
            try:
                int(shap_sample)
            except Exception as e:
                raise ConfigError(f"shap_sample exception converting to int: {e}")

        # check dt_field is is features or aux_fields
        if self.config.get("feature_vs_time") \
                and self.config.get("dt_field") is not None \
                and isinstance(self.config.get("dt_field"), str) \
                and self.config["dt_field"] not in self.config.get("features", []) \
                and self.config["dt_field"] not in self.config.get("aux_fields", []):
            raise ConfigError("invalid dt_field value, must be in features or aux_fields")

        # check psi_bin_type, csi_bin_type, and woe_bin_type (no key, None, 'fixed' or 'quantiles')
        for feature in {'psi_bin_type', 'csi_bin_type', 'woe_bin_type'}:
            bin_type = self.config.get(feature)
            if bin_type not in {None, 'fixed', 'quantiles'}:
                msg = f"if {feature} is present, it must be 'fixed', 'quantiles', or empty"
                raise ConfigError(msg)

        # check calibration_type (no key, None, 'isotonic' or 'logistic')
        if self.config.get('calibration_type') not in {None, 'isotonic', 'logistic'}:
            msg = "if 'calibration_type' is present, it must be 'isotonic', 'logistic', or empty"
            raise ConfigError(msg)

        # check calibration_train_dataset_name (must be valid dataset name)
        if self.config.get('model_calibration') and \
                self.config.get('calibration_train_dataset_name') not in self.config.get("data_file_patterns", []):
            msg = ("if 'model_calibration' is True, calibration_train_dataset_name must be a named"
                   " dataset in data_file_patterns")
            raise ConfigError(msg)

        # check no key, None, or castable to int (>1))
        for feature in {'feature_vs_time_n_bins', 'psi_n_bins', 'csi_n_bins', 'woe_n_bins', 'corr_max_features'}:
            num = self.config.get(feature)
            if num is not None:
                try:
                    int(num)
                except Exception as e:
                    raise ConfigError(f"{feature} exception converting to int: {e}")
                if int(num) <= 1:
                    raise ConfigError(f"if {feature} is an int, it should be > 1")
        # check no key, None, or castable to int (>=1))
        for feature in {'cv_folds', 'top_n_value_counts', 'n_splits'}:
            num = self.config.get(feature)
            if num is not None:
                try:
                    int(num)
                except Exception as e:
                    raise ConfigError(f"{feature} exception converting to int: {e}")
                if int(num) < 1:
                    raise ConfigError(f"if {feature} is an int, it should be >= 1")

        # check required boolean keys
        boolean_keys = {
            'save_scores', 'supervised', 'binary_classification'
        }
        for k in boolean_keys:
            if self.config[k] not in {True, False, None}:
                raise ConfigError(f"{k} must be True, False, or empty")

        # check non-required boolean keys
        boolean_keys = {
            'cross_validation', 'permutation_importance', 'shap',
            'feature_distribution', 'exclude_outliers', 'feature_vs_time',
            'psi', 'csi', 'vif', 'woe_iv', 'correlation', 'summary_stats',
            'binary_splits', 'model_calibration'
        }
        for k in boolean_keys:
            if self.config.get(k) not in {True, False, None}:
                raise ConfigError(f"if {k} is present, it must be True, False, or empty")

        # check quantiles (list of floats in [0, 1])
        if self.config.get('quantiles') is not None:
            if type(self.config["quantiles"]) is not list or len(self.config["quantiles"]) < 1:
                raise ConfigError("quantiles must be a list with len >= 1")
            for q in self.config["quantiles"]:
                if q < 0 or q > 1:
                    raise ConfigError("quantiles must be a list with all floats in [0, 1]")

    def run(self) -> None:
        """Run a complete experiment including (depending on config):
            1) data loading
            2) hyperparameter tuning (grid search, random search, tpe, or atpe)
            3) model training
            4) saving the model object
            5) model evaluation
            6) generating model explanitory artifacts
            7) saving model scores
            8) calibrating a model
            9) tear down
        """

        self.setup()
        self.train()
        self.save_model()
        self.evaluate(self.performance_increment)
        self.explain()
        if self.save_scores:
            self.gen_scores()
        if self.model_calibration:
            self.calibrate(calibration_type=self.calibration_type)
        self.tear_down()

    def setup(self) -> None:
        """
        Setup experiment by loading data, making directories for
        experiment output, and saving the config file.
        """

        # copy config to output_dir
        copyfile(self.config_path, self.output_dir/"config.yaml")
        self.logger.info(f"Config copied to {self.output_dir}/config.yaml")

        # load model
        if not self.config.get('model', None):
            self.load_model()

        # load data
        self.load_data()

    def load_model(self,
                   model_obj: Optional[BaseEstimator] = None,
                   path: Optional[str] = None) -> None:
        """
        Loads a model object from a parameter or file path, or instantiates a
        new model object.

        Parameters
        ----------
            model_obj : optional
                scikit-learn model object with a .predict_proba() method
                (default is None)
            path : str, optional
                file path to scikit-learn model object with a .predict_proba()
                method (default is None)
        """

        # use generic scikit-learn model object (if passed in)
        if model_obj is not None:
            if not isinstance(model_obj, BaseEstimator):
                msg = f"model_obj should be a scikit-learn model object, not {type(model_obj)}"
                self.logger.error(msg)
                raise TypeError(msg)
            self.model = model_obj
            self.logger.info("Model loaded from passed in model object")

        # load model from path (if passed in)
        elif path is not None:
            self._load_model_from_path(path)

        # load model from path (if specified in config)
        elif self.input_model_path:
            self._load_model_from_path(self.input_model_path)

        # instantiate new model
        else:
            self.logger.info("Initializing model")
            self.model = {
                'XGBClassifier': xgb.XGBClassifier(),
                'XGBRegressor': xgb.XGBRegressor(),
                'RandomForestClassifier': ensemble.RandomForestClassifier(),
                'RandomForestRegressor': ensemble.RandomForestRegressor(),
                'DecisionTreeClassifier': tree.DecisionTreeClassifier(),
                'DecisionTreeRegressor': tree.DecisionTreeRegressor(),
                'MLPClassifier': neural_network.MLPClassifier(),
                'MLPRegressor': neural_network.MLPRegressor(),
                'KNeighborsClassifier': neighbors.KNeighborsClassifier(),
                'KNeighborsRegressor': neighbors.KNeighborsRegressor(),
                'LogisticRegression': linear_model.LogisticRegression(),
                'LinearRegression': linear_model.LinearRegression(),
                'KMeans': cluster.KMeans(),
                'DBSCAN': cluster.DBSCAN(),
                'IsolationForest': ensemble.IsolationForest()
            }[self.model_type]

    def _load_model_from_path(self, path: str) -> None:
        """
        Loads a model object from a file path.

        Parameters
        ----------
            path : str
                file path to scikit-learn model object with a .predict_proba() method
        """

        self.logger.info(f"Loading model object from {path}")
        try:
            with open(path, 'rb') as f:
                self.model = pickle.load(f, encoding='latin1')
                self.logger.info(f"Model loaded from path: {path}")
        except Exception:
            self.logger.error(f"Error loading model from path: {path}")
            raise

    def load_data(self) -> None:
        """
        Loads in data, including training, validation, and testing data,
        and possibly other datasets as specified in the config.
        """

        self.logger.info("----- Loading Data -----")

        # get fields to load (features + label + auxiliary fields)
        fields = self.features[:]
        if self.supervised:
            fields.append(self.label)
        for f in self.aux_fields:
            if f is not None and f not in fields:
                fields.append(f)

        # load data
        for name, file_pattern in self.data_file_patterns.items():
            data_path = self.data_dir / file_pattern
            self.logger.info(f"Loading {name} data from {data_path}")
            df = dd.read_csv(data_path, usecols=fields, assume_missing=True)
            df = df.compute()
            df.reset_index(drop=True, inplace=True)  # removes index duplicates when loading multiple files
            df = shuffle(df, random_state=self.seed)
            self.data[name] = {
                'X': df[self.features]
            }
            # include label in supervised learning experiments
            if self.supervised:
                self.data[name]['y'] = df[self.label]
                if self.binary_classification:
                    self.data[name]['y'] = self.data[name]['y'].astype(int)
            # store aux data (in separate object so that **self.data[name] can be used)
            self.aux_data[name] = df[self.aux_fields]

    def train(self, **kwargs) -> None:
        """
        tune hyperparameters, then train a final model with the tuned
        hyperparmeters.
        """

        # initialize and tune hyperparamters
        self.tune_hyperparameters()

        # train model with optimal paramaters
        self.logger.info("----- Training Final Model -----")
        self.model.fit(**self.data['train'], **kwargs)

    def tune_hyperparameters(self) -> None:
        """
        Tune hyperparameters with either grid search, random search, tpe,
        or atpe.
        """

        # initialize hyperparamters
        self.model.set_params(**self.hyperparameters)

        # only tune if configured to
        if not (self.supervised and self.hyperparameter_tuning):
            return

        self.logger.info(f"----- Tuning Hyperparameters (via {self.tuning_algorithm} search) -----")

        # run grid search (if configured)
        if self.tuning_algorithm == 'grid':
            best_params = self._grid_search()

        # run random search, tpe, or atpe hyperparameter optimization algorithm
        elif self.tuning_algorithm in {"random", "tpe", "atpe"}:
            best_params = self._hyperopt_search()

        # write best params to file
        with open(self.log_dir/"parameter_tuning_log.txt", "a") as file:
            file.write(f"Best parameters: {best_params}\n\n")

        # set model to use best paramaters
        self.model.set_params(**best_params)

    def _grid_search_unparallelized(self) -> Dict[str, Union[str, int, float]]:
        """Tune hyperparameters with grid search."""

        # Grid search all possible combinations
        param_dict_list = ParameterGrid(self.tuning_parameters)
        scores = []
        for i, param_dict in enumerate(param_dict_list):
            self.logger.info(f"{i+1} out of {len(param_dict_list)}")
            score = self._train_eval_iteration(param_dict)
            scores.append(score)

        # get parameter set with best score
        if self.hyperparameter_eval_metric in {'average_precision', 'aucpr', 'auc'}:
            best = np.argmax
        elif self.hyperparameter_eval_metric in {'log_loss', 'brier_loss'}:
            best = np.argmin
        best_params: Dict[str, Union[str, int, float]] = param_dict_list[best(scores)]
        return best_params

    def _grid_search(self) -> Dict[str, Union[str, int, float]]:
        """Tune hyperparameters with grid search (in parallel)."""

        # make evaluation scorer
        metric = self.hyperparameter_eval_metric
        scorer = make_scorer(metric_score, metric=metric)

        # config inputs into GridSearchCV
        gs_kwargs = {
            'estimator': self.model,
            'param_grid': self.tuning_parameters,
            'scoring': scorer,
            'n_jobs': self.grid_search_n_jobs,
            'verbose': 3
        }
        if self.cross_validation:
            X = self.data['train']['X']
            y = self.data['train']['y']
            gs_kwargs['cv'] = self.cv_folds
        else:
            X = pd.concat([self.data['train']['X'], self.data['validation']['X']])
            y = pd.concat([self.data['train']['y'], self.data['validation']['y']])
            split_index = np.concatenate([-np.ones(len(self.data['train']['y'])), np.zeros(len(self.data['validation']['y']))])
            ps = PredefinedSplit(split_index)
            gs_kwargs['cv'] = ps
        model = GridSearchCV(**gs_kwargs)

        # Grid search all possible combinations
        model.fit(X, y)

        # get GridSearchCV metrics
        scores = model.cv_results_['mean_test_score']
        param_dict_list = model.cv_results_['params']
        mean_fit_times = model.cv_results_['mean_fit_time']

        # save score output to file
        with open(self.log_dir/"parameter_tuning_log.txt", "a") as file:
            for score, param_dict, fit_time in zip(scores, param_dict_list, mean_fit_times):
                msg = f"Parameters: {param_dict}\n{metric}: {score}\n"
                if self.cross_validation:
                    msg += "mean "
                msg += f"fit time: {fit_time} seconds\n\n"
                file.write(msg)

        # get parameter set with best score
        if metric in {'average_precision', 'aucpr', 'auc'}:
            best = np.argmax
        elif metric in {'log_loss', 'brier_loss'}:
            best = np.argmin
        best_params: Dict[str, Union[str, int, float]] = param_dict_list[best(scores)]
        return best_params

    def _hyperopt_search(self) -> Dict[str, Union[str, int, float]]:
        """Tune hyperparameters with either random search, tpe, or atpe."""

        # define optimization function
        def objective(param_dict):
            score = self._train_eval_iteration(param_dict)
            # if metric is to be maximized, then negate score, since objective() gets minimized
            if self.hyperparameter_eval_metric in {'average_precision', 'aucpr', 'auc'}:
                score = -score
            return {'loss': score, 'status': STATUS_OK}

        # map param kwargs to positional args (since atpe only works with positional arguments)
        def kwargs_to_args(distribution):
            dist_func_str = distribution['function']
            params = distribution['params']
            if dist_func_str == "choice":
                return (params["options"], )
            elif dist_func_str == "uniform":
                return (params["low"], params["high"])
            elif dist_func_str == "quniform":
                return (params["low"], params["high"], params["q"])
            elif dist_func_str == "normal":
                return (params["mu"], params["sigma"])

        # get parameter space
        space = {}
        distribution_functions = {
            'choice': hp.choice,
            'uniform': hp.uniform,
            # change quniform space to integers
            'quniform': lambda hyperparam, *params: pyll.scope.int(hp.quniform(hyperparam, *params)),
            'normal': hp.normal,
        }

        for hyperparam, distribution in self.tuning_parameters.items():
            # get the distribution function and parameters which specify the distribution
            # of possible hyperparameter values
            dist_func = distribution_functions[distribution['function']]
            params = kwargs_to_args(distribution)  # params = distribution['params']
            # add hyperparameter distribution to the hyperparameter space
            space[hyperparam] = dist_func(hyperparam, *params)

        # get tuning algorithm
        algo = {
            "random": rand.suggest,
            "tpe": tpe.suggest,
            "atpe": atpe.suggest,
        }[self.tuning_algorithm]

        # Run hyperparameter tuning
        trials = Trials()
        best_params: Dict[str, Union[str, int, float]]
        best_params = fmin(fn=objective,
                           space=space,
                           algo=algo,
                           max_evals=self.tuning_iterations,
                           trials=trials,
                           rstate=np.random.default_rng(self.seed))

        # correct for fmin casting integers to floats by casting them back to ints
        for hyperparameter, value in best_params.items():
            if int(value) == value:
                best_params[hyperparameter] = int(value)

        # save trial output
        with open(self.log_dir/"parameter_tuning_trials.txt", "a") as f1:
            for trial in trials.trials:
                f1.write(str(trial))
                f1.write("\n\n")
        self.misc_dir.mkdir(exist_ok=True)
        with open(self.misc_dir/"hyperopt_trials.pkl", "wb") as f2:
            pickle.dump(trials, f2)

        return best_params

    def _train_eval_iteration(self, param_dict: Dict[str, Union[str, int, float]]) -> float:
        """
        Run one iteration of training and evaluating a model for
        hyperparameter tuning.

        Parameters
        ----------
            param_dict : dict
                dict of parameters to configure an scikit-learn model object
        """

        start_time = time.time()
        self.logger.info(param_dict)
        score: float

        # train model
        self.model.set_params(**param_dict)
        metric = self.hyperparameter_eval_metric

        if self.cross_validation:
            # make evaluation scorer
            scorer = make_scorer(metric_score, metric=metric)
            # run cv to train and evaluate model
            cv_scores = cross_val_score(self.model, **self.data['train'], scoring=scorer)
            score = cv_scores.mean()
        else:
            # train model
            self.model.fit(**self.data['train'])
            # evaluate model (based on self.hyperparameter_eval_metric)
            val_name = 'validation' if 'validation' in self.data else 'test'
            y_true = self.data[val_name]['y']
            y_score = self.model.predict_proba(self.data[val_name]['X'])[:, 1]
            score = metric_score(y_true, y_score, metric)

        # print and log results
        self.logger.info(f"{metric}: {score}")
        seconds_to_train = round(time.time() - start_time, 2)
        self.logger.info(f"{seconds_to_train} seconds to train")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_dir/"parameter_tuning_log.txt", "a") as file:
            msg = (f"{now}\nParameters: {param_dict}\n{metric}: {score}\n{seconds_to_train}"
                   " seconds to train\n\n")
            file.write(msg)
        return score

    def save_model(self) -> None:
        """Save model object to file."""

        self.model_dir.mkdir(exist_ok=True)
        with open(self.model_dir/'model.pkl', 'wb') as file:
            pickle.dump(self.model, file)
        # TODO (?): also save pmml

    def evaluate(self, increment: float = 0.01) -> None:
        """Evaluate model and generate performance charts."""

        if not self.supervised:
            return

        # make output directory
        self.performance_dir.mkdir(exist_ok=True)

        # Instantiate ModelEvaluate object
        datasets = [(self.data[n]['X'], self.data[n]['y'], n) for n in self.dataset_names]
        self.model_eval = ModelEvaluate(model=self.model,
                                        datasets=datasets,
                                        output_dir=self.performance_dir,
                                        aux_fields=self.aux_fields,
                                        logger=self.logger)

        # generate binary classification metrics
        if self.binary_classification:
            self.model_eval.binary_evaluate(increment)

    def explain(self) -> None:
        """
        Generate model explanitory charts including feature importance
        and shap values.
        """

        # Instantiate ModelExplain object
        datasets = [(self.data[n]['X'], self.data[n]['y'], n) for n in self.dataset_names]
        aux_data = [self.aux_data[n] for n in self.dataset_names]
        model_explain = ModelExplain(model=self.model,
                                     datasets=datasets,
                                     aux_data=aux_data,
                                     output_dir=self.explain_dir,
                                     binary_classification=self.binary_classification,
                                     logger=self.logger)

        # Generate Permutation Feature Importance Tables
        if self.permutation_importance:
            n_repeats = self.perm_imp_n_repeats
            metrics = self.perm_imp_metrics
            model_explain.gen_permutation_importance(n_repeats, metrics, self.seed)

        # Generate Shap Charts
        if self.shap:
            model_explain.plot_shap(self.shap_sample)

        # Generate Feature Distribution Charts
        if self.feature_distribution:
            model_explain.plot_feature_distribution(self.exclude_outliers)

        # Generate Feature vs Time Charts
        if self.feature_vs_time:
            model_explain.plot_feature_vs_time(dt_field=self.dt_field,
                                               dt_format=self.dt_format,
                                               n_bins=self.feature_vs_time_n_bins)

        # Generate PSI Table
        if self.psi:
            model_explain.gen_psi(self.psi_bin_type, self.psi_n_bins)

        # Generate CSI Table
        if self.csi:
            model_explain.gen_csi(self.csi_bin_type, self.csi_n_bins)

        # Generate VIF Table
        if self.vif:
            model_explain.gen_vif()

        # Generate WOE and IV
        if self.woe_iv and self.binary_classification:
            model_explain.gen_woe_iv(self.woe_bin_type, self.woe_n_bins)

        # Generate Correlation Matrix and Heatmap
        if self.correlation:
            model_explain.gen_corr(self.corr_max_features)

        # Generate Summary Statistics
        if self.summary_stats:
            model_explain.gen_summary_statistics(self.quantiles, self.top_n_value_counts)

        # Generate Binary Splits Table
        if self.binary_splits:
            model_explain.gen_binary_splits(self.n_splits)

        # Generate XGBoost Explainability
        if isinstance(self.model, xgb.XGBModel):
            model_explain.xgb_explain()

    def gen_scores(self) -> None:
        """Save model scores for each row"""

        self.logger.info("----- Generating Scores -----")
        self.score_dir.mkdir(exist_ok=True)
        for dataset_name, dataset in self.data.items():
            scores = self.model.predict_proba(dataset['X'])[:, 1]
            scores = pd.Series(scores, name='score', index=dataset['y'].index)
            df = pd.concat([dataset['y'], scores, self.aux_data[dataset_name]], axis=1)
            path = f"{self.score_dir}/{dataset_name}_scores.csv"
            df.to_csv(path, index=False)
            self.logger.info(f"Saved {dataset_name} scores to {path}")

    def calibrate(self, calibration_type: str = 'logistic', bin_type: str = 'uniform', n_bins: int = 5) -> None:
        """
        Calibrate a binary classification model to output probability of true positive
        and generate performance metrics for the caalibrated model.

        Parameters
        ----------
            calibration_type : {'isotonic', 'logistic'}, default = 'logistic'
                The type of calibration model to fit
            bin_type : {'uniform', 'quantile'}, default = 'uniform'
                Strategy used to define the widths of the bins for the calibration plots
            n_bins : int > 1, default = 5
                Number of bins to discretize the [0, 1] interval in the calibration plots
        """

        # Check input
        valid_types = {'uniform', 'quantile'}
        assert bin_type in valid_types, f"bin_type must be in {valid_types}, not {bin_type}"
        assert self.binary_classification, "binary_classification must be True for .calibrate()"

        self.logger.info("----- Calibrating Model -----")

        # Instantiate ModelEvaluate object
        datasets = [(self.data[n]['X'], self.data[n]['y'], n) for n in self.dataset_names]
        model_calibrate = ModelCalibrate(self.model, datasets, self.calibration_dir, self.logger)

        # calibrate model
        model_calibrate.calibrate(self.calibration_train_dataset_name, calibration_type)

        # generate performance charts, tables, and metrics
        model_calibrate.evaluate(bin_type, n_bins)

        # save model calibration object
        model_calibrate.save_model()

    def tear_down(self):
        """
        Final actions at the end of an experiment, including:

        1) Stop our logger from writing to file (so the file isn't locked)
        """

        # remove handler that writes to file from logger
        for handler in self.logger.handlers:
            if getattr(handler, 'baseFilename', None):
                self.logger.removeHandler(handler)
                handler.close()

        # TODO? ...
