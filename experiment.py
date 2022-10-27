
import xgboost as xgb
from sklearn import ensemble, tree, neural_network, neighbors, linear_model, cluster, base
from sklearn.utils import shuffle
from sklearn.model_selection import ParameterGrid, cross_val_score, GridSearchCV, PredefinedSplit
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
import shap
import numpy as np
from pathlib import Path
import yaml, pickle, os, time, random
from datetime import datetime
from shutil import copyfile
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from hyperopt import fmin, rand, tpe, atpe, hp, STATUS_OK, Trials, pyll
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
ProgressBar().register()
import warnings
# warnings.filterwarnings('ignore')

pwd = r"C:\Users\hobbs\Documents\Programming\ML"
os.chdir(pwd)
from model_evaluation import ModelEvaluation, metric_score



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
    
    def __init__(self, config_path):
        """
        Constructs attributes from a config file

        Parameters
        ----------
            config_path : str
                path to yaml config file
        """

        print(f"\n-----Initializing {self.__class__.__name__}-----")

        # Load and validate config file
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            raise ConfigError(f"error loading config file: {e}")
        self.validate_config()
        
        # Set variables
        self.config_path = Path(config_path)
        
        #------ Meta Config -------
        self.version = self.config["version"]
        self.description = self.config["description"]

        #------ Input Config -------
        self.data_dir = Path(self.config["data_dir"])
        self.data_file_patterns = self.config["data_file_patterns"]
        self.input_model_path = self.config.get("input_model_path", None)
        
        #------ Output Config -------
        self.experiment_dir = Path(self.config["experiment_dir"])
        now = datetime.now().strftime("%Y%m%d-%H%M%S") # current datetime
        self.output_dir = self.experiment_dir / f"{self.version}-{now}"
        self.performance_dir = self.output_dir / self.config["performance_dir"]
        self.model_dir = self.output_dir / self.config["model_dir"]
        self.explain_dir = self.output_dir / self.config["explain_dir"]
        self.save_scores = self.config["save_scores"]
        if self.save_scores:
            self.score_dir = self.output_dir / self.config["score_dir"]
        
        #------ Job Config -------
        self.model_type = self.config["model_type"]
        self.supervised = self.config["supervised"]
        self.binary_classification = self.config["binary_classification"]
        self.label = None
        if self.supervised:
            self.label = self.config["label"]
        self.features = self.config["features"]
        self.aux_fields = self.config.get("aux_fields", None)
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
        self.cv_folds = self.config.get("cv_folds", 5)
        self.tuning_algorithm = self.config.get("tuning_algorithm", None)
        self.grid_search_n_jobs = self.config.get("grid_search_n_jobs", 1)
        self.tuning_iterations = self.config.get("tuning_iterations", None)
        self.tuning_parameters = self.config.get("tuning_parameters", None)

        # ------ Model explainability -------
        # Permutation Importance
        self.permutation_importance = self.config.get("permutation_importance", False)
        self.perm_imp_metrics = self.config.get("perm_imp_metrics", "neg_log_loss")
        if isinstance(self.perm_imp_metrics, str):
            self.perm_imp_metrics = [self.perm_imp_metrics]
        self.perm_imp_n_repeats = int(self.config.get("perm_imp_n_repeats", 10))
        # Shapely Values
        self.shap = self.config.get("shap", False)
        self.shap_sample = self.config.get("shap_sample", None)
        # Population Stability Index
        self.psi = self.config.get("psi", False)
        self.psi_bin_types = self.config.get("psi_bin_types", "fixed")
        self.psi_n_bins = self.config.get("psi_n_bins", 10)
        # Characteristic Stability Index
        self.csi = self.config.get("csi", False)
        self.csi_bin_types = self.config.get("csi_bin_types", "fixed")
        self.csi_n_bins = self.config.get("csi_n_bins", 10)
        # Variance Inflation Factor
        self.vif = self.config.get("vif", False)
        # WOE/IV
        self.woe_iv = self.config.get("woe_iv", False)
        self.woe_bin_types = self.config.get("woe_bin_types", "quantiles")
        self.woe_n_bins = self.config.get("woe_n_bins", 10)
        # Correlation
        self.correlation = self.config.get("correlation", False)
        self.corr_max_features = self.config.get("corr_max_features", 100)

        # ------ Other -------
        self.data = {} # where data will be stored
        self.aux_data = {} # where auxiliary fields will be stored
        self.model = None
        
        # specific order for dataset_names (for appropriate early stopping if enabled)
        all_names = set(self.data_file_patterns)
        main_names = {'train', 'test', 'validation'}
        other_names = sorted(all_names.difference(main_names))
        self.dataset_names = ['train'] + other_names
        self.dataset_names.extend([n for n in ['test', 'validation'] if n in all_names])


    def validate_config(self):
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
            'input_model_path', 'score_dir', 'label', 'aux_fields', 'verbose',
            'hyperparameter_tuning',  'hyperparameter_eval_metric',
            'cross_validation', 'cv_folds', 'tuning_algorithm',
            'grid_search_n_jobs',  'tuning_iterations', 'tuning_parameters',
            'permutation_importance', 'perm_imp_metrics', 'perm_imp_n_repeats',
            'shap', 'shap_sample', 'psi', 'psi_bin_types', 'psi_n_bins', 'csi',
            'csi_bin_types', 'csi_n_bins', 'vif', 'woe_iv', 'woe_bin_types',
            'woe_n_bins', 'correlation', 'corr_max_features'
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
            if not self.config.get("score_dir", None):
                raise ConfigError(f"missing score_dir key or value")
        
        # check that features is a list of length >= 1
        if type(self.config["features"]) != list or len(self.config["features"]) < 1:
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
        if self.config["model_type"] not in valid_model_types:
            raise ConfigError(f"invalid model_type: {model_type}")
        
        # check supervised is consistent with model_type
        if self.config["model_type"] in supervised_models and not self.config["supervised"]:
            raise ConfigError(f"supervised should be True when model_type = {model_type}")
        if self.config["model_type"] in unsupervised_models and self.config["supervised"]:
            raise ConfigError(f"supervised should be False when model_type = {model_type}")
        
        # check label is consistent with supervised
        if self.config["supervised"] and not self.config.get("label", None):
            raise ConfigError(f"need label when supervised = True")
        
        # check eval_metric
        if "eval_metric" in self.config.get("hyperparameters", []):            
            if not isinstance(self.config["hyperparameters"]["eval_metric"], list):
                if not isinstance(self.config["hyperparameters"]["eval_metric"], str):
                    raise ConfigError(f"eval_metric should be a string or list")
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
            if self.config.get("tuning_algorithm", None) not in {"grid", "random", "tpe", "atpe"}:
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
                if not self.config.get("tuning_iterations", None):
                    msg = "must specify tuning_iterations for the chosen tuning_algorithm"
                    raise ConfigError(msg)
            valid_metrics = {'average_precision', 'aucpr', 'auc', 'log_loss', 'brier_loss'}
            if self.config.get("hyperparameter_eval_metric", "log_loss") not in valid_metrics:
                raise ConfigError(f"invalid hyperparameter_eval_metric value")

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
                    param_to_datatype = {
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
                            msg = (f"tuning_parameters.{hyperparameter}.params.{param} must"
                                    " be > 0")
                            raise ConfigError(msg)

        # Note: Not checking perm_imp_metrics since there are many possible values that work
        
        # check perm_imp_n_repeats
        try:
            int(self.config.get("perm_imp_n_repeats", 10))
        except Exception as e:
            raise ConfigError(f"perm_imp_n_repeats exception converting to int: {e}")
        
        # check shap_sample (either no key, None or castable to int)
        shap_sample = self.config.get("shap_sample", None)
        if shap_sample is not None:
            try:
                int(shap_sample)
            except Exception as e:
                raise ConfigError(f"shap_sample exception converting to int: {e}")

        # check psi_bin_types, csi_bin_types, and woe_bin_types (no key, None, 'fixed' or 'quantiles')
        for feature in {'psi_bin_types', 'csi_bin_types', 'woe_bin_types'}:
            bin_types = self.config.get(feature, None)
            if bin_types not in {None, 'fixed', 'quantiles'}:
                msg = f"if {feature} is present, it must be 'fixed', 'quantiles', or empty"
                raise ConfigError(msg)

        # check no key, None, or castable to int (>1))
        for feature in {'psi_n_bins', 'csi_n_bins', 'woe_n_bins', 'corr_max_features'}:
            num = self.config.get(feature, None)
            if num is not None:
                try:
                    int(num)
                except Exception as e:
                    raise ConfigError(f"{feature} exception converting to int: {e}")
                if int(num) <= 1:
                    raise ConfigError(f"if {feature} is an int, it should be > 1")
        # check no key, None, or castable to int (>=1))
        for feature in {'cv_folds'}:
            num = self.config.get(feature, None)
            if num is not None:
                try:
                    int(num)
                except Exception as e:
                    raise ConfigError(f"{feature} exception converting to int: {e}")
                if int(num) < 1:
                    raise ConfigError(f"if {feature} is an int, it should be > 1")

        # check required boolean keys
        boolean_keys = {
            'save_scores', 'supervised', 'binary_classification'
        }
        for k in boolean_keys:
            if self.config[k] not in {True, False, None}:
                raise ConfigError(f"{k} must be True, False, or empty")
        
        # check non-required boolean keys
        boolean_keys = {
            'cross_validation', 'permutation_importance', 'shap', 'psi', 'csi',
            'vif', 'woe_iv', 'correlation',
        }
        for k in boolean_keys:
            if self.config.get(k, None) not in {True, False, None}:
                raise ConfigError(f"if {k} is present, it must be True, False, or empty")

    def run(self):
        """Run a complete experiment including (depending on config):
            1) data loading
            2) hyperparameter tuning (grid search, random search, tpe, or atpe)
            3) model training
            4) saving the model object
            5) model evaluation
            6) generating model explanitory artifacts
            7) saving model scores
        """

        self.setup()
        self.train()
        self.save_model()
        self.evaluate()
        self.explain()
        if self.save_scores:
            self.gen_scores()

    def setup(self):
        """
        Setup experiment by loading data, making directories for
        experiment output, and saving the config file.
        """

        # make output dirs
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.performance_dir.mkdir(exist_ok=True)
        
        # copy config to output_dir
        copyfile(self.config_path, self.output_dir/"config.yaml")
        print(f"config copied to {self.output_dir}/config.yaml")
        
        # load model
        if not self.model:
            self.load_model()
        
        # load data
        self.load_data()

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

        # use generic scikit-learn model object (if passed in)
        if model_obj is not None:
            if self.model_type != 'Other':
                msg = (f"model_type should be 'Other', not {self.model_type} if a"
                        " model_obj is being passed into .load_model()")
                raise ConfigError(msg)
            if not isinstance(model_obj, base.BaseEstimator):
                msg = f"model_obj should be a scikit-learn model object, not {type(model_obj)}"
                raise TypeError(msg)
            self.model = model_obj
            print("model loaded from passed in model object")
        
        # load model from path (if passed in)
        elif path is not None:
            self._load_model_from_path(path)
        
        # load model from path (if specified in config)
        elif self.input_model_path:
            self._load_model_from_path(self.input_model_path)
        
        # instantiate new model
        else:
            print("Initializing model")
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

    def _load_model_from_path(self, path):
        """
        Loads a model object from a file path.

        Parameters
        ----------
            path : str
                file path to scikit-learn model object with a .predict_proba() method
        """

        print(f"Loading model object from {path}")
        try:
            with open(path, 'rb') as f:
                self.model = pickle.load(f, encoding='latin1')
                print(f"model loaded from path: {path}")
        except Exception as e:
            print(f"error loading model from path: {path}")
            raise

    def load_data(self):
        """
        Loads in data, including training, validation, and testing data,
        and possibly other datasets as specified in the config.
        """

        print(f"\n-----Loading Data-----")
        
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
            print(f"loading {name} data from {data_path}")
            df = dd.read_csv(data_path, usecols=fields)
            df = df.compute()
            df = shuffle(df, random_state=self.seed)
            self.data[name] = {
                'X': df[self.features]
            }
            # include label in supervised learning experiments
            if self.supervised:
                self.data[name]['y'] = df[self.label]
            # store aux data (in separate object so that **self.data[name] can be used) 
            self.aux_data[name] = df[self.aux_fields]

    def train(self, **kwargs):
        """
        tune hyperparameters, then train a final model with the tuned
        hyperparmeters.
        """

        # initialize and tune hyperparamters
        self.tune_hyperparameters()
        
        # train model with optimal paramaters
        print(f"\n-----Training Final Model-----")
        self.model.fit(**self.data['train'], **kwargs)

    def tune_hyperparameters(self):
        """
        Tune hyperparameters with either grid search, random search, tpe,
        or atpe.
        """

        # initialize hyperparamters
        self.model.set_params(**self.hyperparameters)
        
        # only tune if configured to
        if not (self.supervised and self.hyperparameter_tuning):
            return  

        print(f"\n-----Tuning Hyperparameters (via {self.tuning_algorithm} search)-----")
        
        # run grid search (if configured)
        if self.tuning_algorithm == 'grid':
            best_params = self._grid_search()
        
        # run random search, tpe, or atpe hyperparameter optimization algorithm 
        elif self.tuning_algorithm in {"random", "tpe", "atpe"}:
            best_params = self._hyperopt_search()
        
        # write best params to file
        with open(self.performance_dir/"parameter_tuning_log.txt", "a") as file:
            file.write(f"Best parameters: {best_params}\n\n")
        
        # set model to use best paramaters 
        self.model.set_params(**best_params)

    def _grid_search_unparallelized(self):
        """Tune hyperparameters with grid search."""

        # Grid search all possible combinations
        param_dict_list = ParameterGrid(self.tuning_parameters)
        scores = []
        for i, param_dict in enumerate(param_dict_list):
            print(f"{i+1} out of {len(param_dict_list)}")
            score = self._train_eval_iteration(param_dict)
            scores.append(score)
        
        # get parameter set with best score
        if self.hyperparameter_eval_metric in {'average_precision', 'aucpr', 'auc'}:
            best = np.argmax
        elif self.hyperparameter_eval_metric in {'log_loss', 'brier_loss'}:
            best = np.argmin
        best_params = param_dict_list[best(scores)]
        return best_params

    def _grid_search(self):
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
        with open(self.performance_dir/"parameter_tuning_log.txt", "a") as file:
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
        best_params = param_dict_list[best(scores)]
        return best_params

    def _hyperopt_search(self):
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
            params = kwargs_to_args(distribution) # params = distribution['params']
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
        best_params = fmin(fn=objective, space=space, algo=algo, 
            max_evals=self.tuning_iterations, trials=trials, 
            rstate=np.random.default_rng(self.seed))

        # correct for min casting integers to floats by casting them back to ints
        for hyperparameter, value in best_params.items():
            if int(value) == value:
                best_params[hyperparameter] = int(value)

        # save trial output
        with open(self.performance_dir/"parameter_tuning_trials.txt", "a") as file:
            for trial in trials.trials:
                file.write(str(trial))
                file.write("\n\n")

        return best_params

    def _train_eval_iteration(self, param_dict):
        """
        Run one iteration of training and evaluating a model for
        hyperparameter tuning.

        Parameters
        ----------
            param_dict : dict
                dict of parameters to configure an scikit-learn model object
        """

        start_time = time.time()
        print(param_dict)
        
        # train model
        self.model.set_params(**param_dict)
        metric = self.hyperparameter_eval_metric

        if self.cross_validation:
            # make evaluation scorer
            scorer = make_scorer(metric_score, metric=metric)
            # run cv to train and evaluate model
            cv_scores = cross_val_score(self.model, **self.data['train'], scoring=scorer) # .....
            score = cv_scores.mean()
        else:
            # train model
            self.model.fit(**self.data['train'])
            # evaluate model (based on self.hyperparameter_eval_metric)
            val_name = 'validation' if 'validation' in self.data else 'test'
            y_true = self.data[val_name]['y']
            y_score = self.model.predict_proba(self.data[val_name]['X'])
            score = metric_score(y_true, y_score, metric)
        
        # print and log results
        print(f"{metric}: {score}")
        seconds_to_train = time.time() - start_time
        print(f"{seconds_to_train} seconds to train")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.performance_dir/"parameter_tuning_log.txt", "a") as file:
            msg = (f"{now}\nParameters: {param_dict}\n{metric}: {score}\n{seconds_to_train}"
                    " seconds to train\n\n")
            file.write(msg)
        return score

    def save_model(self):
        """Save model object to file."""

        self.model_dir.mkdir(exist_ok=True)
        with open(self.model_dir/'model.pkl', 'wb') as file:
            pickle.dump(self.model, file)
        # TODO (?): also save pmml

    def evaluate(self, increment=0.01):
        """Evaluate model and generate performance charts."""
        
        if not self.supervised:
            return

        datasets = [(self.data[n]['X'], self.data[n]['y'], n) for n in self.dataset_names] 
        self.model_eval = ModelEvaluation(self.model, datasets, self.performance_dir, self.aux_fields)

        # generate binary classification metrics
        if self.binary_classification:
            self.model_eval.binary_evaluate(increment)

    def explain(self):
        """
        Generate model explanitory charts including feature importance
        and shap values.
        """

        # Generate Permutation Feature Importance Tables
        if self.permutation_importance:
            n_repeats = self.perm_imp_n_repeats
            metrics = self.perm_imp_metrics
            self.gen_permutation_importance(n_repeats, metrics)

        # Generate Shap Charts
        if self.shap:
            self.plot_shap()

        # Generate PSI Table
        if self.psi:
            self.gen_psi(bin_types=self.psi_bin_types, n_bins=self.psi_n_bins)

        # Generate CSI Table
        if self.csi:
            self.gen_csi(bin_types=self.csi_bin_types, n_bins=self.csi_n_bins)

        # Generate VIF Table
        if self.vif:
            self.gen_vif()

        # Generate WOE and IV
        if self.woe_iv and self.binary_classification:
            self.gen_woe_iv()

        # Generate Correlation Matrix and Heatmap
        if self.correlation:
            self.gen_corr(max_features=self.corr_max_features)

    def gen_permutation_importance(self, n_repeats=10, metrics='neg_log_loss'):
        """
        Generate permutation feature importance tables.
        
        Parameters
        ----------
            n_repeats : int, optional
                number of times to permute each feature (default is 10)
            metrics : str or list of str, optional
                metrics used in permutation feature importance calculations (default is 'neg_log_loss').
                e.g.: 'roc_auc', 'average_precision', 'neg_log_loss', 'r2', etc.
                see https://scikit-learn.org/stable/modules/model_evaluation.html for complete list.
        """

        print(f"\n-----Generating Permutation Feature Importances-----")

        # make output directory
        self.explain_dir.mkdir(exist_ok=True)
        
        # generate permutation feature importance for each metric on each dataset
        metrics = self.perm_imp_metrics
        for name in self.dataset_names:
            print(f"running permutation importance on {name} data")
            r = permutation_importance(self.model, **self.data[name],
                n_repeats=n_repeats, random_state=self.seed,
                scoring=metrics)
            imps = []
            for m in metrics:
                # get means and standard deviations of feature importance
                means = pd.Series(r[m]['importances_mean'], name=f"{m}_mean")
                stds = pd.Series(r[m]['importances_std'], name=f"{m}_std")
                imps.extend([means, stds])
            df = pd.concat(imps, axis=1) # dataframe of importance means and stds
            df.index = self.features[:]
            df.sort_values(f"{metrics[0]}_mean", ascending=False, inplace=True)
            df.to_csv(f'{self.explain_dir}/permutation_importance_{name}.csv')

    def plot_shap(self):
        """Generate model explanitory charts involving shap values."""

        plt.close('all')
        
        # Generate Shap Charts
        print(f"\n-----Generating Shap Charts-----")
        savefig_kwargs = {'bbox_inches': 'tight', 'pad_inches': 0.2}
        predict = lambda x: self.model.predict_proba(x)[:,1]
        for dataset_name in self.dataset_names:
            
            # get sample of dataset (gets all data if self.shap_sample is None)
            dataset = self.data[dataset_name]['X'].iloc[:self.shap_sample]
            if len(dataset) > 500000:
                msg = (f"Shap will be slow on {len(dataset)} rows, consider using"
                        " shap_sample in the config to sample fewer rows")
                warnings.warn(msg)

            # Generate partial dependence plots 
            print(f'Plotting {dataset_name} partial dependence plots')
            plot_dir = self.explain_dir/"shap"/dataset_name/"partial_dependence_plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            for feature in tqdm(self.features):        
                fig, ax = shap.partial_dependence_plot(
                    feature, predict, dataset, model_expected_value=True,
                    feature_expected_value=True, show=False, ice=False)
                fig.savefig(f"{plot_dir}/{feature}.png", **savefig_kwargs)
                plt.close()

            # Generate scatter plots (coloured by feature with strongest interaction)
            print(f'Plotting {dataset_name} scatter plots')
            explainer = shap.Explainer(self.model, dataset)
            shap_values = explainer(dataset)
            plot_dir = self.explain_dir/"shap"/dataset_name/"scatter_plots"
            plot_dir.mkdir(exist_ok=True)
            for feature in tqdm(self.features):        
                shap.plots.scatter(shap_values[:,feature], alpha=0.3, 
                    color=shap_values, show=False)
                plt.savefig(f"{plot_dir}/{feature}.png", **savefig_kwargs)
                plt.close()

            # Generate beeswarm plot
            print(f'Plotting {dataset_name} beeswarm plot')
            shap.plots.beeswarm(shap_values, alpha=0.1, max_display=1000, show=False)
            path = self.explain_dir/"shap"/dataset_name/"beeswarm_plot.png"
            plt.savefig(path, **savefig_kwargs)
            plt.close()

            # Generate bar plots
            print(f'Plotting {dataset_name} bar plots')
            shap.plots.bar(shap_values, max_display=1000, show=False)
            path = self.explain_dir/"shap"/dataset_name/"abs_mean_bar_plot.png"
            plt.savefig(path, **savefig_kwargs)
            plt.close()
            shap.plots.bar(shap_values.abs.max(0), max_display=1000, show=False)
            path = self.explain_dir/"shap"/dataset_name/"abs_max_bar_plot.png"
            plt.savefig(path, **savefig_kwargs)
            plt.close()
            # TODO?: make alpha and max_display config variables

    def gen_psi(self, bin_types='fixed', n_bins=10):
        """
        Generate Population Stability Index (PSI) values between all pairs of datasets.

               PSI < 0.1 => no significant population change
        0.1 <= PSI < 0.2 => moderate population change
        0.2 <= PSI       => significant population change

        Note: PSI is symmetric provided the bins are the same, which they are when bin_types='fixed'
        
        Parameters
        ----------
            bin_types : str, optional
                the method for choosing bins, either 'fixed' or 'quantiles' (default is 'fixed')
            n_bins : int, optional
                the number of bins used to compute psi (default is 10)
        """

        print(f"\n-----Generating PSI-----")

        # check for valid input
        assert bin_types in {'fixed', 'quantiles'}, "bin_types must be in {'fixed', 'quantiles'}"

        # intialize output dataframe
        psi_df = pd.DataFrame(columns=['dataset1', 'dataset2', 'psi'])
        
        # get dictionary of scores for all datasets
        scores_dict = {}
        for dataset_name, dataset in self.data.items():
            scores = self.model.predict_proba(dataset['X'])[:,1]
            scores.sort()
            scores_dict[dataset_name] = scores
        
        # compute psi for each pair of datasets
        for i, dataset_name1 in enumerate(self.dataset_names):
            for j in range(i+1, len(self.dataset_names)):
                dataset_name2 = self.dataset_names[j]
                scores1 = scores_dict[dataset_name1]
                scores2 = scores_dict[dataset_name2]
                psi_val = self._psi_compare(scores1, scores2, bin_types=bin_types, n_bins=n_bins)
                row = {
                    'dataset1': dataset_name1,
                    'dataset2': dataset_name2,
                    'psi': psi_val
                }
                psi_df = psi_df.append(row, ignore_index=True)
        
        # save output to csv
        self.explain_dir.mkdir(exist_ok=True)
        psi_df.to_csv(self.explain_dir/'psi.csv', index=False)

    def _psi_compare(self, scores1, scores2, bin_types='fixed', n_bins=10):
        """
        Compute Population Stability Index (PSI) between two datasets.

        Parameters
        ----------
            scores1 : numpy.ndarray or pandas.core.series.Series
                scores for one of the datasets
            scores2 : numpy.ndarray or pandas.core.series.Series
                scores for the other dataset
            bin_types : str, optional
                the method for choosing bins, either 'fixed' or 'quantiles' (default is 'fixed')
            n_bins : int, optional
                the number of bins used to compute psi (default is 10)
        ...
        """

        # get bins
        min_val = min(min(scores1), min(scores2)) # TODO? could bring this up a function for efficiency
        max_val = max(min(scores1), max(scores2))
        if bin_types == 'fixed':
            bins = [min_val + (max_val - min_val) * i / n_bins for i in range(n_bins + 1)]
        elif bin_types == 'quantiles':
            bins = pd.qcut(scores1, q=n_bins, retbins=True, duplicates='drop')[1]
            n_bins = len(bins) - 1 # some bins could be dropped due to duplication
        eps = 1e-6
        bins[0] -= -eps
        bins[-1] += eps

        # group data into bins and get percentage rates
        scores1_bins = pd.cut(scores1, bins=bins, labels=range(n_bins))
        scores2_bins = pd.cut(scores2, bins=bins, labels=range(n_bins))
        df1 = pd.DataFrame({'score1': scores1, 'bin': scores1_bins})
        df2 = pd.DataFrame({'score2': scores2, 'bin': scores2_bins})
        grp1 = df1.groupby('bin').count()['score1']
        grp2 = df2.groupby('bin').count()['score2']
        grp1_rate = (grp1 / sum(grp1)).rename('rate1')
        grp2_rate = (grp2 / sum(grp2)).rename('rate2')
        grp_rates = pd.concat([grp1_rate, grp2_rate], axis=1).fillna(0)

        # add a small value when the percent is zero
        grp_rates = grp_rates.applymap(lambda x: eps if x == 0 else x)

        # calculate psi
        psi_vals = (grp_rates['rate1'] - grp_rates['rate2']) * np.log(grp_rates['rate1'] / grp_rates['rate2'])
        return psi_vals.mean()

    def gen_csi(self, bin_types='fixed', n_bins=10):
        """
        Generate Characteristic Stability Index (CSI) values for all features between all pairs of datasets.

        Note: CSI is symmetric provided the bins are the same, which they are when bin_types='fixed'
        
        Parameters
        ----------
            bin_types : str, optional
                the method for choosing bins, either 'fixed' or 'quantiles' (default is 'fixed')
            n_bins : int, optional
                the number of bins used to compute csi (default is 10)
        """
        
        print(f"\n-----Generating CSI-----")

        # check for valid input
        assert bin_types in {'fixed', 'quantiles'}, "bin_types must be in {'fixed', 'quantiles'}"

        # intialize output dataframe
        csi_df = pd.DataFrame(columns=['dataset1', 'dataset2', 'feature', 'csi'])

        for feature in tqdm(self.features):
            
            # get dictionary of values for the given feature from all datasets
            vals_dict = {}
            for dataset_name, dataset in self.data.items():
                scores = dataset['X'][feature]
                scores.sort_values()
                vals_dict[dataset_name] = scores
            
            # compute csi for each pair of datasets
            for i, dataset_name1 in enumerate(self.dataset_names):
                for j in range(i+1, len(self.dataset_names)):
                    dataset_name2 = self.dataset_names[j]
                    scores1 = vals_dict[dataset_name1]
                    scores2 = vals_dict[dataset_name2]
                    csi_val = self._psi_compare(scores1, scores2, bin_types=bin_types, n_bins=n_bins)
                    row = {
                        'dataset1': dataset_name1,
                        'dataset2': dataset_name2,
                        'feature': feature,
                        'csi': csi_val
                    }
                    csi_df = csi_df.append(row, ignore_index=True)
            
        # save output to csv
        self.explain_dir.mkdir(exist_ok=True)
        csi_df.sort_values('csi', ascending=False, inplace=True)
        csi_df.to_csv(self.explain_dir/'csi_long.csv', index=False)

        # convert csi dataframe to wide format
        csi_df['datasets'] = csi_df['dataset1'] + '-' + csi_df['dataset2']
        csi_df = csi_df[['feature','datasets','csi']]
        csi_df.set_index('feature', inplace=True)
        csi_df = csi_df.pivot(columns='datasets')['csi']
        
        # reorder to the same order as self.features, and save to csv
        csi_df.reset_index(inplace=True)
        csi_df['feature'] = pd.Categorical(csi_df['feature'], self.features)
        csi_df = csi_df.sort_values("feature").set_index("feature")
        csi_df.to_csv(self.explain_dir/'csi_wide.csv')

    def gen_vif(self):
        """
        Generate Variance Inflation Factor (VIF) tables for each dataset.

        VIF = 1  => no correlation
        VIF > 10 => high correlation between an independent variable and the others
        """

        print(f"\n-----Generating VIF-----")
        
        # make directory for VIF tables
        vif_dir = self.explain_dir / "vif"
        vif_dir.mkdir(parents=True, exist_ok=True)

        # calculate VIF values for each feature in each dataset
        for dataset_name, dataset in tqdm(self.data.items()):
            df = pd.DataFrame({'feature': self.features})
            X = dataset['X'].values
            df["vif"] = [variance_inflation_factor(X, i) for i in range(len(self.features))]
            df.to_csv(vif_dir/f'vif_{dataset_name}.csv', index=False)

    def gen_woe_iv(self, bin_types='quantiles', n_bins=10):
        """
        Generate Weight of Evidence and Information Value tables for each dataset.

                IV < 0.02 => not useful for prediction
        0.02 <= IV < 0.1  => weak predictive power
        0.1  <= IV < 0.3  => medium predictive power
        0.3  <= IV < 0.5  => strong predictive power
        0.5  <= IV        => suspicious predictive power

        Parameters
        ----------
            bin_types : str, optional
                the method for choosing bins, either 'fixed' or 'quantiles' (default is 'quantiles')
            n_bins : int, optional
                the number of bins used to compute woe and iv (default is 10)
        """

        if not self.binary_classification:
            msg = "self.binary_classification must be True to run .gen_woe_iv()"
            warnings.warn(msg)
            return

        print(f"\n-----Generating WOE and IV-----")

        for dataset_name, dataset in self.data.items():

            print(f"generating woe and iv for {dataset_name} dataset")

            # make output directory
            woe_dir = self.explain_dir / 'woe_iv'
            woe_dir.mkdir(parents=True, exist_ok=True)

            # initialize lists to accumulate data
            woe_df_list = []
            iv_list = []

            # get labels
            y_true = self.data[dataset_name]['y']
            
            # generate woe and iv for all features
            for feature in tqdm(self.features):

                # get feature data
                values = self.data[dataset_name]['X'][feature]

                # get bins
                if bin_types == 'fixed':
                    min_val = min(values)
                    max_val = max(values)
                    bins = [min_val + (max_val - min_val) * i / n_bins for i in range(n_bins + 1)]
                elif bin_types == 'quantiles':
                    bins = pd.qcut(values, q=n_bins, retbins=True, duplicates='drop')[1]
                eps = 1e-6
                bins[0] -= -eps # add buffer to include points right at the edge.
                bins[-1] += eps

                # group data into bins
                value_bins = pd.cut(values, bins=bins)
                df = pd.DataFrame({'label': y_true, 'bin': value_bins})
                
                # get counts
                df = df.groupby(['bin']).agg({'label': [sum,len]})['label']
                df['cnt_0'] = df['len'] - df['sum'] # count of 0-label events
                df.rename(columns={'sum': 'cnt_1'}, inplace=True) # count of 1-label events

                # reformat dataframe
                df.drop(columns=['len'], inplace=True)
                df.reset_index(inplace=True)
                df['feature'] = feature # add feature
                df = df.reindex(columns=['feature', 'bin', 'cnt_0','cnt_1']) # reorder columns
                
                # get rates
                df['pct_0'] = df['cnt_0'] / df['cnt_0'].sum()
                df['pct_1'] = df['cnt_1'] / df['cnt_1'].sum()

                # get WOEs and IV
                df['woe'] = np.log(df['pct_1'] / df['pct_0'])
                df['adj_woe'] = np.log(
                    ((df['cnt_1'] + 0.5) / df['cnt_1'].sum())
                  / ((df['cnt_0'] + 0.5) / df['cnt_0'].sum()))
                iv = (df['woe'] * (df['pct_1'] - df['pct_0'])).sum()
                adj_iv = (df['adj_woe'] * (df['pct_1'] - df['pct_0'])).sum()

                # append to lists
                woe_df_list.append(df)
                iv_list.append({'feature': feature, 'iv': iv, 'adj_iv': adj_iv})

            woe_df = pd.concat(woe_df_list)
            woe_df.to_csv(woe_dir/f'woe_{dataset_name}.csv', index=False)

            iv_df = pd.DataFrame.from_records(iv_list)
            iv_df.sort_values('adj_iv', ascending=False, inplace=True)
            iv_df.index.name = 'index'
            iv_df.to_csv(woe_dir/f'iv_{dataset_name}.csv')

    def gen_corr(self, max_features=100):
        """
        Generate correlation matrix and heatmap for each dataset.

        Parameters
        ----------
            max_features : int, optional
                the maximum number of features allowed for charts and plots to be generated
        """

        # check input
        if len(self.features) > max_features:
            msg = (f"not computing correlation matrix since there are {len(self.features)}"
                   f" features, which more than max_features = {max_features}")
            warnings.warn(msg)
            return
        
        print(f"\n-----Generating Correlation Charts-----")

        # make output directory
        corr_dir = self.explain_dir / 'correlation'
        corr_dir.mkdir(parents=True, exist_ok=True)

        for dataset_name, dataset in self.data.items():
            print(f"generating correlations for {dataset_name} dataset")
            corr = dataset['X'].corr()
            corr_long = pd.melt(corr.reset_index(), id_vars='index') # unpivot to long format
            # write to csv
            corr.index.name = 'feature'
            corr.to_csv(corr_dir/f'corr_{dataset_name}.csv')
            corr_long.rename(columns={'variable': 'feature_1', 'index': 'feature_2', 'value': 'correlation'}, inplace=True)
            corr_long = corr_long.reindex(columns=['feature_1', 'feature_2', 'correlation']) # reorder columns
            corr_long = corr_long[corr_long.feature_1 != corr_long.feature_2]
            corr_long.sort_values('correlation', key=abs, ascending=False, inplace=True)
            corr_long.to_csv(corr_dir/f'corr_{dataset_name}_long.csv', index=False)
            # plot heat map
            self.plot_corr_heatmap(corr, corr_dir/f'heatmap_{dataset_name}.png', data_type='corr')


    def plot_corr_heatmap(self, data, output_path, data_type='corr'):
        """
        Plot correlation heat map.

        Parameters
        ----------
            data : dataframe
                input data, either raw data, or correlation matrix, or long-format correlation matrix
            output_path : str
                the location where the plot should be written.
                note that this function assumes that the parent folder exists.
            data_type : str, optional
                the type of input passed into the data argument (default is 'corr')
                - 'features' => the raw feature table is passed in
                - 'corr' => a correlation matrix is passed in
                - 'corr_long' => a correlation matrix converted to long format is passed in

        Credits: https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
        """

        # check inputs
        valid_data_types = {'features', 'corr', 'corr_long'}
        assert data_type in valid_data_types, f"invalid data_type: {data_type}"

        # process data into long-format correlation matrix, if required
        if data_type == 'features':
            data = data.corr()
        if data_type in {'features', 'corr'}:
            data.index.name = 'index'
            data = pd.melt(data.reset_index(), id_vars='index') # unpivot to long format
            data = data.reindex(columns=['variable', 'index', 'value']) # reorder columns
        data.columns = ['feature_1', 'feature_2', 'correlation']

        features = self.features

        # set up colours
        n_colors = 256 # Use 256 colors for the diverging color palette
        palette = sns.diverging_palette(20, 220, n=n_colors) # Create the palette
        color_min, color_max = [-1, 1] # Range of values that will be mapped to the palette, i.e. min and max possible correlation

        def value_to_color(val):
            val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
            ind = int(val_position * (n_colors - 1)) # target index in the color palette
            return palette[ind]  

        # parameterize sizes of objects in the image
        size_factor = 1
        plot_size = len(features) * size_factor
        figsize = (plot_size * 15 / 14, plot_size) # multiple by 15/14 so the left main plot is square
        font_size = 20 * size_factor
        square_size_scale = (40 * size_factor) ** 2
        size = data['correlation'].abs() # size of squares is dependend on absolute correlation
        
        # map feature to integer coordinates
        feat_to_num = {feature: i for i, feature in enumerate(features)} 

        with plt.style.context('seaborn-darkgrid'): 
         
            # create figure
            fig = plt.figure(figsize=figsize, dpi=50)
            plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1) # Setup a 1x15 grid
            
            # Use the leftmost 14 columns of the grid for the main plot
            ax = plt.subplot(plot_grid[:,:-1])

            # make main plot
            ax.scatter(
                x=data['feature_1'].map(feat_to_num), # Use mapping for feature 1
                y=data['feature_2'].map(feat_to_num), # Use mapping for feature 2
                s=size * square_size_scale, # Vector of square sizes, proportional to size parameter
                c=data['correlation'].apply(value_to_color), # Vector of square color values, mapped to color palette
                marker='s' # Use square as scatterplot marker
            )
            
            # Show column labels on the axes
            ax.set_xticks([feat_to_num[v] + 0.3 for v in features]) # add major ticks for the labels
            ax.set_yticks([feat_to_num[v] for v in features])
            ax.set_xticklabels(features, rotation=45, horizontalalignment='right', fontsize=font_size) # add labels
            ax.set_yticklabels(features, fontsize=font_size)
            ax.grid(False, 'major') # hide major grid lines
            ax.grid(True, 'minor')
            ax.set_xticks([feat_to_num[v] + 0.5 for v in features], minor=True) # add minor ticks for grid lines
            ax.set_yticks([feat_to_num[v] + 0.5 for v in features], minor=True)

            # set axis limits
            ax.set_xlim([-0.5, len(features) - 0.5]) 
            ax.set_ylim([-0.5, len(features) - 0.5])

            # hide all ticks
            plt.tick_params(axis='both', which='both', bottom=False, left=False)        

            # Add color legend on the right side of the plot
            ax = plt.subplot(plot_grid[:,-1]) # Use the rightmost column of the plot

            col_x = [0] * len(palette) # Fixed x coordinate for the bars
            bar_y = np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars

            bar_height = bar_y[1] - bar_y[0]
            ax.barh(
                y=bar_y,
                width=[5]*len(palette), # Make bars 5 units wide
                left=col_x, # Make bars start at 0
                height=bar_height,
                color=palette,
                linewidth=0
            )
            ax.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
            ax.grid(False) # Hide grid
            ax.set_facecolor('white') # Make background white
            ax.set_xticks([]) # Remove horizontal ticks
            ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3)) # Show vertical ticks for min, middle and max
            ax.set_yticklabels([-1, 0, 1], fontsize=font_size)
            ax.yaxis.tick_right() # Show vertical ticks on the right 

            # save figure
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()

    def gen_scores(self):
        """Save model scores for each row"""

        print(f"\n-----Generating Scores-----")
        self.score_dir.mkdir(exist_ok=True)
        for dataset_name, dataset in tqdm(self.data.items()):
            scores = self.model.predict_proba(dataset['X'])[:,1]
            scores = pd.Series(scores, name='score', index=dataset['y'].index)
            df = pd.concat([dataset['y'], scores, self.aux_data[dataset_name]], axis=1)
            path = f"{self.score_dir}/{dataset_name}_scores"
            df.to_csv(path, index=False)
            print(f"Saved {dataset_name} scores to {path}")

      
