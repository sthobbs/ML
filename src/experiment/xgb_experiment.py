from experiment.experiment import Experiment, ConfigError
from typing import Optional
import xgboost as xgb


class XGBExperiment(Experiment):
    """
    Class for training and evaluating XGBoost models.

    Author:
       Steve Hobbs
       github.com/sthobbs
    """

    def __init__(self, config_path: str) -> None:
        """
        Initializes XBGBoost experiment from a config file.

        Parameters
        ----------
            config_path : str
                Path to yaml config file.
        """

        super().__init__(config_path)

        # reorder for dataset_names (since last dataset is used for early stopping if enabled)
        if self.hyperparameters.get("early_stopping_rounds") or 'early_stopping_round' in self.tuning_parameters:
            all_names = set(self.data_file_patterns)
            main_names = {'train', 'test', 'validation'}
            other_names = sorted(all_names.difference(main_names))
            self.dataset_names = ['train'] + other_names
            self.dataset_names.extend([n for n in ['test', 'validation'] if n in all_names])

    def validate_config(self) -> None:
        """Ensure that the config file is valid."""

        super().validate_config()
        valid_model_types = {'XGBClassifier', 'XGBRegressor'}
        if self.config["model_type"] not in valid_model_types:
            raise ConfigError(f"model_type must be in {valid_model_types}")
        if not self.config["supervised"]:
            raise ConfigError("supervised must be True")

    def load_model(self,
                   model_obj: Optional[xgb.XGBModel] = None,
                   path: Optional[str] = None) -> None:
        """
        Loads a model object from a parameter or file path, or instantiates a
        new model object.

        Parameters
        ----------
            model_obj : str, optional
                XGBoost model.
            path : str, optional
                File path to XGBoost model object (default is None).
        """

        super().load_model(model_obj, path)
        assert isinstance(self.model, xgb.XGBModel), "self.model must be an XGBoost model."

    def train(self, **fit_params) -> None:
        """
        Tune hyperparameters, then train a final XGBoost model with
        the tuned hyperparmeters.

        Parameters
        ----------
            **fit_params : optional
                Keyword arguments to pass to the model's .fit() method.
        """

        fit_params['verbose'] = self.verbose
        fit_params['eval_set'] = [(self.data[n]['X'], self.data[n]['y']) for n in self.dataset_names]
        super().train(**fit_params)

    def save_model(self) -> None:
        """Save the XGBoost model object as both .pkl and .bin files."""

        # save pickle version
        super().save_model()
        # save json and binary binary versions
        self.model.save_model(self.model_dir/'model.json')
        self.model.save_model(self.model_dir/'model.ubj')

    def evaluate(self, increment: float = 0.01) -> None:
        """
        Evaluate XGboost model and generate performance charts.

        Parameters
        ----------
            increment : float
                Increment to use when generating charts.
        """

        # generate general metrics
        super().evaluate(increment)
        # generate XGBoost metrics
        self.model_eval.xgb_evaluate(self.dataset_names)
