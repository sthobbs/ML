from experiment.experiment import Experiment, ConfigError
from typing import Optional
import xgboost as xgb


class XGBExperiment(Experiment):
    """
    Class for training and evaluating XGBoost models

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

        super().__init__(config_path)

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
                scikit-learn model object with a .predict_proba() method
                (default is None)
            path : str, optional
                file path to scikit-learn model object with a .predict_proba()
                method (default is None)
        """

        super().load_model(model_obj, path)
        assert isinstance(self.model, xgb.XGBModel), "self.model must be an XGBoost model"

    def train(self, **kwargs) -> None:
        """
        tune hyperparameters, then train a final XGBoost model with
        the tuned hyperparmeters.
        """

        kwargs['verbose'] = self.verbose
        kwargs['eval_set'] = [(self.data[n]['X'], self.data[n]['y']) for n in self.dataset_names]
        super().train(**kwargs)

    def save_model(self) -> None:
        """Save the XGBoost model object as both .pkl and .bin files."""

        # save pickle version
        super().save_model()
        # save json and binary binary versions
        self.model.save_model(self.model_dir/'model.json')
        self.model.save_model(self.model_dir/'model.ubj')

    def evaluate(self, increment: float = 0.01) -> None:
        """Evaluate XGboost model and generate performance charts."""

        # generate general metrics
        super().evaluate(increment)
        # generate XGBoost metrics
        self.model_eval.xgb_evaluate(self.dataset_names)
