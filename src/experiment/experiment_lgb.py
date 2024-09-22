from experiment.experiment import Experiment, ConfigError
from typing import Optional
import lightgbm as lgb


class LGBExperiment(Experiment):
    """
    Class for training and evaluating LightGBM models.

    Author:
       Steve Hobbs
       github.com/sthobbs
    """

    def __init__(self, config_path: str) -> None:
        """
        Initializes LightGBM experiment from a config file.

        Parameters
        ----------
            config_path : str
                Path to yaml config file.
        """

        super().__init__(config_path)

    def validate_config(self) -> None:
        """Ensure that the config file is valid."""

        super().validate_config()
        valid_model_types = {'LGBMClassifier', 'LGBMRegressor'}
        if self.config["model_type"] not in valid_model_types:
            raise ConfigError(f"model_type must be in {valid_model_types}")
        if not self.config["supervised"]:
            raise ConfigError("supervised must be True")

    def load_model(self,
                   model_obj: Optional[lgb.LGBMModel] = None,
                   path: Optional[str] = None) -> None:
        """
        Loads a model object from a parameter or file path, or instantiates a
        new model object.

        Parameters
        ----------
            model_obj : str, optional
                LightGBM model.
            path : str, optional
                File path to LightGBM model object (default is None).
        """

        super().load_model(model_obj, path)
        assert isinstance(self.model, lgb.LGBMModel), "self.model must be an LightGBM model."

    def train(self, **fit_params) -> None:
        """
        Tune hyperparameters, then train a final XGBoost model with
        the tuned hyperparmeters.

        Parameters
        ----------
            **fit_params : optional
                Keyword arguments to pass to the model's .fit() method.
        """

        # currently LGBM only supports early stopping with all datasets in eval_set,
        # so we're only going to use one dataset if early stopped is enabled.
        # https://github.com/microsoft/LightGBM/issues/6360
        # in the future, once they allow it use one of multiple datasets, we can change this
        if 'early_stopping_round' in self.hyperparameters or 'early_stopping_round' in self.tuning_parameters:
            if 'validation' in self.dataset_names:
                eval_set = [(self.data['validation']['X'], self.data['validation']['y'])]
                dataset_name = 'validation'
            elif 'test' in self.dataset_names:
                eval_set = [(self.data['test']['X'], self.data['test']['y'])]
                dataset_name = 'test'
            else:
                for name in self.dataset_names:
                    if name == 'train':
                        continue
                    eval_set = [(self.data[name]['X'], self.data[name]['y'])]
                    dataset_name = name
                    break
            self.logger.info(f"Early stopping enabled. Using only one dataset, {dataset_name}, for evaluation.")
            fit_params['eval_names'] = dataset_name
        else:
            eval_set = [(self.data[n]['X'], self.data[n]['y']) for n in self.dataset_names]
            fit_params['eval_names'] = self.dataset_names
        fit_params['eval_set'] = eval_set
        fit_params['eval_metric'] = self.eval_metric
        super().train(**fit_params)

    def save_model(self) -> None:
        """Save the XGBoost model object as both .pkl and .bin files."""

        # save pickle version
        super().save_model()
        # save json and binary binary versions
        self.model.booster_.save_model(self.model_dir/'model.txt')

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
        # generate LightGBM metrics
        self.model_eval.lgb_evaluate()
