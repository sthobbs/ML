from experiment.experiment_lgb import LGBExperiment
from experiment.experiment_xgb import XGBExperiment
# from experiment.experiment import Experiment


if __name__ == "__main__":
    config_path = "./examples/configs/xgb_config.yaml"
    exp = XGBExperiment(config_path)
    exp.run()

    # config_path = "./examples/configs/lgb_config.yaml"
    # exp = LGBExperiment(config_path)
    # exp.run()
