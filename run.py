from experiment.xgb_experiment import XGBExperiment


if __name__ == "__main__":
    config_path = "./config.yaml"
    exp = XGBExperiment(config_path)
    exp.run()
