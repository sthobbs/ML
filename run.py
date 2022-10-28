from xgb_experiment import XGBExperiment


if __name__ == "__main__":
    config_path = "C:/Users/hobbs/Documents/Programming/ML/config.yaml"
    exp = XGBExperiment(config_path)
    exp.run()