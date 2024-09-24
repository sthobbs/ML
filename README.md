# ML Experiments
![Tests](https://github.com/sthobbs/ML/actions/workflows/tests.yaml/badge.svg)

This repo enables reproducible machine learning experiments, including:
- EDA (distribution plots, summary statistics, correlations, feature vs time plots, etc.)
- Hyperparameter tuning (using grid search, random search, tpe, or atpe) (with optional cross validation)
- Model evaluation (PR curve, ROC curve, score distribution, KS-statistic, performance tables, etc.)
- Model explainability (shapely, PSI/CSI, VIF, WoE/IV, permutation feature importance)
- Model calibration (using isotonic or logistic regression)
- XGBoost and LightGBM metrics (tree-based feature importance, performance during training)

See example output in ./examples

Example code to run an XGBoost experiment:
```
from experiment_xgb import XGBExperiment

config_path = "./examples/configs/xgb_config.yaml"
exp = XGBExperiment(config_path)
exp.run()
```

---------------
#### Initial setup
For the initial setup, run the following commands to create a new virtual environment, install dependencies, and install the local package:

conda create --name exp python=3.12  
conda activate exp  
pip install -r requirements_dev.txt  
pip install -r requirements.txt  
pip install -e .  # with your current directory is the one with pyproject.toml
