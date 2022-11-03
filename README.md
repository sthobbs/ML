# ML Experiments
![Tests](https://github.com/sthobbs/ML/actions/workflows/tests.yaml/badge.svg)

This repo packages up code for reproducible machine learning experiments, including:
- Hyperparameter tuning (using grid search, random search, tpe, or atpe) (with optional cross validation)
- Model evaluation (PR curve, ROC curve, score distribution, KS-statistic, various tables, etc.)
- Model explainability (shapely, PSI/CSI, VIF, WoE/IV, permutation feature importance, correlation matrix)
- Model calibration
- XGBoost-related objects (plots of n_estimators vs performance metrics, tree-based feature importance)

Example code to run an XGBoost experiment:
```
from xgb_experiment import XGBExperiment

config_path = ".../config.yaml"
exp = XGBExperiment(config_path)
exp.run()
```
See config.yaml for details on how to configure an experiment.