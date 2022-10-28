# ML Experiments
This repo packages up code for reproducible machine learning experiments, including:
- Hyperparameter tuning (using grid search, random search, tpe, or atpe) (with optional cross validation)
- Model evaluation (PR curve, ROC curve, score distribution, KS-statistic, various tables, etc.)
- Model explainability (shapely, PSI/CSI, VIF, WoE/IV, permutation feature importance, correlation matrix)
- XGBoost-related objects (plots of n_estimators vs performance metrics, tree-based feature importance)
- etc.

Example code to run an XGBoost experiment:
```
from xgb_experiment import XGBExperiment

config_path = "C:/Users/hobbs/Documents/Programming/ML/config.yaml"
exp = XGBExperiment(config_path)
exp.run()
```
See config.yaml for details on how to configure the experiment.