from utils import non_empty_file
import pytest


# 1) check that feature importance tables are generated
@pytest.mark.parametrize("file_name", [
    "permutation_importance_other.csv",
    "permutation_importance_test.csv",
    "permutation_importance_train.csv",
    "permutation_importance_validation.csv",
    "xgb_feature_importance.csv",
])
def test_permutation_importance(file_name, xgb_exp_explained):
    path = xgb_exp_explained.explain_dir / "feature_importance" / file_name
    assert non_empty_file(path)


# 2) check that (a sample of) shap plots are generated
# TODO: fix all the warnings generated from shap (seems to be from underlying cython code)
@pytest.mark.parametrize("file_name", [
    "other/partial_dependence_plots/V1.png",
    "test/scatter_plots/V3.png",
    "train/abs_max_bar_plot.png",
    "validation/abs_mean_bar_plot.png",
    "test/beeswarm_plot.png",
])
def test_explain_shap(file_name, xgb_exp_explained):
    path = xgb_exp_explained.explain_dir / "shap" / file_name
    assert non_empty_file(path)


# 3) check that psi and csi tables are generated
@pytest.mark.parametrize("file_name", [
    "psi.csv",
    "csi_long.csv",
    "csi_wide.csv",
])
def test_explain_psi_csi(file_name, xgb_exp_explained):
    path = xgb_exp_explained.explain_dir / file_name
    assert non_empty_file(path)


# 4) check that psi and csi tables are generated with quantile bins
@pytest.mark.parametrize("file_name", [
    "psi.csv",
    "csi_long.csv",
    "csi_wide.csv",
])
def test_explain_psi_csi_quantiles(file_name, xgb_exp_explained_quantiles):
    path = xgb_exp_explained_quantiles.explain_dir / file_name
    assert non_empty_file(path)


# 5) check that vif tables are generated
@pytest.mark.parametrize("file_name", [
    "vif_other.csv",
    "vif_test.csv",
    "vif_train.csv",
    "vif_validation.csv",
])
def test_explain_vif(file_name, xgb_exp_explained):
    path = xgb_exp_explained.explain_dir / "vif" / file_name
    assert non_empty_file(path)


# 6) check that woe/iv tables are generated
@pytest.mark.parametrize("file_name", [
    "woe_other.csv",
    "woe_test.csv",
    "woe_train.csv",
    "woe_validation.csv",
    "iv_other.csv",
    "iv_test.csv",
    "iv_train.csv",
    "iv_validation.csv",
])
def test_explain_woe_iv(file_name, xgb_exp_explained):
    path = xgb_exp_explained.explain_dir / "woe_iv" / file_name
    assert non_empty_file(path)


# 7) check that woe/iv tables are generated with quantile bins
@pytest.mark.parametrize("file_name", [
    "woe_other.csv",
    "woe_test.csv",
    "woe_train.csv",
    "woe_validation.csv",
    "iv_other.csv",
    "iv_test.csv",
    "iv_train.csv",
    "iv_validation.csv",
])
def test_explain_woe_iv_quantiles(file_name, xgb_exp_explained_quantiles):
    path = xgb_exp_explained_quantiles.explain_dir / "woe_iv" / file_name
    assert non_empty_file(path)


# 8) check that correlation tables and plots are generated
@pytest.mark.parametrize("file_name", [
    "corr_other.csv",
    "corr_test.csv",
    "corr_train.csv",
    "corr_validation.csv",
    "corr_other_long.csv",
    "corr_test_long.csv",
    "corr_train_long.csv",
    "corr_validation_long.csv",
    "heatmap_other.png",
    "heatmap_test.png",
    "heatmap_train.png",
    "heatmap_validation.png",
])
def test_explain_correlation(file_name, xgb_exp_explained):
    path = xgb_exp_explained.explain_dir / "correlation" / file_name
    assert non_empty_file(path)
