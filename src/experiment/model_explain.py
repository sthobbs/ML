from pathlib import Path
import logging
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator
import shap
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import xgboost as xgb
from typing import Optional, List, Union, Tuple
import numpy.typing as npt
from datetime import timedelta
matplotlib.use('agg')


class ModelExplain():
    """
    Generates model explainability results.

    Author:
       Steve Hobbs
       github.com/sthobbs
    """

    def __init__(self,
                 model: Optional[BaseEstimator] = None,
                 datasets: Optional[List[Tuple[
                    Union[pd.core.frame.DataFrame, pd.core.series.Series],
                    Union[pd.core.frame.DataFrame, pd.core.series.Series],
                    str]]] = None,
                 aux_data: Optional[List[Union[pd.core.frame.DataFrame, pd.core.series.Series]]] = None,
                 output_dir: Optional[Union[str, Path]] = None,
                 binary_classification: Optional[bool] = None,
                 logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the ModelExplain class.

        Parameters
        ----------
            model :
                Scikit-learn classifier with a .predict_proba() method.
            datasets :
                List of (X, y, dataset_name) triples.
                e.g. [(X_train, y_train, 'Train'), (X_val, y_val, 'Validation'), (X_test, y_test, 'Test')]
                All datasets, X, should have the same columns.
            aux_data :
                Auxiliary data fields that aren't model features (e.g. timestamps).
                List index corresponds to datasets index.
            output_dir : str, optional
                String path to folder where output will be written.
            binary_classification : bool, optional
                If the model is a binary classification model.
            logger : logging.Logger, optional
                Logger.
        """

        # check for valid input
        assert binary_classification is None or isinstance(binary_classification, bool), \
            "`binary_classification` must be a boolean or None."

        if model is not None:
            self.model = model
        if datasets is None:
            datasets = []
        self.datasets = datasets
        if aux_data is None:
            aux_data = []
        self.aux_data = aux_data

        # Determine if binary classification based on datasets
        if binary_classification is None:
            self.binary_classification = False
            if len(datasets) > 0:
                self.binary_classification = all(datasets[i][1].nunique() == 2 for i in range(len(datasets)))
        else:
            self.binary_classification = binary_classification

        # Make directories
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set plot context
        self.plot_context = 'seaborn-v0_8-darkgrid'

        # Set up logger
        if logger is None:
            # create logger
            logger = logging.getLogger(__name__).getChild(self.__class__.__name__).getChild(str(id(self)))
            logger.setLevel(logging.INFO)
            # create formatter
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            # create and add handlers for console output
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        self.logger = logger

    def gen_permutation_importance(self,
                                   n_repeats: int = 10,
                                   metrics: str = 'neg_log_loss',
                                   seed: int = 1) -> None:
        """
        Generate permutation feature importance tables.

        Parameters
        ----------
            n_repeats : int, optional
                Number of times to permute each feature (default is 10).
            metrics : str or list of str, optional
                Metrics used in permutation feature importance calculations (default is 'neg_log_loss').
                e.g.: 'roc_auc', 'average_precision', 'neg_log_loss', 'r2', etc.
                See https://scikit-learn.org/stable/modules/model_evaluation.html for complete list.
            seed : int, optional
                Random seed (default is 1).
        """

        self.logger.info("----- Generating Permutation Feature Importances -----")

        # make output directory
        importance_dir = self.output_dir / "feature_importance"
        importance_dir.mkdir(parents=True, exist_ok=True)

        # generate permutation feature importance for each metric on each dataset
        for X, y, dataset_name in self.datasets:
            r = permutation_importance(self.model, X, y, n_repeats=n_repeats,
                                       random_state=seed, scoring=metrics)
            imps = []
            for m in metrics:
                # get means and standard deviations of feature importance
                means = pd.Series(r[m]['importances_mean'], name=f"{m}_mean")
                stds = pd.Series(r[m]['importances_std'], name=f"{m}_std")
                imps.extend([means, stds])
            df = pd.concat(imps, axis=1)  # dataframe of importance means and stds
            df.index = X.columns
            df.sort_values(f"{metrics[0]}_mean", ascending=False, inplace=True)
            df.to_csv(f'{importance_dir}/permutation_importance_{dataset_name}.csv')
            self.logger.info(f"Generated permutation importance ({dataset_name} data)")

    def plot_shap(self, shap_sample: Optional[int] = None) -> None:
        """
        Generate model explanitory charts involving shap values.

        Parameters
        ----------
            shap_sample : int, optional
                Number of rows to sample from the dataset (default is None).
        """

        self.logger.info("----- Generating Shap Charts -----")

        assert self.model is not None, "self.model can't be None to run plot_shap()."
        plt.close('all')

        # Generate Shap Charts
        savefig_kwargs = {'bbox_inches': 'tight', 'pad_inches': 0.2}
        def predict(x): return self.model.predict_proba(x)[:, 1]
        for X, y, dataset_name in self.datasets:
            # get sample of dataset (gets all data if self.shap_sample is None)
            dataset = X.iloc[:shap_sample]
            if len(dataset) > 500000:
                msg = (f"Shap will be slow on {len(dataset)} rows, consider using"
                       " shap_sample in the config to sample fewer rows")
                self.logger.warning(msg)

            # Generate partial dependence plots
            plot_dir = self.output_dir/"shap"/dataset_name/"partial_dependence_plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            for feature in tqdm(dataset.columns):
                fig, ax = shap.partial_dependence_plot(
                    feature, predict, dataset, model_expected_value=True,
                    feature_expected_value=True, show=False, ice=False)
                fig.savefig(f"{plot_dir}/{feature}.png", **savefig_kwargs)
                plt.close()
            self.logger.info(f'Plotted partial dependence plots ({dataset_name} data)')

            # Generate scatter plots (coloured by feature with strongest interaction)
            explainer = shap.Explainer(self.model, dataset)
            shap_values = explainer(dataset)
            plot_dir = self.output_dir/"shap"/dataset_name/"scatter_plots"
            plot_dir.mkdir(exist_ok=True)
            for feature in tqdm(dataset.columns):
                shap.plots.scatter(shap_values[:, feature], alpha=0.3, color=shap_values, show=False)
                plt.savefig(f"{plot_dir}/{feature}.png", **savefig_kwargs)
                plt.close()
            self.logger.info(f'Plotted scatter plots ({dataset_name} data)')

            # Generate beeswarm plot
            shap.plots.beeswarm(shap_values, alpha=0.1, max_display=1000, show=False)
            path = self.output_dir/"shap"/dataset_name/"beeswarm_plot.png"
            plt.savefig(path, **savefig_kwargs)
            plt.close()
            self.logger.info(f'Plotted beeswarm plot ({dataset_name} data)')

            # Generate bar plots
            shap.plots.bar(shap_values, max_display=1000, show=False)
            path = self.output_dir/"shap"/dataset_name/"abs_mean_bar_plot.png"
            plt.savefig(path, **savefig_kwargs)
            plt.close()
            shap.plots.bar(shap_values.abs.max(0), max_display=1000, show=False)
            path = self.output_dir/"shap"/dataset_name/"abs_max_bar_plot.png"
            plt.savefig(path, **savefig_kwargs)
            plt.close()
            self.logger.info(f'Plotted bar plots ({dataset_name} data)')
            # TODO?: make alpha and max_display config variables

    def plot_feature_distribution(self, exclude_outliers: bool = False) -> None:
        """
        Generate histogram of feature values.

        Parameters
        ----------
            exclude_outliers : bool
                Whether to exclude outliers (default is False).
        """

        self.logger.info("----- Generating Feature Distribution Charts -----")

        plt.close('all')
        savefig_kwargs = {'bbox_inches': 'tight', 'pad_inches': 0.5}
        with plt.style.context(self.plot_context):
            for X, y, dataset_name in self.datasets:
                plot_dir = self.output_dir/"distribution"/dataset_name
                plot_dir.mkdir(parents=True, exist_ok=True)  # make directory for distribution plots
                for feature in tqdm(X.columns):
                    bins = self._gen_histogram_bins(X[feature], y, exclude_outliers)  # get bins for histogram
                    plt.figure()
                    common_kwargs = {"stat": "probability", "bins": bins, "kde": False}
                    if self.binary_classification:  # plot distribution for each class
                        sns.histplot(X[feature][y == 0], label="Class: 0", **common_kwargs, color="dodgerblue", alpha=0.5)
                        sns.histplot(X[feature][y == 1], label="Class: 1", **common_kwargs, color="orange", alpha=0.4)
                    else:
                        sns.histplot(X[feature], **common_kwargs)
                    plt.title(f"{feature} Distribution ({dataset_name} data)")
                    plt.legend(borderaxespad=0, frameon=True)
                    plt.savefig(f'{plot_dir}/{feature}.png', **savefig_kwargs)
                    plt.close()
                self.logger.info(f'Plotted distribution plots ({dataset_name} data)')

    def _gen_histogram_bins(self,
                            X_feature,
                            y_true,
                            exclude_outliers: bool = False) -> npt.NDArray[np.float64]:
        """
        Generate bins for histogram.

        Parameters
        ----------
            X_feature : pd.Series
                Feature values.
            y_true : pd.Series
                Target values.
            exclude_outliers : bool
                Whether to exclude extreme outliers from histogram (default is False).
        """

        # exclude extreme outliers
        if exclude_outliers:
            iqr_multiplier = 5  # IQR multiplier for outliers to be excluded (i.e. 5x the IQR)
            not_outliers: pd.Series
            if self.binary_classification:  # outliers should be outliers of both classes to be excluded
                X_0 = X_feature[y_true == 0]
                X_1 = X_feature[y_true == 1]
                iqr_0 = X_0.quantile(0.75) - X_0.quantile(0.25)  # interquartile range
                iqr_1 = X_1.quantile(0.75) - X_1.quantile(0.25)
                non_outlier_ub = max(X_0.median() + iqr_multiplier * iqr_0, X_1.median() + iqr_multiplier * iqr_1)
                non_outlier_lb = min(X_0.median() - iqr_multiplier * iqr_0, X_1.median() - iqr_multiplier * iqr_1)
                not_outliers = X_feature.between(non_outlier_lb, non_outlier_ub)
            else:
                iqr = X_feature.quantile(0.75) - X_feature.quantile(0.25)
                not_outliers = ((X_feature - X_feature.median()) / iqr).abs() <= iqr_multiplier
            X_feature = X_feature[not_outliers]
            y_true = y_true[not_outliers]

        # 1. look for integer range in [-1, 50] that covers at least 99.5% of data (for integer-valued features)
        for lb in range(-1, 51):  # find smallest integer >= -1 that has any values
            if (X_feature == lb).any():
                break
        if self.binary_classification:  # if binary classification, require at least 99.5% of data from each class
            X_0 = X_feature[y_true == 0]
            X_1 = X_feature[y_true == 1]
            total_0 = X_0.shape[0]
            total_1 = X_1.shape[0]
            cnt_0 = cnt_1 = 0
            for ub in range(lb, 51):  # find smallest integer >= lb that has at least 99.5% of data from each class
                cnt_0 += (X_0 == ub).sum()
                cnt_1 += (X_1 == ub).sum()
                if (cnt_0 / total_0 >= 0.995) and (cnt_1 / total_1 >= 0.995):
                    bins = np.arange(lb, ub + 2) - 0.5
                    return bins
        else:  # if not binary classification, require at least 99.5% of all data
            total = X_feature.sum()
            cnt = 0
            for ub in range(lb, 51):  # find smallest integer >= lb that has at least 99.5% of data
                cnt += (X_feature == lb).sum()
                if cnt / total >= 0.995:
                    bins = np.arange(lb, ub + 2) - 0.5
                    return bins

        # 2. use numpy auto select bins
        bins = np.histogram_bin_edges(X_feature, bins='auto')

        # 3. if too many bins, use 50 equal-sized bins
        if len(bins) > 51:
            bins = np.histogram_bin_edges(X_feature, bins=50)

        # 4. if more bins than unique values, then make one bin for each unique value
        unique_vals = X_feature.unique()
        if len(bins) > len(unique_vals) + 1:
            unique_vals.sort()
            mid_points = (unique_vals[:-1] + unique_vals[1:]) / 2
            left = 2 * unique_vals[0] - mid_points[0]
            right = 2 * unique_vals[-1] - mid_points[-1]
            bins = np.concatenate([[left], mid_points, [right]])

        return bins

    def plot_feature_vs_time(self,
                             dt_field: str = 'timestamp',
                             dt_format: str = 'mixed',
                             n_bins: int = 15) -> None:
        """
        Plot features over time using matplotlib.

        Parameters
        ----------
            dt_field: str
                Name of the datetime field in the dataset (default: 'timestamp').
            dt_format: str
                Format of the datetime field in the dataset, example input includes
                "%Y-%m-%d %H:%M:%S" or "%Y-%m-%d %H:%M:%S.%f" (default: 'mixed').
            n_bins: int
                Number of bins to use for equal time intervals (default: 15).
        """

        self.logger.info("----- Generating Feature vs Time Charts -----")

        assert dt_field in self.datasets[0][0].columns \
            or dt_field in self.aux_data[0].columns, \
            f"`dt_field` = {dt_field} should be a column in the dataset or aux_data."
        assert "datetime_bin" not in self.datasets[0][0].columns \
            and "datetime_bin" not in self.aux_data[0].columns, \
            "'datetime_bin' should not be a column in the dataset or aux_data, since this name will be used for binning"

        plt.close('all')
        savefig_kwargs = {'bbox_inches': 'tight', 'pad_inches': 0.2}
        with plt.style.context(self.plot_context):
            for i in range(len(self.datasets)):
                X, y, dataset_name = self.datasets[i]
                aux_data = self.aux_data[i]
                aux_fields = [i for i in list(aux_data.columns) if i not in X.columns]
                aux_data = aux_data[aux_fields]
                # make directory for distribution plots
                plot_dir = self.output_dir/"feature_vs_time"/dataset_name
                plot_dir.mkdir(parents=True, exist_ok=True)  # make directory for distribution plots

                # split data into bins based on equal time intervals
                df: pd.DataFrame
                label: str | None
                if self.binary_classification:
                    df = pd.concat([y, X, aux_data], axis=1)
                    label = df.columns[0]
                else:
                    df = pd.concat([X, aux_data], axis=1)
                    label = None
                df[dt_field] = pd.to_datetime(df[dt_field], format=dt_format, dayfirst=False)  # convert to datetime
                max_dt = df[dt_field].max()
                min_dt = df[dt_field].min()
                total_seconds = (max_dt - min_dt).total_seconds()
                seconds_per_bin = total_seconds / n_bins  # seconds in each bin
                df = df.sort_values(dt_field, ascending=True)
                lb = 0
                curr = min_dt  # current datetime lower bound
                while curr < max_dt:
                    midpoint_dt = curr + timedelta(seconds=seconds_per_bin / 2)
                    curr += timedelta(seconds=seconds_per_bin)
                    ub = df[dt_field].searchsorted(curr)  # binary search for upper bound (index of first value >= curr)
                    df.loc[df.iloc[lb:ub-1].index, "datetime_bin"] = midpoint_dt  # use midpoint as bin label
                    lb = ub

                # generate plot for each feature
                for feature in tqdm(X.columns):
                    plt.figure()
                    sns.lineplot(x="datetime_bin", y=feature, data=df, hue=label)
                    plt.xlabel(dt_field)
                    plt.xticks(rotation=45)
                    plt.title(f"{feature} Over Time ({dataset_name} data)")
                    plt.legend(borderaxespad=0.5, frameon=True, title=label)
                    plt.savefig(f'{plot_dir}/{feature}.png', **savefig_kwargs)
                    plt.close()
                self.logger.info(f'Plotted feature vs time plots ({dataset_name} data)')

    def gen_psi(self, bin_type: str = 'fixed', n_bins: int = 10) -> None:
        """
        Generate Population Stability Index (PSI) values between all pairs of datasets.

               PSI < 0.1 => no significant population change
        0.1 <= PSI < 0.2 => moderate population change
        0.2 <= PSI       => significant population change

        Note: PSI is symmetric provided the bins are the same, which they are when bin_type='fixed'

        Parameters
        ----------
            bin_type : str, optional
                The method for choosing bins, either 'fixed' or 'quantiles' (default is 'fixed').
            n_bins : int, optional
                The number of bins used to compute psi (default is 10).
        """

        self.logger.info("----- Generating PSI -----")

        # check for valid input
        assert bin_type in {'fixed', 'quantiles'}, "`bin_type` must be in {'fixed', 'quantiles'}."
        assert self.model is not None, "self.model can't be None to run gen_psi()."

        # make output directory
        psi_csi_dir = self.output_dir / 'psi_csi'
        psi_csi_dir.mkdir(parents=True, exist_ok=True)

        # intialize output list
        psi_list = []

        # get dictionary of scores for all datasets
        scores_dict = {}
        for X, _, dataset_name in self.datasets:
            scores = self.model.predict_proba(X)[:, 1]
            scores.sort()
            scores_dict[dataset_name] = scores

        # compute psi for each pair of datasets
        dataset_names = [dataset_name for _, _, dataset_name in self.datasets]
        for i, dataset_name1 in enumerate(dataset_names):
            for j in range(i+1, len(dataset_names)):
                dataset_name2 = dataset_names[j]
                scores1 = scores_dict[dataset_name1]
                scores2 = scores_dict[dataset_name2]
                psi_val = self._psi_compare(scores1, scores2, bin_type=bin_type, n_bins=n_bins)
                row = {
                    'dataset1': dataset_name1,
                    'dataset2': dataset_name2,
                    'psi': psi_val
                }
                psi_list.append(row)

        # convert output to dataframe
        psi_df = pd.DataFrame.from_records(psi_list)
        psi_df = psi_df.reindex(columns=['dataset1', 'dataset2', 'psi'])  # reorder columns

        # save output to csv
        psi_df.to_csv(psi_csi_dir/'psi.csv', index=False)

    def _psi_compare(self,
                     scores1: Union[npt.NDArray[np.float64], pd.core.series.Series],
                     scores2: Union[npt.NDArray[np.float64], pd.core.series.Series],
                     bin_type: str = 'fixed',
                     n_bins: int = 10) -> float:
        """
        Compute Population Stability Index (PSI) between two datasets.

        Parameters
        ----------
            scores1 : numpy.ndarray or pandas.core.series.Series
                Scores for one of the datasets.
            scores2 : numpy.ndarray or pandas.core.series.Series
                Scores for the other dataset.
            bin_type : str, optional
                The method for choosing bins, either 'fixed' or 'quantiles' (default is 'fixed').
            n_bins : int, optional
                The number of bins used to compute psi (default is 10).
        """

        # get bins
        min_val = min(min(scores1), min(scores2))  # TODO? could bring this up a function for efficiency
        max_val = max(min(scores1), max(scores2))
        if bin_type == 'fixed':
            bins = [min_val + (max_val - min_val) * i / n_bins for i in range(n_bins + 1)]
        elif bin_type == 'quantiles':
            bins = pd.qcut(scores1, q=n_bins, retbins=True, duplicates='drop')[1]
            n_bins = len(bins) - 1  # some bins could be dropped due to duplication
        eps = 1e-6
        bins[0] -= -eps
        bins[-1] += eps

        # group data into bins and get percentage rates
        scores1_bins = pd.cut(scores1, bins=bins, labels=range(n_bins))
        scores2_bins = pd.cut(scores2, bins=bins, labels=range(n_bins))
        df1 = pd.DataFrame({'score1': scores1, 'bin': scores1_bins})
        df2 = pd.DataFrame({'score2': scores2, 'bin': scores2_bins})
        grp1 = df1.groupby('bin', observed=False).count()['score1']
        grp2 = df2.groupby('bin', observed=False).count()['score2']
        grp1_rate = (grp1 / sum(grp1)).rename('rate1')
        grp2_rate = (grp2 / sum(grp2)).rename('rate2')
        grp_rates = pd.concat([grp1_rate, grp2_rate], axis=1).fillna(0)

        # add a small value when the percent is zero
        grp_rates = grp_rates.map(lambda x: eps if x == 0 else x)

        # calculate psi
        psi_vals = (grp_rates['rate1'] - grp_rates['rate2']) * np.log(grp_rates['rate1'] / grp_rates['rate2'])
        psi: float = psi_vals.mean()
        return psi

    def gen_csi(self, bin_type: str = 'fixed', n_bins: int = 10) -> None:
        """
        Generate Characteristic Stability Index (CSI) values for all features between all pairs of datasets.

        Note: CSI is symmetric provided the bins are the same, which they are when bin_type='fixed'

        Parameters
        ----------
            bin_type : str, optional
                The method for choosing bins, either 'fixed' or 'quantiles' (default is 'fixed').
            n_bins : int, optional
                The number of bins used to compute csi (default is 10).
        """

        self.logger.info("----- Generating CSI -----")

        # check for valid input
        assert bin_type in {'fixed', 'quantiles'}, "`bin_type` must be in {'fixed', 'quantiles'}."

        # make output directory
        psi_csi_dir = self.output_dir / 'psi_csi'
        psi_csi_dir.mkdir(parents=True, exist_ok=True)

        # intialize output list
        csi_list = []

        features = self.datasets[0][0].columns
        for feature in tqdm(features):

            # get dictionary of values for the given feature from all datasets
            vals_dict = {}
            for X, _, dataset_name in self.datasets:
                scores = X[feature]
                scores.sort_values()
                vals_dict[dataset_name] = scores

            # compute csi for each pair of datasets
            dataset_names = [dataset_name for _, _, dataset_name in self.datasets]
            for i, dataset_name1 in enumerate(dataset_names):
                for j in range(i+1, len(dataset_names)):
                    dataset_name2 = dataset_names[j]
                    scores1 = vals_dict[dataset_name1]
                    scores2 = vals_dict[dataset_name2]
                    csi_val = self._psi_compare(scores1, scores2, bin_type=bin_type, n_bins=n_bins)
                    row = {
                        'dataset1': dataset_name1,
                        'dataset2': dataset_name2,
                        'feature': feature,
                        'csi': csi_val
                    }
                    csi_list.append(row)

        # convert output to dataframe
        csi_df = pd.DataFrame.from_records(csi_list)
        csi_df = csi_df.reindex(columns=['dataset1', 'dataset2', 'feature', 'csi'])  # reorder columns

        # save output to csv
        csi_df.sort_values('csi', ascending=False, inplace=True)
        csi_df.to_csv(psi_csi_dir/'csi_long.csv', index=False)

        # convert csi dataframe to wide format
        csi_df['datasets'] = csi_df['dataset1'] + '-' + csi_df['dataset2']
        csi_df = csi_df[['feature', 'datasets', 'csi']]
        csi_df.set_index('feature', inplace=True)
        csi_df = csi_df.pivot(columns='datasets')['csi']

        # reorder to the same order as features, and save to csv
        csi_df.reset_index(inplace=True)
        csi_df['feature'] = pd.Categorical(csi_df['feature'], features)
        csi_df = csi_df.sort_values("feature").set_index("feature")
        csi_df.to_csv(psi_csi_dir/'csi_wide.csv')

    def gen_vif(self) -> None:
        """
        Generate Variance Inflation Factor (VIF) tables for each dataset.

        VIF = 1  => no correlation
        VIF > 10 => high correlation between an independent variable and the others
        """

        self.logger.info("----- Generating VIF -----")

        # make directory for VIF tables
        vif_dir = self.output_dir / "vif"
        vif_dir.mkdir(parents=True, exist_ok=True)

        # calculate VIF values for each feature in each dataset
        features = self.datasets[0][0].columns
        for X, _, dataset_name in tqdm(self.datasets):
            df = pd.DataFrame({'feature': features})
            df["vif"] = [variance_inflation_factor(X.values, i) for i in range(len(features))]
            df.to_csv(vif_dir/f'vif_{dataset_name}.csv', index=False)

    def gen_woe_iv(self, bin_type: str = 'quantiles', n_bins: int = 10) -> None:
        """
        Generate Weight of Evidence and Information Value tables for each dataset.

        Only applicable to binary classification models.

                IV < 0.02 => not useful for prediction
        0.02 <= IV < 0.1  => weak predictive power
        0.1  <= IV < 0.3  => medium predictive power
        0.3  <= IV < 0.5  => strong predictive power
        0.5  <= IV        => suspicious predictive power

        Parameters
        ----------
            bin_type : str, optional
                The method for choosing bins, either 'fixed' or 'quantiles' (default is 'quantiles').
            n_bins : int, optional
                The number of bins used to compute woe and iv (default is 10).
        """

        self.logger.info("----- Generating WOE and IV -----")

        # make output directory
        woe_dir = self.output_dir / 'woe_iv'
        woe_dir.mkdir(parents=True, exist_ok=True)

        for X, y, dataset_name in self.datasets:

            # initialize lists to accumulate data
            woe_df_list = []
            iv_list = []

            # temporarily ignore divide by 0 warnings
            np.seterr(divide='ignore')

            # generate woe and iv for all features
            for feature in tqdm(X.columns):

                # get feature data
                values = X[feature]

                # get bins
                if bin_type == 'fixed':
                    min_val = min(values)
                    max_val = max(values)
                    bins = [min_val + (max_val - min_val) * i / n_bins for i in range(n_bins + 1)]
                elif bin_type == 'quantiles':
                    bins = pd.qcut(values, q=n_bins, retbins=True, duplicates='drop')[1]
                eps = 1e-6
                bins[0] -= -eps  # add buffer to include points right at the edge.
                bins[-1] += eps

                # group data into bins
                value_bins = pd.cut(values, bins=bins)
                df = pd.DataFrame({'label': y, 'bin': value_bins})

                # get counts
                df = df.groupby(['bin'], observed=False).agg({'label': ["sum", len]})['label']
                df['cnt_0'] = df['len'] - df['sum']  # count of 0-label events
                df.rename(columns={'sum': 'cnt_1'}, inplace=True)  # count of 1-label events

                # reformat dataframe
                df.drop(columns=['len'], inplace=True)
                df.reset_index(inplace=True)
                df['feature'] = feature  # add feature
                df = df.reindex(columns=['feature', 'bin', 'cnt_0', 'cnt_1'])  # reorder columns

                # get rates
                df['pct_0'] = df['cnt_0'] / df['cnt_0'].sum()
                df['pct_1'] = df['cnt_1'] / df['cnt_1'].sum()

                # get WOEs and IV
                df['woe'] = np.log(df['pct_1'] / df['pct_0'])
                df['adj_woe'] = np.log(
                    ((df['cnt_1'] + 0.5) / df['cnt_1'].sum()) /
                    ((df['cnt_0'] + 0.5) / df['cnt_0'].sum()))
                iv = (df['woe'] * (df['pct_1'] - df['pct_0'])).sum()
                adj_iv = (df['adj_woe'] * (df['pct_1'] - df['pct_0'])).sum()

                # append to lists
                woe_df_list.append(df)
                iv_list.append({'feature': feature, 'iv': iv, 'adj_iv': adj_iv})

            # turn divide by 0 warnings back on
            np.seterr(divide='warn')

            # save woe and iv tables
            woe_df = pd.concat(woe_df_list)
            woe_df.to_csv(woe_dir/f'woe_{dataset_name}.csv', index=False)

            iv_df = pd.DataFrame.from_records(iv_list)
            iv_df.sort_values('adj_iv', ascending=False, inplace=True)
            iv_df.index.name = 'index'
            iv_df.to_csv(woe_dir/f'iv_{dataset_name}.csv')
            self.logger.info(f"Generated WOE and IV ({dataset_name} data)")

    def gen_corr(self, max_features: int = 100) -> None:
        """
        Generate correlation matrix and heatmap for each dataset.

        Parameters
        ----------
            max_features : int, optional
                The maximum number of features allowed for charts and plots to be generated.
        """

        self.logger.info("----- Generating Correlation -----")

        # check input
        features = self.datasets[0][0].columns
        if len(features) > max_features:
            msg = (f"Not computing correlation matrix since there are {len(features)}"
                   f" features, which more than max_features = {max_features}")
            self.logger.warning(msg)
            return

        # make output directory
        corr_dir = self.output_dir / 'correlation'
        corr_dir.mkdir(parents=True, exist_ok=True)

        # for dataset_name, dataset in self.data.items():
        for X, _, dataset_name in self.datasets:
            corr = X.corr()
            corr_long = pd.melt(corr.reset_index(), id_vars='index')  # unpivot to long format
            # write to csv
            corr.index.name = 'feature'
            corr.to_csv(corr_dir/f'corr_{dataset_name}.csv')
            corr_long.rename(columns={'variable': 'feature_1', 'index': 'feature_2', 'value': 'correlation'}, inplace=True)
            corr_long = corr_long.reindex(columns=['feature_1', 'feature_2', 'correlation'])  # reorder columns
            corr_long = corr_long[corr_long.feature_1 != corr_long.feature_2]
            corr_long.sort_values('correlation', key=abs, ascending=False, inplace=True)
            corr_long.to_csv(corr_dir/f'corr_long_{dataset_name}.csv', index=False)
            # plot heat map
            self.logger.info(f"Generated correlations ({dataset_name} data)")
            self._plot_corr_heatmap(corr, corr_dir/f'heatmap_{dataset_name}.png', data_type='corr')
            self.logger.info(f"Plotted correlation heatmap ({dataset_name} data)")

    def _plot_corr_heatmap(self,
                           data: pd.core.frame.DataFrame,
                           output_path: Union[str, Path],
                           data_type: str = 'corr') -> None:
        """
        Plot correlation heat map.

        Parameters
        ----------
            data : pandas.core.frame.DataFrame
                Input data, either raw data, or correlation matrix.
            output_path : str
                The location where the plot should be written.
                Note that this function assumes that the parent folder exists.
            data_type : str, optional
                The type of input passed into the data argument (default is 'corr').
                - 'features' => the raw feature table is passed in
                - 'corr' => a correlation matrix is passed in

        Credits: https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
        """

        # check inputs
        valid_data_types = {'features', 'corr'}
        assert data_type in valid_data_types, f"Invalid `data_type`: {data_type}"

        # process data into long-format correlation matrix, if required
        if data_type == 'features':
            data = data.corr()
        features = data.columns
        data.index.name = 'index'
        data = pd.melt(data.reset_index(), id_vars='index')  # unpivot to long format
        data = data.reindex(columns=['variable', 'index', 'value'])  # reorder columns
        data.columns = ['feature_1', 'feature_2', 'correlation']

        # set up colours
        n_colors = 256  # Use 256 colors for the diverging color palette
        palette = sns.diverging_palette(20, 220, n=n_colors)  # Create the palette
        color_min, color_max = [-1, 1]  # Range of values that will be mapped to the palette, i.e. min and max possible correlation

        def value_to_color(val):
            val_position = float((val - color_min)) / (color_max - color_min)  # position of value in the input range, relative to the length of the input range
            ind = int(val_position * (n_colors - 1))  # target index in the color palette
            return palette[ind]

        # parameterize sizes of objects in the image
        size_factor = 1
        plot_size = len(features) * size_factor
        figsize = (plot_size * 15 / 14, plot_size)  # multiple by 15/14 so the left main plot is square
        font_size = 20 * size_factor
        square_size_scale = (40 * size_factor) ** 2
        size = data['correlation'].abs()  # size of squares is dependend on absolute correlation

        # map feature to integer coordinates
        feat_to_num = {feature: i for i, feature in enumerate(features)}

        with plt.style.context(self.plot_context):

            # create figure
            plt.figure(figsize=figsize, dpi=50)
            plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1)  # Setup a 1x15 grid

            # Use the leftmost 14 columns of the grid for the main plot
            ax = plt.subplot(plot_grid[:, :-1])

            # make main plot
            ax.scatter(
                x=data['feature_1'].map(feat_to_num),  # Use mapping for feature 1
                y=data['feature_2'].map(feat_to_num),  # Use mapping for feature 2
                s=size * square_size_scale,  # Vector of square sizes, proportional to size parameter
                c=data['correlation'].apply(value_to_color),  # Vector of square color values, mapped to color palette
                marker='s'  # Use square as scatterplot marker
            )

            # Show column labels on the axes
            ax.set_xticks([feat_to_num[v] + 0.3 for v in features])  # add major ticks for the labels
            ax.set_yticks([feat_to_num[v] for v in features])
            ax.set_xticklabels(features, rotation=45, horizontalalignment='right', fontsize=font_size)  # add labels
            ax.set_yticklabels(features, fontsize=font_size)
            ax.grid(False, 'major')  # hide major grid lines
            ax.grid(True, 'minor')
            ax.set_xticks([feat_to_num[v] + 0.5 for v in features], minor=True)  # add minor ticks for grid lines
            ax.set_yticks([feat_to_num[v] + 0.5 for v in features], minor=True)

            # set axis limits
            ax.set_xlim((-0.5, len(features) - 0.5))
            ax.set_ylim((-0.5, len(features) - 0.5))

            # hide all ticks
            plt.tick_params(axis='both', which='both', bottom=False, left=False)

            # Add color legend on the right side of the plot
            ax = plt.subplot(plot_grid[:, -1])  # Use the rightmost column of the plot

            col_x = [0] * len(palette)  # Fixed x coordinate for the bars
            bar_y = np.linspace(color_min, color_max, n_colors)  # y coordinates for each of the n_colors bars

            bar_height = bar_y[1] - bar_y[0]
            ax.barh(
                y=bar_y,
                width=[5]*len(palette),  # Make bars 5 units wide
                left=col_x,  # Make bars start at 0
                height=bar_height,
                color=palette,
                linewidth=0
            )
            ax.set_xlim(1, 2)  # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
            ax.grid(False)  # Hide grid
            ax.set_facecolor('white')  # Make background white
            ax.set_xticks([])  # Remove horizontal ticks
            ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3))  # Show vertical ticks for min, middle and max
            ax.set_yticklabels(["-1", "0", "1"], fontsize=font_size)
            ax.yaxis.tick_right()  # Show vertical ticks on the right

            # save figure
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()

    def gen_summary_statistics(self,
                               quantiles: Optional[List[float]] = None,
                               top_n_value_counts: int = 5) -> None:
        """
        Generate summary statistics.

        Parameters
        ----------
            quantiles: Optional[List[float]]
                List of quantiles to compute.
            top_n_value_counts: int
                Number of top value counts to compute for each column.
        """

        self.logger.info("----- Generating Summary Statistics -----")

        # make output directory
        summary_statistics_dir = self.output_dir / 'summary_statistics'
        summary_statistics_dir.mkdir(parents=True, exist_ok=True)

        # generate summary statistics for each dataset
        dataset_summary_statistics = {}
        for X, y, dataset_name in self.datasets:

            # compute basic summary statistics
            df = pd.concat([y, X], axis=1)
            null_cnt = df.isna().sum()  # number of missing values
            median = df.median()
            iqr = df.quantile(0.75) - df.quantile(0.25)  # interquartile range
            outlier_cnt = (((df - median) / iqr).abs() > 2.22).sum()  # number of outliers (2.22 iqr is approx z-score of 3)
            n = len(df)
            inf = float('inf')
            ninf = float('-inf')
            # import pdb; pdb.set_trace()
            dfs = [
                df.dtypes,               # data types
                df.iloc[0, :],           # sample value
                null_cnt,                # number of missing values
                null_cnt / n,            # missing data rate
                df.nunique(),            # number of unique values
                df.nunique() / n,        # rate of unique values
                (df == 0).sum(),         # number of zeros
                (df == 0).sum() / n,     # rate of zeros
                (df < 0).sum(),          # number of negative values
                (df < 0).sum() / n,      # rate of negative values
                (df == inf).sum(),       # number of infinity values
                (df == inf).sum() / n,   # rate of infinity values
                (df == ninf).sum(),      # number of -infinity values
                (df == ninf).sum() / n,  # rate of -infinity values
                df.mean(),               # mean
                median,                  # median
                df.mode().T[0],          # mode
                df.std(),                # std
                df.min(),                # min
                df.max(),                # max
                iqr,                     # interquartile range
                outlier_cnt,             # number of outliers
                outlier_cnt / n,         # rate of outliers
                df.skew(),               # skewness
                df.kurtosis()            # kurtosis
            ]
            summary_stat_columns = ['Data types', 'Sample value', 'Null count', 'Null rate',
                                    'Unique count', 'Unique rate', 'Zero count', 'Zero rate',
                                    'Negative count', 'Negative rate', 'Infinite count', 'Infinite rate',
                                    'Negative infinite count', 'Negative infinite rate', 'Mean',
                                    'Median', 'Mode', 'Standard deviation', 'Minimum', 'Maximum',
                                    'Interquartile range', 'Outlier count', 'Outlier rate',
                                    'Skewness', 'Kurtosis']
            summary_stats = pd.concat(dfs, axis=1, keys=summary_stat_columns)
            summary_stats = summary_stats.round({
                'Null rate': 6,
                'Unique rate': 6,
                'Zero rate': 6,
                'Negative rate': 6,
                'Infinite rate': 6,
                'Negative infinite rate': 6,
                'Mean': 6,
                'Median': 6,
                'Standard deviation': 6,
                'Interquartile range': 6,
                'Outlier rate': 6,
                'Skewness': 6,
                'Kurtosis': 6,
            })
            # write to csv
            summary_stats.index.name = 'feature'
            summary_stats.to_csv(summary_statistics_dir / f'summary_stats_{dataset_name}.csv')
            self.logger.info(f'Generated summary statistics ({dataset_name} data)')

            # compute quantiles
            if quantiles is None:  # set quantiles if not specified
                quantiles = [0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1]
            quantile_columns = {q: f"{round(100 * q)}%" for q in quantiles}  # quantile column names
            if 0 in quantiles:
                quantile_columns[0] = "Minimum"
            if 1 in quantiles:
                quantile_columns[1] = "Maximum"
            quantile_df = df.quantile(q=quantiles).T.rename(columns=quantile_columns)
            quantile_df = quantile_df.round(6)
            # write to csv
            quantile_df.index.name = 'Feature'
            quantile_df.to_csv(summary_statistics_dir / f'quantiles_{dataset_name}.csv')
            self.logger.info(f'Generated quantiles ({dataset_name} data)')

            # compute top n value counts
            dfs = []
            for col in df.columns:
                value_counts = df[col].value_counts().head(top_n_value_counts)
                value_counts = value_counts.reset_index().rename({col: "Value"}, axis=1)
                value_counts.insert(0, "Feature", col)
                dfs.append(value_counts)
            all_value_counts = pd.concat(dfs, axis=0)
            all_value_counts.rename({"count": "Count"}, axis=1, inplace=True)
            all_value_counts['Proportion'] = (all_value_counts['Count'] / len(df)).round(6)

            # write to csv
            all_value_counts = all_value_counts.reset_index().rename({"index": "Rank"}, axis=1)
            all_value_counts.to_csv(summary_statistics_dir / f'value_counts_{dataset_name}.csv', index=False)
            self.logger.info(f'Generated value counts ({dataset_name} data)')

            # compute dataset summary statistics
            total_null_cnt = null_cnt.sum()
            duplicate_cnt = df.duplicated().sum()
            dataset_summary_statistics[dataset_name] = {
                'Number of features': str(X.shape[1]),  # convert ints to str, so they aren't written as floats in csv
                'Number of samples': str(n),
                'Null cells': str(total_null_cnt),
                'Null cell rate': round(total_null_cnt / (n * X.shape[1]), 6),
                'Duplicate rows': str(duplicate_cnt),
                'Duplicate row rate': round(duplicate_cnt / n, 6)
            }

        # write dataset summary statistics to csv
        dataset_summary_statistics_df = pd.DataFrame(dataset_summary_statistics)
        dataset_summary_statistics_df.to_csv(summary_statistics_dir / 'dataset_summary_statistics.csv')

    def gen_binary_splits(self, n_splits=10) -> None:
        """
        Generate binary splits table for each feature in each dataset.

        Parameters
        ----------
            n_splits: int
                Number of splits to generate.
        """

        self.logger.info("----- Generating Binary Splits Tables -----")

        assert self.binary_classification, '`binary_classification` must be set to True to generate binary splits'

        # define colour map functions for formatting the table later
        def colour_column(col) -> list[str]:
            """
            Map values in col above the median to green and below the median
            to red with a gradient. Returns a list of formatted strings.

            Parameters
            ----------
                col: pd.Series
                    Column to colour.
            """

            min_val = col.min()
            max_val = col.max()
            med_val = col.median()

            def colour_map(val: float) -> tuple[int, int, int]:
                """
                Map values above the median to green and below the
                median to red with a gradient.

                Parameters
                ----------
                    val: float
                        Value to colour.
                """

                if val >= med_val:
                    cmap = plt.get_cmap('Greens')
                    x = (val - med_val) / (max_val - med_val)
                else:
                    cmap = plt.get_cmap('Reds')
                    x = (med_val - val) / (med_val - min_val)
                r, g, b, _ = cmap(x)
                r, g, b = int(256 * r), int(256 * g), int(256 * b)
                return r, g, b

            format_list = []
            for v in col:
                r, g, b = colour_map(v)
                format_list.append(f'background-color: rgba({r}, {g}, {b}, 0.5)')

            return format_list

        def shade_every_other_row(col):
            """
            Colour every other row, except for the top level index.

            Parameters
            ----------
                col: pd.Series
                    Column to colour.
            """

            if col.name == 0:  # special case for top level index
                return ['background-color: rgba(150, 150, 150, 0.5);'] * len(col)
            return ['background-color: rgba(220, 220, 220, 0.5);'
                    if x % 2 == 0 else '' for x in range(len(col))]

        # make output directory
        binary_splits_dir = self.output_dir / 'binary_splits'
        binary_splits_dir.mkdir(parents=True, exist_ok=True)

        # generate binary splits for each dataset
        for X, y, dataset_name in self.datasets:
            # get dataframes for each class
            df = pd.concat([y, X], axis=1)
            df_0 = df[df.iloc[:, 0] == 0]
            df_1 = df[df.iloc[:, 0] == 1]
            sum_0 = df_0.shape[0]
            sum_1 = df_1.shape[0]

            # generate binary splits table
            records = []
            for feature in tqdm(X.columns):
                # get splits for each feature
                if df[feature].nunique() <= n_splits + 1:  # use unique values as splits if there aren't too many
                    splits = df[feature].unique()
                    splits.sort()
                    splits = splits[:-1]
                else:  # use quantiles if many unique values
                    quantiles = np.arange(1 / (n_splits + 1), 1, 1 / (n_splits + 1))
                    splits = df[feature].quantile(quantiles, interpolation='nearest')
                    splits = splits.round(2).unique()
                    splits.sort()

                # compute metrics for each split
                int_splits = True if (splits == splits.astype(int)).all() else False  # all splits are integers #
                # TODO? could check all values of this feature are integers
                for s in splits:
                    count_0 = (df_0[feature] > s).sum()  # get counts
                    count_1 = (df_1[feature] > s).sum()
                    prop_0 = count_0 / sum_0  # get proportions
                    prop_1 = count_1 / sum_1
                    fp_to_tp = count_0 / count_1 if count_1 > 0 else np.inf
                    p = count_1 / (count_0 + count_1)  # precision
                    r = count_1 / sum_1  # recall
                    f1 = 2 * p * r / (p + r) if p + r > 0 else 0  # f1 score
                    if prop_1 == 0 or prop_1 == 1:
                        iv = 0
                    else:
                        iv = (prop_0 - prop_1) * np.log(np.divide(prop_0, prop_1)) + \
                            (prop_1 - prop_0) * np.log(np.divide(1 - prop_0, 1 - prop_1))  # information value
                    record = {
                        'Feature': feature,
                        'Split': f'> {s}' if int_splits else f'> {s:.2f}',  # round to 2 decimal places if non-integer splits
                        'Class 0 %': round(100 * prop_0, 2),
                        'Class 1 %': round(100 * prop_1, 2),
                        'FP to TP': round(fp_to_tp, 2),
                        'F1-Score': round(f1, 4),
                        'IV': round(iv, 4),
                    }
                    records.append(record)
            t = pd.DataFrame.from_records(records)
            t = t.set_index(['Feature', 'Split'])

            # format table

            # add colours to cells
            s = t.style.apply(colour_column, subset=['F1-Score', 'IV'], axis=0)  # colour F1-Score and IV based on value
            s.apply(shade_every_other_row, subset=['Class 0 %', 'Class 1 %', 'FP to TP'], axis=0)  # shade every 2nd row
            s.apply_index(shade_every_other_row, axis=0)  # shade every other row in Split, and all rows in Feature
            headers = {'selector': 'th.col_heading', 'props': 'background-color: rgba(150, 150, 150, 0.5); color: black;'}
            s.set_table_styles([headers], overwrite=False)  # shade column headers

            # round values
            s.format({'Class 0 %': "{:.2f}",
                      'Class 1 %': "{:.2f}",
                      'FP to TP':  "{:.2f}",
                      'F1-Score':  "{:.4f}",
                      'IV':        "{:.4f}"})

            # cell padding & alignment
            s.set_properties(**{'text-align': 'right'})  # align cells right
            align_text = [
                {'selector': 'th.row_heading.level1', 'props': 'text-align: right;'},  # right-align Splits in row index
                {'selector': 'th.col_heading', 'props': 'text-align: center;'},  # center-align column names
                {'selector': "th.col_heading", 'props': 'padding-left: 10px; padding-right: 10px;'},  # pad column names
                {'selector': "td", 'props': 'padding-left: 0px; padding-right: 20px;'},  # pad regular cells
                {'selector': "td.col4", 'props': 'padding-left: 18px; padding-right: 18px;'},  # pad IV (custom padding for short name)
                {'selector': "th.row_heading.level0", 'props': 'padding-left: 4px; padding-right: 4px;'},  # pad Features
                {'selector': "th.row_heading.level1", 'props': 'padding-left: 4px; padding-right: 10px;'}]  # pad Splits
            s.set_table_styles(align_text, overwrite=False)

            # table properties
            s.set_table_styles([
                {'selector': ' ',
                 'props': 'margin: 0; font-family: "Helvetica", "Arial", sans-serif; border-collapse: collapse; border: none;'}
            ], overwrite=False)

            # add borders and lines
            s.set_table_styles([
                {"selector": "", "props": [("border", "2px solid")]},  # table border
                {"selector": "th.col_heading", "props": [("border-left", "2px solid")]},  # column heading left border
                {"selector": "th.row_heading.level0", "props": [("border-right", "2px solid")]},  # Feature value right border
            ], overwrite=False)
            for _, group_df in t.groupby('Feature'):  # add lines between features
                s.set_table_styles({group_df.index[0]: [{'selector': '', 'props': 'border-top: 2px solid black;'}]},
                                   overwrite=False, axis=1)

            # row hover
            s.set_table_styles([
                {'selector': 'tr:hover', 'props': 'background-color: LightSteelBlue'}  # for cell hover use <td> instead of <tr>
            ], overwrite=False)

            # save html export
            s.to_html(binary_splits_dir / f'binary_splits_{dataset_name}.html')
            self.logger.info(f'Generated binary splits ({dataset_name} data)')

    def xgb_explain(self) -> None:
        """Generate model explanitory charts specific to XGBoost models."""

        self.logger.info("----- Generating XGBoost Feature Importances -----")

        assert isinstance(self.model, xgb.XGBModel), f'self.model is type {type(self.model)}, which is not an XGBoost Model.'

        # Get XGBoost feature importance
        imp_types = ['gain', 'total_gain', 'weight', 'cover', 'total_cover']  # importance types
        bstr = self.model.get_booster()
        imps = [pd.Series(bstr.get_score(importance_type=t), name=t) for t in imp_types]
        df = pd.concat(imps, axis=1)  # dataframe of importances
        df = df.apply(lambda x: x / x.sum(), axis=0)  # normalize so each column sums to 1

        # add in 0 importance features
        features = self.datasets[0][0].columns
        feats_series = pd.Series(index=features, name='temp', dtype='float64')
        df = pd.concat([df, feats_series], axis=1)
        df.drop(columns='temp', inplace=True)
        df.fillna(0, inplace=True)

        # sort and save
        df.sort_values('gain', ascending=False, inplace=True)
        importance_dir = self.output_dir / "feature_importance"
        importance_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(importance_dir/'xgb_feature_importance.csv')
