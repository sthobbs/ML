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
                 output_dir: Optional[Union[str, Path]] = None,
                 logger: Optional[logging.Logger] = None) -> None:
        """
        Parameters
        ----------
            model :
                scikit-learn classifier with a .predict_proba() method.
            datasets :
                List of (X, y, dataset_name) triples.
                e.g. [(X_train, y_train, 'Train'), (X_val, y_val, 'Validation'), (X_test, y_test, 'Test')]
                All datasets, X, should have the same columns.
            output_dir : str, optional
                string path to folder where output will be written.
            logger : logging.Logger, optional
                logger.
        """

        if model is not None:
            self.model = model
        if datasets is None:
            datasets = []
        self.datasets = datasets

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
                number of times to permute each feature (default is 10)
            metrics : str or list of str, optional
                metrics used in permutation feature importance calculations (default is 'neg_log_loss').
                e.g.: 'roc_auc', 'average_precision', 'neg_log_loss', 'r2', etc.
                see https://scikit-learn.org/stable/modules/model_evaluation.html for complete list.
        """

        self.logger.info("----- Generating Permutation Feature Importances -----")

        # make output directory
        importance_dir = self.output_dir / "feature_importance"
        importance_dir.mkdir(parents=True, exist_ok=True)

        # generate permutation feature importance for each metric on each dataset
        for X, y, dataset_name in self.datasets:
            self.logger.info(f"running permutation importance on {dataset_name} data")
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

    def plot_shap(self, shap_sample: Optional[int] = None) -> None:
        """Generate model explanitory charts involving shap values."""

        plt.close('all')
        assert self.model is not None, "self.model can't be None to run plot_shap()"

        # Generate Shap Charts
        self.logger.info("----- Generating Shap Charts -----")
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
            self.logger.info(f'Plotting {dataset_name} partial dependence plots')
            plot_dir = self.output_dir/"shap"/dataset_name/"partial_dependence_plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            for feature in tqdm(dataset.columns):
                fig, ax = shap.partial_dependence_plot(
                    feature, predict, dataset, model_expected_value=True,
                    feature_expected_value=True, show=False, ice=False)
                fig.savefig(f"{plot_dir}/{feature}.png", **savefig_kwargs)
                plt.close()

            # Generate scatter plots (coloured by feature with strongest interaction)
            self.logger.info(f'Plotting {dataset_name} scatter plots')
            explainer = shap.Explainer(self.model, dataset)
            shap_values = explainer(dataset)
            plot_dir = self.output_dir/"shap"/dataset_name/"scatter_plots"
            plot_dir.mkdir(exist_ok=True)
            for feature in tqdm(dataset.columns):
                shap.plots.scatter(shap_values[:, feature], alpha=0.3, color=shap_values, show=False)
                plt.savefig(f"{plot_dir}/{feature}.png", **savefig_kwargs)
                plt.close()

            # Generate beeswarm plot
            self.logger.info(f'Plotting {dataset_name} beeswarm plot')
            shap.plots.beeswarm(shap_values, alpha=0.1, max_display=1000, show=False)
            path = self.output_dir/"shap"/dataset_name/"beeswarm_plot.png"
            plt.savefig(path, **savefig_kwargs)
            plt.close()

            # Generate bar plots
            self.logger.info(f'Plotting {dataset_name} bar plots')
            shap.plots.bar(shap_values, max_display=1000, show=False)
            path = self.output_dir/"shap"/dataset_name/"abs_mean_bar_plot.png"
            plt.savefig(path, **savefig_kwargs)
            plt.close()
            shap.plots.bar(shap_values.abs.max(0), max_display=1000, show=False)
            path = self.output_dir/"shap"/dataset_name/"abs_max_bar_plot.png"
            plt.savefig(path, **savefig_kwargs)
            plt.close()
            # TODO?: make alpha and max_display config variables

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
                the method for choosing bins, either 'fixed' or 'quantiles' (default is 'fixed')
            n_bins : int, optional
                the number of bins used to compute psi (default is 10)
        """

        self.logger.info("----- Generating PSI -----")

        # check for valid input
        assert bin_type in {'fixed', 'quantiles'}, "bin_type must be in {'fixed', 'quantiles'}"

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
        self.output_dir.mkdir(exist_ok=True)
        psi_df.to_csv(self.output_dir/'psi.csv', index=False)

    def _psi_compare(self,
                     scores1: Union[np.ndarray, pd.core.series.Series],
                     scores2: Union[np.ndarray, pd.core.series.Series],
                     bin_type: str = 'fixed',
                     n_bins: int = 10) -> float:
        """
        Compute Population Stability Index (PSI) between two datasets.

        Parameters
        ----------
            scores1 : numpy.ndarray or pandas.core.series.Series
                scores for one of the datasets
            scores2 : numpy.ndarray or pandas.core.series.Series
                scores for the other dataset
            bin_type : str, optional
                the method for choosing bins, either 'fixed' or 'quantiles' (default is 'fixed')
            n_bins : int, optional
                the number of bins used to compute psi (default is 10)
        ...
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
                the method for choosing bins, either 'fixed' or 'quantiles' (default is 'fixed')
            n_bins : int, optional
                the number of bins used to compute csi (default is 10)
        """

        self.logger.info("----- Generating CSI -----")

        # check for valid input
        assert bin_type in {'fixed', 'quantiles'}, "bin_type must be in {'fixed', 'quantiles'}"

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
        self.output_dir.mkdir(exist_ok=True)
        csi_df.sort_values('csi', ascending=False, inplace=True)
        csi_df.to_csv(self.output_dir/'csi_long.csv', index=False)

        # convert csi dataframe to wide format
        csi_df['datasets'] = csi_df['dataset1'] + '-' + csi_df['dataset2']
        csi_df = csi_df[['feature', 'datasets', 'csi']]
        csi_df.set_index('feature', inplace=True)
        csi_df = csi_df.pivot(columns='datasets')['csi']

        # reorder to the same order as features, and save to csv
        csi_df.reset_index(inplace=True)
        csi_df['feature'] = pd.Categorical(csi_df['feature'], features)
        csi_df = csi_df.sort_values("feature").set_index("feature")
        csi_df.to_csv(self.output_dir/'csi_wide.csv')

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
                the method for choosing bins, either 'fixed' or 'quantiles' (default is 'quantiles')
            n_bins : int, optional
                the number of bins used to compute woe and iv (default is 10)
        """

        self.logger.info("----- Generating WOE and IV -----")

        # make output directory
        woe_dir = self.output_dir / 'woe_iv'
        woe_dir.mkdir(parents=True, exist_ok=True)

        for X, y, dataset_name in self.datasets:

            self.logger.info(f"generating woe and iv for {dataset_name} dataset")

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

            woe_df = pd.concat(woe_df_list)
            woe_df.to_csv(woe_dir/f'woe_{dataset_name}.csv', index=False)

            iv_df = pd.DataFrame.from_records(iv_list)
            iv_df.sort_values('adj_iv', ascending=False, inplace=True)
            iv_df.index.name = 'index'
            iv_df.to_csv(woe_dir/f'iv_{dataset_name}.csv')

    def gen_corr(self, max_features: int = 100) -> None:
        """
        Generate correlation matrix and heatmap for each dataset.

        Parameters
        ----------
            max_features : int, optional
                the maximum number of features allowed for charts and plots to be generated
        """

        # check input
        features = self.datasets[0][0].columns
        if len(features) > max_features:
            msg = (f"not computing correlation matrix since there are {len(features)}"
                   f" features, which more than max_features = {max_features}")
            self.logger.warning(msg)
            return

        self.logger.info("----- Generating Correlation Charts -----")

        # make output directory
        corr_dir = self.output_dir / 'correlation'
        corr_dir.mkdir(parents=True, exist_ok=True)

        # for dataset_name, dataset in self.data.items():
        for X, _, dataset_name in self.datasets:
            self.logger.info(f"generating correlations for {dataset_name} dataset")
            corr = X.corr()
            corr_long = pd.melt(corr.reset_index(), id_vars='index')  # unpivot to long format
            # write to csv
            corr.index.name = 'feature'
            corr.to_csv(corr_dir/f'corr_{dataset_name}.csv')
            corr_long.rename(columns={'variable': 'feature_1', 'index': 'feature_2', 'value': 'correlation'}, inplace=True)
            corr_long = corr_long.reindex(columns=['feature_1', 'feature_2', 'correlation'])  # reorder columns
            corr_long = corr_long[corr_long.feature_1 != corr_long.feature_2]
            corr_long.sort_values('correlation', key=abs, ascending=False, inplace=True)
            corr_long.to_csv(corr_dir/f'corr_{dataset_name}_long.csv', index=False)
            # plot heat map
            self.plot_corr_heatmap(corr, corr_dir/f'heatmap_{dataset_name}.png', data_type='corr')

    def plot_corr_heatmap(self,
                          data: pd.core.frame.DataFrame,
                          output_path: Union[str, Path],
                          data_type: str = 'corr') -> None:
        """
        Plot correlation heat map.

        Parameters
        ----------
            data : pandas.core.frame.DataFrame
                input data, either raw data, or correlation matrix
            output_path : str
                the location where the plot should be written.
                note that this function assumes that the parent folder exists.
            data_type : str, optional
                the type of input passed into the data argument (default is 'corr')
                - 'features' => the raw feature table is passed in
                - 'corr' => a correlation matrix is passed in

        Credits: https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
        """

        # check inputs
        valid_data_types = {'features', 'corr'}
        assert data_type in valid_data_types, f"invalid data_type: {data_type}"

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
            ax.set_xlim([-0.5, len(features) - 0.5])
            ax.set_ylim([-0.5, len(features) - 0.5])

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
            ax.set_yticklabels([-1, 0, 1], fontsize=font_size)
            ax.yaxis.tick_right()  # Show vertical ticks on the right

            # save figure
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()

    def xgb_explain(self) -> None:
        """Generate model explanitory charts specific to XGBoost models."""

        assert isinstance(self.model, xgb.XGBModel), f'model is type {type(self.model)}, which is not an XGBoost Model'

        # Get XGBoost feature importance
        self.logger.info("----- Generating XGBoost Feature Importances -----")
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
