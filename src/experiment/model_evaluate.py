import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, \
    auc, roc_curve, det_curve, DetCurveDisplay, brier_score_loss, log_loss, \
    roc_auc_score
from sklearn.base import BaseEstimator
import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from tqdm import tqdm
from scipy.stats import ks_2samp
import logging
from typing import Optional, List, Union, Tuple
matplotlib.use('agg')


def metric_score(y_true, y_score, metric: str) -> float:
    """
    Evaluate model scores for a given performance metric.

    Parameters
    ----------
        y_true : array-like of shape (n_sample,)
            ground truth labels.
        y_score : array-like of shape (n_sample,)
            model scores.
        metric: {'average_precision', 'aucpr', 'auc', 'log_loss', 'brier_loss'}
            Metric used to measure performance.
    """

    valid_metrics = {'average_precision', 'aucpr', 'auc', 'log_loss', 'brier_loss'}
    assert metric in valid_metrics, "invalid metric"

    score: float

    # Average precision
    if metric == 'average_precision':
        score = average_precision_score(y_true, y_score)

    # Area under the precision-recall curve
    elif metric == 'aucpr':
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        score = auc(recall, precision)

    # Area under the ROC curve
    elif metric == 'auc':
        score = roc_auc_score(y_true, y_score)

    # Log loss (cross entropy)
    elif metric == 'log_loss':
        score = log_loss(y_true, y_score)

    # Brier score loss
    elif metric == 'brier_loss':
        score = brier_score_loss(y_true, y_score)

    return score


class ModelEvaluate():
    """
    Generates plots and charts to evaluate a model.

    Author:
       Steve Hobbs
       github.com/sthobbs
    """

    def __init__(self,
                 model: Optional[BaseEstimator] = None,
                 datasets: Optional[List[Tuple[
                    Union[np.ndarray, pd.core.frame.DataFrame, pd.core.series.Series],
                    Union[np.ndarray, pd.core.frame.DataFrame, pd.core.series.Series],
                    str]]] = None,
                 datasets_have_y_score: Optional[bool] = False,
                 output_dir: Optional[Union[str, Path]] = None,
                 aux_fields: Optional[List[str]] = None,
                 logger: Optional[logging.Logger] = None) -> None:
        """
        Parameters
        ----------
            model :
                scikit-learn classifier with a .predict_proba() method.
            datasets :
                List of (X, y, dataset_name) triples, or list of (y_score, y, dataset_name) triples
                e.g. [(X_train, y_train, 'Train'), (X_val, y_val, 'Validation'), (X_test, y_test, 'Test')]
            datasets_have_y_score :
                True if datasets is a list of (y_score, y, dataset_name) triples,
                False if datasets is a list of (X, y, dataset_name) triples, (default is False).
            output_dir : str, optional
                string path to folder where output will be written.
            aux_fields : list, optional
                auxiliary fields to use to create additional metrics.
            logger : logging.Logger, optional
                logger.
        """

        self.model = model
        if datasets is None:
            datasets = []
        self.datasets = datasets
        self.datasets_have_y_score = datasets_have_y_score

        # Make directories
        if output_dir:
            print('output_dir = ', output_dir)
            self.output_dir = Path(output_dir)
            self.plots_subdir = self.output_dir / 'plots'
            self.tables_subdir = self.output_dir / 'tables'
            self.plots_subdir.mkdir(parents=True, exist_ok=True)
            self.tables_subdir.mkdir(exist_ok=True)

        # Set auxiliary fields
        if aux_fields is None:
            aux_fields = []
        self.aux_fields = aux_fields

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

    def binary_evaluate(self, increment: float = 0.01) -> None:
        """
        Generate plots and tables for a binary classification model.

        class label values must be either 0 or 1.

        Parameters
        ----------
            increment : float
                threshold increment to use when checking performance on a sequence of thresholds
        """

        assert self.model is not None or self.datasets_have_y_score, \
            "self.model must not be None to run this method, unless self.datasets_have_y_score is True"
        assert self.datasets is not None, "self.datasets must not be None to run this method"
        assert self.output_dir is not None, "self.output_dir must not be None to run this method"
        plt.close('all')

        for X, y_true, dataset_name in self.datasets:
            self.logger.info(f"----- Generating {dataset_name} Data Metrics -----")
            y_score = X if self.datasets_have_y_score else self.model.predict_proba(X)
            if y_score.ndim == 2:
                assert y_score.shape[1] == 2, f'unexpected y_score shape: {y_score.shape}'
                y_score = y_score[:, 1]

            # Generate Precision/Recall vs Threshold
            precision, recall, thresholds = precision_recall_curve(y_true, y_score)
            self._plot_precision_recall_threshold(precision, recall, thresholds, dataset_name)

            # Generate Precision vs Recall
            average_precision = average_precision_score(y_true, y_score)
            precision_recall_auc = auc(recall, precision)
            self._plot_precision_recall(precision, recall, dataset_name, average_precision, precision_recall_auc)

            # Generate ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)  # same as sklearn.metrics.roc_auc_score(y_true, y_score)
            self._plot_roc(fpr, tpr, dataset_name, roc_auc)

            # Generate CAP Curve
            accuracy_ratio = self._plot_cap(y_true, y_score, dataset_name)

            # Generate Detection Error Tradeoff Curve
            fpr, fnr, _ = det_curve(y_true, y_score)
            self._plot_det(fpr, fnr, dataset_name)

            # Generate Score Histogram
            self._plot_score_hist(y_true, y_score, dataset_name)

            # Generate Threshold vs Metrics Table
            self._threshold_table(y_true, y_score, dataset_name, increment=increment)

            # Generate Metrics Table
            self._metrics_table(y_true, y_score, dataset_name, roc_auc, precision_recall_auc, average_precision, accuracy_ratio)

        # Generate Kolmogorov-Smirnov (KS) Statistic
        self.ks_statistic()

        plt.close('all')

    def xgb_evaluate(self, dataset_names: Optional[List[str]] = None) -> None:
        """
        Generate plots and tables specific to XGBoost models.

        Parameters
        ----------
            dataset_names:
                list of dataset names (in order) for the datasets passed into eval_set when the model was fit.
                e.g. ['Train', 'Test', 'Validation'] (train with validation last, since that's used for early stopping)
        """

        assert isinstance(self.model, xgb.XGBModel), f'model is type {type(self.model)}, which is not an XGBoost Model'
        assert self.output_dir is not None, "self.output_dir must not be None to run this method"
        assert self.datasets is not None, "self.datasets must not be None to run this method"
        plt.close('all')

        self.logger.info("----- Generating XGBoost Metrics -----")

        # Plot training metrics vs n_estimators
        default_names = list(self.model.evals_result().keys())  # default names for evaluation sets (e.g. ['validation_0', 'validation_1'])
        if dataset_names is None:  # use default names is no dataset names are passed in
            dataset_names = default_names
        assert len(dataset_names) == len(default_names), "len(dataset_names) doesn't match training eval_set"
        name_pairs = list(zip(dataset_names, default_names))
        evals_result = self.model.evals_result() if self.model.evals_result() else {}
        eval_dict = evals_result.get(default_names[0], {})
        metrics = list(eval_dict.keys())  # evaluation metrics used in training (e.g. ['aucpr', 'logloss'])
        n_estimators = len(eval_dict.get(metrics[0], []))  # max n_estimators
        plt.figure()
        with plt.style.context(self.plot_context):
            for metric in metrics:
                for dataset_name, default_name in name_pairs:
                    plt.plot(range(1, n_estimators+1), evals_result.get(default_name, {}).get(metric), label=dataset_name)
                plt.xlabel('n_estimators')
                plt.ylabel(metric)
                plt.title(f'n_estimators vs {metric}')
                plt.legend(bbox_to_anchor=(1, 1.02), loc='lower right', borderaxespad=0, frameon=True)
                plt.savefig(f'{self.plots_subdir}/n_estimators_vs_{metric}.png', bbox_inches='tight', pad_inches=0.3)
                plt.clf()
                self.logger.info(f'Plotted n_estimators vs {metric}')
        plt.close()

        # Generate n_estimates vs training metrics tables
        for dataset_name, default_name in name_pairs:
            df = pd.DataFrame(evals_result.get(default_name), index=range(1, n_estimators+1))
            df.index.name = 'n_estimators'
            df.to_csv(f'{self.tables_subdir}/n_estimators_vs_metrics_{dataset_name}.csv')
            self.logger.info(f'Generated n_estimators vs metrics table ({dataset_name} data)')

        # Generate optimal n_estimates table
        df = pd.DataFrame(index=metrics, columns=dataset_names)
        df.index.name = 'Metric'
        metrics_to_minimize = {
            'rmse', 'rmsle', 'mae', 'mape', 'mphe', 'logloss', 'error',
            'merror', 'mlogloss', 'poisson-nloglik', 'gamma-nloglik', 'cox-nloglik',
            'gamma-deviance', 'tweedie-nloglik', 'aft-nloglik'
        }
        metrics_to_maximize = {'auc', 'aucpr'}
        for metric in metrics:
            if metric in metrics_to_minimize or metric.startswith('error@'):
                f = np.argmin
            elif metric in metrics_to_maximize or metric.startswith('ndcg') or metric.startswith('map'):
                f = np.argmax
            else:
                self.logger.warning(f"unexpected metric '{metric}', skipping it")
                continue
            for dataset_name, default_name in name_pairs:
                n_estimator_performance = np.array(evals_result.get(default_name, {}).get(metric))
                best_n_estimators = f(n_estimator_performance) + 1
                df.at[metric, dataset_name] = best_n_estimators
        df.to_csv(f'{self.tables_subdir}/optimal_n_estimators.csv')
        self.logger.info('Generated optimal n_estimators table')

    def _plot_precision_recall_threshold(self,
                                         precision: np.ndarray,
                                         recall: np.ndarray,
                                         thresholds: np.ndarray,
                                         dataset_name: str) -> None:
        """
        Plot the precision and recall against the threshold.

        Parameters
        ----------
            precision : numpy.ndarray
                model precision at various thresholds
            recall : numpy.ndarray
                model recall at various thresholds
            thresholds : numpy.ndarray
                various thresholds
            dataset_name : str
                name of dataset to generate plot of
        """

        plt.figure()
        with plt.style.context(self.plot_context):
            plt.plot(thresholds, precision[:-1], 'g-', label='Precision')
            plt.plot(thresholds, recall[:-1], 'b--', label='Recall')
            plt.xlim(-0.05, 1.05)
            plt.ylim(-0.05, 1.05)
            plt.xlabel('Threshold')
            plt.title(f'Precision/Recall vs Threshold ({dataset_name} data)')
            plt.legend(bbox_to_anchor=(0.94, 0.055), loc='lower right', borderaxespad=0, frameon=True)
            plt.savefig(f'{self.plots_subdir}/precision_recall_vs_threshold_{dataset_name}.png')
            self.logger.info(f'Plotted Precision/Recall vs Threshold ({dataset_name} data)')
        plt.close()

    def _plot_precision_recall(self,
                               precision: np.ndarray,
                               recall: np.ndarray,
                               dataset_name: str,
                               average_precision: float,
                               precision_recall_auc: float) -> None:
        """
        Plot the precision against the recall.

        Parameters
        ----------
            precision : numpy.ndarray
                model precision at various thresholds
            recall : numpy.ndarray
                model recall at various thresholds
            dataset_name : str
                name of dataset to generate plot of
            average_precision : float
                average precision of the model
            precision_recall_auc : float
                area under the PR curve of the model
        """

        plt.figure()
        with plt.style.context(self.plot_context):
            plt.plot(recall, precision, 'b-')
            plt.xlim(-0.05, 1.05)
            plt.ylim(-0.05, 1.05)
            plt.ylabel('Precision')
            plt.xlabel('Recall')
            plt.title(f'Precision vs Recall ({dataset_name} data)')
            plt.text(0.015, 0.02,
                     f"Average Precision:    \u2009\u2009\u2009{average_precision:.8f} \nPrecision Recall AUC: {precision_recall_auc:.8f}",
                     size=10, ha="left", va="bottom",
                     bbox=dict(boxstyle="round, rounding_size=0.2",
                               ec=(0.8, 0.8, 0.8, 1),
                               fc=(0.9176470588235294, 0.9176470588235294, 0.9490196078431372, 0.75)))
            plt.savefig(f'{self.plots_subdir}/precision_vs_recall_{dataset_name}.png')
            self.logger.info(f'Plotted Precision vs Recall ({dataset_name} data)')
        plt.close()

    def _plot_roc(self,
                  fpr: np.ndarray,
                  tpr: np.ndarray,
                  dataset_name: str,
                  roc_auc: float) -> None:
        """
        Plot ROC curve.

        Parameters
        ----------
            fpr : numpy.ndarray
                false positive rate at various thresholds
            tpr : numpy.ndarray
                true positive rate at various thresholds
            dataset_name : str
                name of dataset to generate plot of
            roc_auc : float
                area under the ROC curve of the model
        """

        plt.figure()
        with plt.style.context(self.plot_context):
            plt.plot(fpr, tpr, 'b-')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim(-0.05, 1.05)
            plt.ylim(-0.05, 1.05)
            plt.ylabel('TPR')
            plt.xlabel('FPR')
            plt.title(f'ROC ({dataset_name} data)')
            plt.text(0.985, 0.02, f"ROC AUC: {roc_auc:.8f}",
                     size=10, ha="right", va="bottom",
                     bbox=dict(boxstyle="round, rounding_size=0.2",
                               ec=(0.8, 0.8, 0.8, 1),
                               fc=(0.9176470588235294, 0.9176470588235294, 0.9490196078431372, 0.75)))
            plt.savefig(f'{self.plots_subdir}/roc_{dataset_name}.png')
            self.logger.info(f'Plotted ROC ({dataset_name} data)')
        plt.close()

    def _plot_det(self,
                  fpr: np.ndarray,
                  fnr: np.ndarray,
                  dataset_name: str) -> None:
        """
        Plot detection error tradeoff curve.

        Parameters
        ----------
            fpr : numpy.ndarray
                false positive rate at various thresholds
            fnr : numpy.ndarray
                false negative rate at various thresholds
            dataset_name : str
                name of dataset to generate plot of
        """

        plt.figure()
        with plt.style.context(self.plot_context):
            DetCurveDisplay(fpr=fpr, fnr=fnr).plot()
            plt.title(f'Detection Error Tradeoff ({dataset_name} data)')
            plt.savefig(f'{self.plots_subdir}/det_{dataset_name}.png')
            self.logger.info(f'Plotted Detection Error Tradeoff ({dataset_name} data)')
        plt.close()


    def _plot_cap(self, y_true, y_score, dataset_name: str) -> float:
        """
        Plot cumulative accuracy profile curve and return accuracy ratio.

        Parameters
        ----------
            y_true : array-like of shape (n_sample,)
                ground truth labels.
            y_score : array-like of shape (n_sample,)
                model scores.
            dataset_name : str
                name of dataset to generate plot of
        """

        total_cnt = len(y_true)
        pos_cnt = y_true.sum()

        plt.figure()
        with plt.style.context(self.plot_context):

            # plot perfect model CAP
            x_perfect = [0, pos_cnt, total_cnt]
            y_perfect = [0, pos_cnt, pos_cnt]
            plt.plot(x_perfect, y_perfect, c='grey', label='Perfect Model') 

            # plot actual model CAP
            ordered_y_true = [y for _, y in sorted(zip(y_score, y_true), reverse=True)] 
            x_actual = np.arange(0, total_cnt + 1)
            y_actual = np.append([0], np.cumsum(ordered_y_true))
            plt.plot(x_actual, y_actual, c='b', label='Actual Model') 

            # plot random model CAP
            x_random = [0, total_cnt]
            y_random = [0, pos_cnt]
            plt.plot(x_random, y_random, 'k--', label='Random Model')

            # compute accuracy ratio
            random_area = auc(x_random, y_random)  # area under random model
            perfect_area = auc(x_perfect, y_perfect)  # area under perfect model
            actual_area = auc(x_actual, y_actual)  # area under actual model
            accuracy_ratio = (actual_area - random_area) / (perfect_area - random_area)

            # configure plot
            plt.xlim(-0.05 * total_cnt, 1.05 * total_cnt)
            plt.ylim(-0.05 * pos_cnt, 1.05 * pos_cnt)
            plt.ylabel('Positive Events')
            plt.xlabel('Events')
            plt.title(f'CAP ({dataset_name} data)')
            plt.legend(bbox_to_anchor=(0.94, 0.125), loc='lower right', borderaxespad=0, frameon=True)
            plt.text(0.975 * total_cnt, 0.02 * pos_cnt, f"Accuracy Ratio: {accuracy_ratio:.8f}",
                     size=10, ha="right", va="bottom",
                     bbox=dict(boxstyle="round, rounding_size=0.2",
                               ec=(0.8, 0.8, 0.8, 1),
                               fc=(0.9176470588235294, 0.9176470588235294, 0.9490196078431372, 0.75)))
            plt.savefig(f'{self.plots_subdir}/cap_{dataset_name}.png')
            self.logger.info(f'Plotted CAP ({dataset_name} data)')
        plt.close()

        return accuracy_ratio

    def _plot_score_hist(self, y_true, y_score, dataset_name: str) -> None:
        """
        Plot score histogram vs class label.

        Parameters
        ----------
            y_true : array-like of shape (n_sample,)
                ground truth labels.
            y_score : array-like of shape (n_sample,)
                model scores.
            dataset_name : str
                name of dataset to generate plot of
        """

        plt.figure()
        with plt.style.context(self.plot_context):
            bins = np.arange(0, 1.02, 0.02)
            kw_args = {"stat": "probability", "bins": bins, "kde": False, "alpha": 0.4}
            sns.histplot(y_score[y_true == 0], label="Class: 0", **kw_args, color="dodgerblue")
            sns.histplot(y_score[y_true == 1], label="Class: 1", **kw_args, color="orange")
            plt.xlabel('Score')
            plt.title(f"Score Histogram ({dataset_name} data)")
            plt.legend(bbox_to_anchor=(0.99, 0.99), loc='upper right', borderaxespad=0, frameon=True)
            plt.savefig(f'{self.plots_subdir}/score_histogram_{dataset_name}.png')
            self.logger.info(f'Plotted Score Histogram ({dataset_name} data)')
        plt.close()

    def _threshold_table(self, y_true, y_score, dataset_name: str, increment: float = 0.01) -> None:
        """
        Evaluate model performance at various thresholds, and save results to a csv.

        Parameters
        ----------
            y_true : array-like of shape (n_sample,)
                ground truth labels.
            y_score : array-like of shape (n_sample,)
                model scores.
            dataset_name : str
                name of dataset to generate plot of
            increment : float, default = 0.01
                difference between consecutive threshold values to evaluate performance at
        """

        assert 0 < increment < 1, f'increment={increment}, it should be >0 and <=1'

        # get data from numeric auxiliary fields to create additional metrics
        numeric_aux_fields = []
        if self.aux_fields:
            idx = [i for i, (_, _, name) in enumerate(self.datasets) if name == dataset_name][0]
            numeric_fields = set(self.datasets[idx][0].select_dtypes([np.number]).columns)
            numeric_aux_fields = [f for f in self.aux_fields if f in numeric_fields]
            aux_data = self.datasets[idx][0][numeric_aux_fields]

        # initialize performance tracking table
        performance = []

        # make sorted dataframe of data, with extra fields
        df = pd.DataFrame({'y_true': y_true, 'y_score': y_score})
        for f in numeric_aux_fields:
            df[f] = aux_data[f] * df['y_true']  # aux fields only counted for positive class
        df = df.sort_values('y_score').reset_index(drop=True)

        # initialize true positive, false positive, false negative, and true negative counts
        tp = df['y_true'].sum()
        fp = len(df) - tp
        fn = 0
        tn = 0
        aux_tp = {f: df[f].sum() for f in numeric_aux_fields}  # tp sum for aux fields
        idx_lower = 0  # threshold lower bound index
        for threshold in tqdm(np.arange(0, 1, increment)):
            # update performance dataframe
            row = {
                'threshold': threshold,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn,
            } | aux_tp
            performance.append(row)
            # find index of next threshold and update counts/sums
            idx_upper = df['y_score'].searchsorted(threshold + increment)  # index of first value >= threshold + increment
            df_subset = df.loc[idx_lower: idx_upper-1, ['y_true'] + numeric_aux_fields]  # records in idx range
            df_pos = df_subset.loc[df_subset['y_true'] == 1]  # postive class records in idx range
            pos_count = len(df_pos)  # number of positive cases in threshold range
            neg_count = idx_upper - idx_lower - pos_count  # number of negative cases in threshold range
            tp -= pos_count
            fp -= neg_count
            fn += pos_count
            tn += neg_count
            for k, v in aux_tp.items():  # update tp sum for aux fields
                aux_tp[k] -= df_pos[k].sum()
            idx_lower = idx_upper

        # convert to dataframe
        performance_df = pd.DataFrame.from_records(performance)
        columns = ['threshold', 'tp', 'fp', 'fn', 'tn'] + numeric_aux_fields
        performance_df = performance_df.reindex(columns=columns)  # reorder columns

        # combine fields
        performance_df['precision'] = performance_df['tp'] / (performance_df['tp'] + performance_df['fp'])
        performance_df['recall'] = performance_df['tp'] / (performance_df['tp'] + performance_df['fn'])
        performance_df['fp_to_tp'] = performance_df['fp'] / performance_df['tp']
        performance_df['accuracy'] = (performance_df['tp'] + performance_df['tn']) / len(df)
        performance_df['F1'] = 2 * performance_df['tp'] / (2 * performance_df['tp'] + performance_df['fp'] + performance_df['fn'])
        performance_df['specificity'] = performance_df['tn'] / (performance_df['tn'] + performance_df['fp'])
        performance_df['fpr'] = 1 - performance_df['specificity']
        performance_df.reset_index(drop=True)

        # add weighted recall for aux fields & rename aux fields
        for f in numeric_aux_fields:
            performance_df[f"{f}_weighted_recall"] = performance_df[f] / performance_df.at[0, f]
        performance_df.rename(columns={f: f"{f}_tp_sum" for f in numeric_aux_fields}, inplace=True)

        # save file
        performance_df.to_csv(f'{self.tables_subdir}/threshold_vs_metrics_{dataset_name}.csv', index=False)
        self.logger.info(f'Generated threshold performance table ({dataset_name} data)')

    def _metrics_table(self,
                       y_true,
                       y_score,
                       dataset_name: str,
                       roc_auc: float,
                       precision_recall_auc: float,
                       average_precision: float,
                       accuracy_ratio: float) -> None:
        """
        Generate table of metrics for binary classification model evaluation.

        Parameters
        ----------
            y_true : array-like of shape (n_sample,)
                ground truth labels.
            y_score : array-like of shape (n_sample,)
                model scores.
            dataset_name : str
                name of dataset to generate plot of
            roc_auc : float
                area under the ROC curve of the model
            precision_recall_auc : float
                area under the PR curve of the mode
            average_precision : float
                average precision of the model
            accuracy_ratio : float
                accuracy ratio of the model
        """

        log_loss_ = log_loss(y_true, y_score)  # cross entropy
        brier_score = brier_score_loss(y_true, y_score)
        metrics = [
            ('ROC AUC', roc_auc),
            ('Precision Recall AUC', precision_recall_auc),
            ('Average Precision', average_precision),
            ('Accuracy Ratio', accuracy_ratio),
            ('Log Loss (Cross Entropy)', log_loss_),
            ('Brier Score Loss', brier_score),
        ]
        df = pd.DataFrame(metrics, columns=['Metric', 'Value'])
        df.to_csv(f'{self.tables_subdir}/metrics_{dataset_name}.csv', index=False)
        self.logger.info(f'Generated metrics table ({dataset_name} data)')

    def ks_statistic(self) -> None:
        """Generate Kolmogorov-Smirnov (KS) Statistic."""

        # intialize output table
        performance = []

        # generate KS Statistic for each dataset
        for X, y_true, dataset_name in self.datasets:
            y_score = X if self.datasets_have_y_score else self.model.predict_proba(X)
            if y_score.ndim == 2:
                assert y_score.shape[1] == 2, f'unexpected y_score shape: {y_score.shape}'
                y_score = y_score[:, 1]
            ks_stat, p_value = ks_2samp(y_score[y_true == 0], y_score[y_true == 1])
            row = {'dataset': dataset_name, 'ks': ks_stat, 'p-value': p_value}
            performance.append(row)

        # convert to dataframe
        performance_df = pd.DataFrame.from_records(performance)
        performance_df = performance_df.reindex(columns=['dataset', 'ks', 'p-value'])  # reorder columns

        # save output to csv
        performance_df.to_csv(self.tables_subdir/"ks_statistics.csv", index=False)
        self.logger.info('Generated Kolmogorov-Smirnov (KS) Statistic table')
