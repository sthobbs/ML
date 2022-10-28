
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, \
    auc, roc_curve, det_curve, DetCurveDisplay, brier_score_loss, log_loss, \
    roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from tqdm import tqdm
from scipy.stats import ks_2samp
import logging


def metric_score(y_true, y_score, metric):
    """Evaluate model scores for a given performance metric."""

    valid_metrics = {'average_precision', 'aucpr', 'auc', 'log_loss', 'brier_loss'}
    assert metric in valid_metrics, "invalid metric"
    
    # Average precision
    if metric == 'average_precision':
        return average_precision_score(y_true, y_score)
    
    # Area under the precision-recall curve
    elif metric == 'aucpr':
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)
    
    # Area under the ROC curve
    elif metric == 'auc':
        return roc_auc_score(y_true, y_score)
    
    # Log loss (cross entropy)
    elif metric == 'log_loss':
        return log_loss(y_true, y_score)
    
    # Brier score loss
    elif metric == 'brier_loss':
        return brier_score_loss(y_true, y_score)


class ModelEvaluate():
    """
    Generates plots and charts to evaluate a model.

    Author:
       Steve Hobbs
       github.com/sthobbs
    """

    def __init__(self, model=None, datasets=None, output_dir=None, aux_fields=None, logger=None):
        """
        Parameters
        ----------
            model :
                scikit-learn classifier with a .predict_proba() method.
            datasets :
                List of (X, y, dataset_name) triples.
                e.g. [(X_train, y_train, 'Train'), (X_val, y_val, 'Validation'), (X_test, y_test, 'Test')]
            output_dir : str, optional
                string path to folder where output will be written.
            aux_fields : list, optional
                auxiliary fields to use to create additional metrics.
            logger : logging.Logger, optional
                logger.
        """

        self.model = model      
        self.datasets = datasets
        
        # Make directories
        if output_dir:
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
        self.plot_context = 'seaborn-darkgrid'

        # Set up logger
        self.logger = logger
        if logger is None:
            # create logger
            self.logger = logging.getLogger(__name__).getChild(self.__class__.__name__).getChild(str(id(self)))
            self.logger.setLevel(logging.INFO)
            # create formatter
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            # create and add handlers for console output
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def binary_evaluate(self, increment=0.01):
        """
        Generate plots and tables for a binary classification model.

        class label values must be either 0 or 1.

        Parameters
        ----------
            increment :
                threshold increment to use when checking performance on a sequence of thresholds
        """

        assert self.model is not None, "self.model must not be None to run this method"
        assert self.datasets is not None, "self.datasets must not be None to run this method"
        assert self.output_dir is not None, "self.output_dir must not be None to run this method"
        plt.close('all')
        
        for X, y_true, dataset_name in self.datasets:
            self.logger.info(f"----- Generating {dataset_name} Data Metrics -----")
            y_score = self.model.predict_proba(X)[:,1]
            
            # Generate Precision/Recall vs Threshold
            precision, recall, thresholds = precision_recall_curve(y_true, y_score)
            self._plot_precision_recall_threshold(precision, recall, thresholds, dataset_name)

            # Generate Precision vs Recall
            average_precision = average_precision_score(y_true, y_score)
            precision_recall_auc = auc(recall, precision)
            self._plot_precision_recall(precision, recall, dataset_name, average_precision, precision_recall_auc)

            # Generate ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr) # same as sklearn.metrics.roc_auc_score(y_true, y_score)
            self._plot_roc(fpr, tpr, dataset_name, roc_auc)

            # Generate Detection Error Tradeoff Curve
            fpr, fnr, _ = det_curve(y_true, y_score)
            self._plot_det(fpr, fnr, dataset_name)

            # Generate Score Histogram
            self._plot_score_hist(y_true, y_score, dataset_name)
            
            # Generate Threshold vs Metrics Table
            self._threshold_table(y_true, y_score, dataset_name, increment=increment, aux_data=X[self.aux_fields])

            # Generate Metrics Table
            self._metrics_table(y_true, y_score, dataset_name, roc_auc, precision_recall_auc, average_precision)

        # Generate Kolmogorov-Smirnov (KS) Statistic
        self.ks_statistic()

        plt.close('all')

    def xgb_evaluate(self, dataset_names=None):
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

        self.logger.info(f"----- Generating XGBoost Metrics -----")

        # Plot training metrics vs n_estimators
        default_names = list(self.model.evals_result().keys()) # default names for evaluation sets (e.g. ['validation_0', 'validation_1'])
        if dataset_names is None: # use default names is no dataset names are passed in
            dataset_names = default_names
        assert len(dataset_names) == len(default_names), "len(dataset_names) doesn't match training eval_set"
        name_pairs = list(zip(dataset_names, default_names))
        metrics = list(self.model.evals_result().get(default_names[0]).keys()) # evaluation metrics used in training (e.g. ['error', 'auc', 'aucpr', 'logloss'])
        n_estimators = len(self.model.evals_result().get(default_names[0]).get(metrics[0])) # max number of estimators we have performance metrics for
        plt.figure()
        with plt.style.context(self.plot_context):
            for metric in metrics:
                for dataset_name, default_name in name_pairs:
                    plt.plot(range(1,n_estimators+1), self.model.evals_result().get(default_name).get(metric), label=dataset_name)
                plt.xlabel('n_estimators')
                plt.ylabel(metric)
                plt.title(f'n_estimators vs {metric}')
                plt.legend(bbox_to_anchor=(1, 1.02), loc='lower right', borderaxespad=0, frameon=True)
                plt.savefig(f'{self.plots_subdir}/n_estimators_vs_{metric}.png',
                    bbox_inches='tight', pad_inches=0.3)
                plt.clf()
                self.logger.info(f'Plotted n_estimators vs {metric}')
        plt.close()

        # Generate n_estimates vs training metrics tables
        for dataset_name, default_name in name_pairs:
            df = pd.DataFrame(self.model.evals_result().get(default_name), index=range(1,n_estimators+1))
            df.index.name = 'n_estimators'
            df.to_csv(f'{self.tables_subdir}/n_estimators_vs_metrics_{dataset_name}.csv')
            self.logger.info(f'Generated n_estimators vs metrics table ({dataset_name} data)')

        # Generate optimal n_estimates table
        df = pd.DataFrame(index=metrics, columns=dataset_names)
        df.index.name = 'Metric'
        metrics_to_minimize = {'rmse', 'rmsle', 'mae', 'mape', 'mphe', 'logloss', 'error', \
            'merror', 'mlogloss', 'poisson-nloglik', 'gamma-nloglik', 'cox-nloglik', \
            'gamma-deviance', 'tweedie-nloglik', 'aft-nloglik'}
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
                best_n_estimators = f(self.model.evals_result().get(default_name).get(metric)) + 1
                df.at[metric, dataset_name] = best_n_estimators
        df.to_csv(f'{self.tables_subdir}/optimal_n_estimators.csv')
        self.logger.info(f'Generated optimal n_estimators table')

    def _plot_precision_recall_threshold(self, precision, recall, thresholds, dataset_name):
        """Plot the precision and recall against the threshold."""

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

    def _plot_precision_recall(self, precision, recall, dataset_name, average_precision, precision_recall_auc):
        """Plot the precision against the recall."""

        plt.figure()
        with plt.style.context(self.plot_context):
            plt.plot(recall, precision, 'b-')
            plt.xlim(-0.05, 1.05)
            plt.ylim(-0.05, 1.05)
            plt.text(0.015, 0.02,
                     f"Average Precision:    \u2009\u2009\u2009{average_precision:.8f} \nPrecision Recall AUC: {precision_recall_auc:.8f}",
                     size=10, ha="left", va="bottom",
                     bbox=dict(boxstyle="round, rounding_size=0.2",
                               ec=(0.8, 0.8, 0.8, 1),
                               fc=(0.9176470588235294, 0.9176470588235294, 0.9490196078431372, 0.75)))
            plt.ylabel('Precision')
            plt.xlabel('Recall')
            plt.title(f'Precision vs Recall ({dataset_name} data)')
            plt.savefig(f'{self.plots_subdir}/precision_vs_recall_{dataset_name}.png')
            self.logger.info(f'Plotted Precision vs Recall ({dataset_name} data)')
        plt.close()

    def _plot_roc(self, fpr, tpr, dataset_name, roc_auc):
        """Plot ROC curve."""

        plt.figure()
        with plt.style.context(self.plot_context):
            plt.plot(fpr, tpr, 'b-')
            plt.plot([0,1], [0,1], 'k--')
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

    def _plot_det(self, fpr, fnr, dataset_name):
        """Plot ROC curve."""

        plt.figure()
        with plt.style.context(self.plot_context):
            DetCurveDisplay(fpr=fpr, fnr=fnr).plot()
            plt.title(f'Detection Error Tradeoff ({dataset_name} data)')
            plt.savefig(f'{self.plots_subdir}/det_{dataset_name}.png')
            self.logger.info(f'Plotted Detection Error Tradeoff ({dataset_name} data)')
        plt.close()

    def _plot_score_hist(self, y_true, y_score, dataset_name):
        """Plot score histogram vs class label."""

        plt.figure()
        with plt.style.context(self.plot_context):
            bins = np.arange(0, 1.02, 0.02)
            kw_args = {"stat": "probability", "bins": bins, "kde": False, "alpha": 0.4}
            sns.histplot(y_score[y_true==0], label="Class: 0", **kw_args, color="dodgerblue")
            sns.histplot(y_score[y_true==1], label="Class: 1", **kw_args, color="orange")
            plt.xlabel('Score')
            plt.title(f"Score Histogram ({dataset_name} data)")
            plt.legend(bbox_to_anchor=(0.99, 0.99), loc='upper right', borderaxespad=0, frameon=True)
            plt.savefig(f'{self.plots_subdir}/score_histogram_{dataset_name}.png')
            self.logger.info(f'Plotted Score Histogram ({dataset_name} data)')
        plt.close()

    def _threshold_table(self, y_true, y_score, dataset_name, increment=0.01, aux_data=None):
        """
        Evaluate model performance at various thresholds, and save results to a csv.

        Parameters
        ----------
            increment :
                difference between consecutive threshold values to evaluate performance at 
        """

        assert 0 < increment < 1, f'increment={increment}, it should be >0 and <=1'
        
        # initialize performance tracking table
        performance = []

        # make sorted dataframe of data, with extra fields
        df = pd.DataFrame({'y_true':y_true, 'y_score':y_score})
        for f in self.aux_fields:
            df[f] = aux_data[f] * df['y_true'] # aux fields only counted for positive class
        df = df.sort_values('y_score').reset_index(drop=True)
        
        # initialize true positive, false positive, false negative, and true negative counts
        tp = df['y_true'].sum()
        fp = len(df) - tp
        fn = 0
        tn = 0
        aux_tp = {f: df[f].sum() for f in self.aux_fields} # tp sum for aux fields
        idx_lower = 0 # threshold lower bound index
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
            idx_upper = df['y_score'].searchsorted(threshold + increment) # index of first value >= threshold + increment
            df_subset = df.loc[idx_lower: idx_upper-1, ['y_true'] + self.aux_fields] # records in idx range
            df_pos = df_subset.loc[df_subset['y_true'] == 1] # postive class records in idx range
            pos_count = len(df_pos) # number of positive cases in threshold range
            neg_count = idx_upper - idx_lower - pos_count # number of negative cases in threshold range
            tp -= pos_count
            fp -= neg_count 
            fn += pos_count
            tn += neg_count
            for k, v in aux_tp.items(): # update tp sum for aux fields
                aux_tp[k] -= df_pos[k].sum()
            idx_lower = idx_upper

        # convert to dataframe
        performance = pd.DataFrame.from_records(performance)
        columns = ['threshold', 'tp', 'fp', 'fn', 'tn'] + self.aux_fields
        performance = performance.reindex(columns=columns) # reorder columns

        # combine fields
        performance['precision'] = performance['tp'] / (performance['tp'] + performance['fp'])
        performance['recall'] = performance['tp'] / (performance['tp'] + performance['fn'])
        performance['fp_to_tp'] = performance['fp'] / performance['tp']
        performance['accuracy'] = (performance['tp'] + performance['tn']) / len(df)
        performance['F1'] = 2 * performance['tp'] / (2 * performance['tp'] + performance['fp'] + performance['fn'])
        performance['specificity'] = performance['tn'] / (performance['tn'] + performance['fp'])
        performance['fpr'] = 1 - performance['specificity']
        performance.reset_index(drop=True)
        
        # add weighted recall for aux fields & rename aux fields
        for f in self.aux_fields:
            performance[f"{f}_weighted_recall"] = performance[f] / performance.at[0, f]
        performance.rename(columns={f: f"{f}_tp_sum" for f in self.aux_fields}, inplace=True)
        
        # save file
        performance.to_csv(f'{self.tables_subdir}/threshold_vs_metrics_{dataset_name}.csv', index=False)
        self.logger.info(f'Generated threshold performance table ({dataset_name} data)')

    def _metrics_table(self, y_true, y_score, dataset_name, roc_auc, precision_recall_auc, average_precision):
        """Generate table of metrics for binary classification model evaluation."""

        log_loss_ = log_loss(y_true, y_score) # cross entropy
        brier_score = brier_score_loss(y_true, y_score)
        metrics = [
            ('ROC AUC', roc_auc),
            ('Precision Recall AUC', precision_recall_auc),
            ('Average Precision', average_precision),
            ('Log Loss (Cross Entropy)', log_loss_),
            ('Brier Score Loss', brier_score),
        ]
        df = pd.DataFrame(metrics, columns=['Metric', 'Value'])
        df.to_csv(f'{self.tables_subdir}/metrics_{dataset_name}.csv', index=False)
        self.logger.info(f'Generated metrics table ({dataset_name} data)')

    def ks_statistic(self):
        """Generate Kolmogorov-Smirnov (KS) Statistic."""

        # intialize output table
        performance = []

        # generate KS Statistic for each dataset
        for X, y_true, dataset_name in self.datasets:
            y_score = self.model.predict_proba(X)[:,1]
            ks_stat, p_value = ks_2samp(y_score[y_true==0], y_score[y_true==1])
            row = {'dataset': dataset_name, 'ks': ks_stat, 'p-value': p_value}
            performance.append(row)

        # convert to dataframe
        performance = pd.DataFrame.from_records(performance)
        columns = ['threshold', 'tp', 'fp', 'fn', 'tn'] + self.aux_fields
        performance = performance.reindex(columns=['dataset', 'ks', 'p-value']) # reorder columns

        # save output to csv
        performance.to_csv(self.tables_subdir/f"ks_statistic", index=False)
        self.logger.info(f'Generated Kolmogorov-Smirnov (KS) Statistic table')