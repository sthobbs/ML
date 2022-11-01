
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
import pickle
from typing import List, Optional, Union
from experiment.model_evaluate import ModelEvaluate


class ModelCalibrate():
    """Calibrate model so that it outputs a probability."""

    def __init__(self,
                 model: BaseEstimator,
                 datasets: List[tuple[
                    Union[np.ndarray, pd.core.series.Series],
                    Union[np.ndarray, pd.core.series.Series],
                    str]],
                 output_dir: Union[str, Path],
                 logger: Optional[logging.Logger] = None) -> None:
        """
        Parameters
        ----------
            model :
                scikit-learn classifier with a .predict_proba() method.
            datasets :
                List of (X, y, dataset_name) triples.
                e.g. [(X_train, y_train, 'Train'), (X_val, y_val, 'Validation'), (X_test, y_test, 'Test')]
            output_dir : str
                string path to folder where output will be written.
            logger : logging.Logger, optional
                logger.
        """

        self.model = model
        self.datasets = datasets

        # Make directories
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set plot context
        self.plot_context = 'seaborn-darkgrid'

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

    def calibrate(self, train_dataset_name: str, calibration_type: str = 'logistic') -> None:
        """
        Calibrate a binary classification model to output probability of true positive.

        Parameters
        ----------
            train_dataset_name :  str
                The name of the dataset to use to train the calibration model (usually 'validation')
            calibration_type : {'isotonic', 'logistic'}, default = 'logistic'
                The type of calibration model to fit
        """

        # check for invalid input
        valid_types = {'isotonic', 'logistic'}
        assert calibration_type in valid_types, \
            f"calibration_type must be in {valid_types}, not {calibration_type}"

        # calibrate model
        idx = [i for i, (_, _, name) in enumerate(self.datasets) if name == train_dataset_name][0]
        y_score = self.model.predict_proba(self.datasets[idx][0])[:, 1]
        y_true = self.datasets[idx][1]
        self.calibrator = Calibrator(calibration_type)
        self.calibrator.fit(y_score, y_true)

    def save_model(self) -> None:
        """Save calibration model object to file."""

        model_dir = self.output_dir / 'model'
        model_dir.mkdir(parents=True, exist_ok=True)
        output_path = model_dir/'calibration_model.pkl'
        self.calibrator.save_model(output_path)

    def evaluate(self, bin_type: str = 'uniform', n_bins: int = 5, increment: float = 0.01) -> None:
        """
        Generate performance metrics for the caalibrated model.

        Parameters
        ----------
            bin_type : {'uniform', 'quantile'}, default = 'uniform'
                Strategy used to define the widths of the bins for the calibration plots
            n_bins : int > 1, default = 5
                Number of bins to discretize the [0, 1] interval in the calibration plots
            increment : float between 0 and 1,
                the threshold increment used for performance evaluation of the calibrated model
        """

        # check for invalid input
        valid_types = {'uniform', 'quantile'}
        assert bin_type in valid_types, f"bin_type must be in {valid_types}, not {bin_type}"
        assert isinstance(n_bins, int), f"n_bins must be an integer, not {n_bins}"
        assert n_bins > 1, f"n_bins must be > 1, not {bin_type}"

        # make directories and close existing figures
        comparison_dir = self.output_dir / 'comparison'
        comparison_dir.mkdir(parents=True, exist_ok=True)
        performance_dir = self.output_dir / 'performance'
        performance_dir.mkdir(exist_ok=True)

        plt.close('all')

        # evaluate calibration on each dataset
        calibrated_scores = []
        for X, y_true, dataset_name in self.datasets:

            y_score = self.model.predict_proba(X)[:, 1]
            y_cal = self.calibrator.predict_proba(y_score)  # calibrated score
            calibrated_scores.append(y_cal)

            # plot calibration curves (i.e. reliability diagrams)
            output_path = f'{comparison_dir}/calibration_curves_{dataset_name}.png'
            self._plot_calibration_curve(y_true, y_score, y_cal, dataset_name, output_path, bin_type, n_bins)

            # plot performance comparisons
            output_path = f'{comparison_dir}/compare_{dataset_name}.png'
            self._plot_comparisons(y_true, y_score, y_cal, output_path)

            # generate calibration mapping plot and table
            fig_path = f'{comparison_dir}/score_mapping_{dataset_name}.png'
            table_path = f'{comparison_dir}/mapping_table_{dataset_name}.csv'
            self._calibration_mapping(fig_path, table_path)

        # Evaluate model on calibrated score and generate performance charts
        datasets = [(y_cal, y, n) for y_cal, (X, y, n) in zip(calibrated_scores, self.datasets)]
        model_eval = ModelEvaluate(self.calibrator, datasets, performance_dir, logger=self.logger)
        model_eval.binary_evaluate(increment)

    def _plot_calibration_curve(self,
                                y_true,
                                y_score,
                                y_cal,
                                dataset_name: str,
                                output_path: str,
                                bin_type: str = 'uniform',
                                n_bins: int = 5) -> None:
        """
        plot calibration curve (i.e. reliability diagram).

        Parameters
        ----------
            y_true : array-like of shape (n_sample,)
                ground truth labels.
            y_score : array-like of shape (n_sample,)
                model scores.
            y_score : array-like of shape (n_sample,)
                model scores after passing through a calibration function.
            dataset_name : str
                the name of the dataset.
            output_path : str
                string path to location where output plot will be written.
            bin_type : {'uniform', 'quantile'}, default = 'uniform'
                Strategy used to define the widths of the bins for the calibration plots
            n_bins : int > 1, default = 5
                Number of bins to discretize the [0, 1] interval in the calibration plots
        """

        # check for invalid input
        valid_types = {'uniform', 'quantile'}
        assert bin_type in valid_types, f"bin_type must be in {valid_types}, not {bin_type}"
        assert isinstance(n_bins, int), f"n_bins must be an integer, not {n_bins}"
        assert n_bins > 1, f"n_bins must be > 1, not {bin_type}"

        with plt.style.context(self.plot_context):

            fontsize = 20
            plt.figure(figsize=(20, 10))

            # plot uncalibrated score
            prob_true, prob_score = calibration_curve(y_true, y_score, n_bins=n_bins, strategy=bin_type)
            plt.subplot(1, 2, 1)
            plt.plot(prob_score, prob_true)
            plt.plot([[0, 0], [1, 1]])
            plt.title(f"Uncalibrated Curve for {dataset_name} Dataset", fontsize=fontsize)
            plt.xlabel("Raw Score", fontsize=fontsize)
            plt.ylabel("Probability of TP", fontsize=fontsize)

            # plot calibrated score
            prob_true, prob_cal_score = calibration_curve(y_true, y_cal, n_bins=n_bins, strategy=bin_type)
            plt.subplot(1, 2, 2)
            plt.plot(prob_cal_score, prob_true)
            plt.plot([[0, 0], [1, 1]])
            plt.title(f"Uncalibrated Curve for {dataset_name} Dataset", fontsize=fontsize)
            plt.xlabel("Raw Score", fontsize=fontsize)
            plt.ylabel("Probability of TP", fontsize=fontsize)
            plt.savefig(output_path)
            self.logger.info(f'Plotted Calibration Curves ({dataset_name} data)')

        plt.close()

    def _plot_comparisons(self, y_true, y_score, y_cal, output_path: str) -> None:
        """
        plot performance comparison charts.

        Parameters
        ----------
            y_true : array-like of shape (n_sample,)
                ground truth labels.
            y_score : array-like of shape (n_sample,)
                model scores.
            y_score : array-like of shape (n_sample,)
                model scores after passing through a calibration function.
            output_path : str
                string path to location where output plot will be written.
        """
        # raw score
        fpr, tpr, _ = roc_curve(y_true, y_score)
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)

        # calibrated score
        fpr_cal, tpr_cal, _ = roc_curve(y_true, y_score)
        precision_cal, recall_cal, thresholds_cal = precision_recall_curve(y_true, y_cal)

        with plt.style.context(self.plot_context):

            plt.figure(figsize=(20, 5))
            fontsize = 15

            # plot ROC
            plt.subplot(1, 4, 1)
            plt.plot(fpr, tpr, 'g-', label="Raw")
            plt.plot(fpr_cal, tpr_cal, 'b--', label='Calibrated')
            plt.xlim(-0.05, 1.05)
            plt.ylim(-0.05, 1.05)
            plt.xlabel('FPR', fontsize=fontsize)
            plt.ylabel('TPR', fontsize=fontsize)
            plt.title('ROC', fontsize=fontsize)
            plt.legend(loc="lower right")

            # plot Pecision vs Recall
            plt.subplot(1, 4, 2)
            plt.plot(recall, precision, 'g-', label="Raw")
            plt.plot(recall_cal, precision_cal, 'b--', label='Calibrated')
            plt.xlim(-0.05, 1.05)
            plt.ylim(-0.05, 1.05)
            plt.xlabel('Recall', fontsize=fontsize)
            plt.ylabel('Precision', fontsize=fontsize)
            plt.title('Precision vs Recall', fontsize=fontsize)
            plt.legend(loc="upper right")

            # plot Recall vs Score
            plt.subplot(1, 4, 3)
            plt.plot(thresholds, recall[:-1], 'g-', label="Raw")
            plt.plot(thresholds_cal, recall_cal[:-1], 'b--', label='Calibrated')
            plt.xlim(-0.05, 1.05)
            plt.ylim(-0.05, 1.05)
            plt.xlabel('Score', fontsize=fontsize)
            plt.ylabel('Recall', fontsize=fontsize)
            plt.title('Recall vs Score', fontsize=fontsize)
            plt.legend(loc="upper right")

            # plot Pecision vs Score
            plt.subplot(1, 4, 4)
            plt.plot(thresholds, precision[:-1], 'g-', label="Raw")
            plt.plot(thresholds_cal, precision_cal[:-1], 'b--', label='Calibrated')
            plt.xlim(-0.05, 1.05)
            plt.ylim(-0.05, 1.05)
            plt.xlabel('Score', fontsize=fontsize)
            plt.ylabel('Precision', fontsize=fontsize)
            plt.title('Precision vs Score', fontsize=fontsize)
            plt.legend(loc="lower right")
            plt.savefig(output_path)
        plt.close()

    def _calibration_mapping(self, fig_path: str, table_path: str, increment: float = 0.0001) -> None:
        """
        plot performance comparison charts.

        Parameters
        ----------
            increment : float between 0 and 1, default = 0.0001
                The size of increment of values to map
            fig_path : str
                string path to location where output plot will be written.
            table_path : str
                string path to location where output table will be written.
        """

        # score calibration mapping
        increment = 0.0001
        score = np.arange(0, 1, increment)
        score_cal = self.calibrator.predict_proba(score)
        plt.plot(score, score_cal)
        plt.xlabel("Score")
        plt.ylabel("Calibrated Score")
        plt.title("Raw Score to Calibrated Score Mapping")
        plt.savefig(fig_path)
        plt.close()
        df = pd.DataFrame({"score": score, "calibrated_score": score_cal})
        df.to_csv(table_path, index=False)


class Calibrator():
    """Wrapper for calibration model"""

    def __init__(self, calibration_type: str = 'isotonic') -> None:
        """
        Parameters
        ----------
            calibration_type : {'isotonic', 'logistic'}, default = 'logistic'
                The type of calibration model to fit
        """

        # check for invalid input
        valid_types = {'isotonic', 'logistic'}
        assert calibration_type in valid_types, \
            f"calibration_type must be in {valid_types}, not {calibration_type}"

        # make calibration model
        if calibration_type == 'isotonic':
            self.model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        elif calibration_type == 'logistic':
            self.model = LogisticRegression()

        self.calibration_type = calibration_type

    def fit(self, y_score, y_true) -> None:
        """
        Fit calibration model.

        Parameters
        ----------
            y_score : array-like of shape (n_sample,)
                model scores.
            y_true : array-like of shape (n_sample,)
                ground truth labels.
        """

        if y_score.ndim == 1:
            y_score = y_score.reshape(-1, 1)  # make into 2d array for .fit
        self.model.fit(y_score, y_true)

    def predict_proba(self, y_score):
        """
        Calibrate scores.

        Parameters
        ----------
            y_score : array-like of shape (n_sample,)
                model scores.
        """

        if y_score.ndim == 1:
            y_score = y_score.reshape(-1, 1)  # make into 2d array
        if self.calibration_type == 'isotonic':
            return self.model.transform(y_score)
        elif self.calibration_type == 'logistic':
            return self.model.predict_proba(y_score)[:, 1]

    def save_model(self, output_path: Union[str, Path]) -> None:
        """
        Save calibration model object to file.

        Parameters
        ----------
            output_path : str
                string path to location where calibration model will be written.
        """

        with open(output_path, 'wb') as file:
            pickle.dump(self.model, file)
        # TODO (?): also save pmml
