import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod


class BaseBatchDriftDetector(ABC):
    """
    Base class for batch drift detection.
    The class can be extended to create a method for batch drift detection over a source dataset `X_src` and target
    dataset `X_target`, or a source dataset distributions `distr_src` and a target dataset distributions `distr_target`
    The main method to implement is the  `calculate_drift()` method.
    """

    def __init__(
        self, X_src=None, X_target=None, distr_src=None, distr_target=None, features=None, correction="bonferroni", p_val=0.05
    ):
        self.X_src = X_src
        self.X_target = X_target
        self.distr_src = distr_src
        self.distr_target = distr_target
        self.features = features
        self.correction = correction
        self.p_val = p_val
        self.drift_metrics = None

    def set_datasets(self, X_src, X_target):
        self.X_src = X_src
        self.X_target = X_target
        self.drift_metrics = None

    def set_distributions(self, distr_src, distr_target):
        self.distr_src = distr_src
        self.distr_target = distr_target
        self.drift_metrics = None

    def _apply_correction(self, p_vals_test):
        """
        Applies correction of the p-value.
        """

        if self.p_val is None:
            raise ValueError("p_val value is not setted")

        n_tests = len(p_vals_test)
        if self.correction is None:
            threshold = self.p_val
            drift_pred = p_vals_test < self.p_val
        elif self.correction == 'bonferroni':
            threshold = self.p_val / n_tests
            drift_pred = (p_vals_test < threshold).any()
        else:
            raise ValueError("Invalid correction type")
        return drift_pred, threshold

    def _check_datasets(self):
        if (self.X_src is None) or (self.X_target is None):
            raise ValueError("X_src and X_target are needed to compute the metric")

        if (not isinstance(self.X_src, np.ndarray)) or (not isinstance(self.X_target, np.ndarray)):
            raise ValueError("X_src and X_target must be numpy arrays")

        if (len(self.X_src.shape) != 2) or (len(self.X_target.shape) != 2):
            raise ValueError("X_src and X_target must be 2-dimensional np.arrays")

        if self.X_src.shape[1] != self.X_target.shape[1]:
            raise ValueError("X_src and X_target must have the same number of features")

    def _check_distributions(self):
        if (self.distr_src is None) or (self.distr_target is None):
            raise ValueError("distr_src and distr_target are need to compute the metric")

        if len(self.distr_src) != len(self.distr_target):
            raise ValueError("distr_src and distr_target  must have the same number of features")

        for i in range(len(self.distr_src)):
            if len(self.distr_src[i]) != len(self.distr_target[i]):
                raise ValueError("distr_src and distr_target have different dimensions for feature ", i)

    @abstractmethod
    def calculate_drift(self):
        """
        Method to implement in extended classes
        """
        raise NotImplementedError()

    def _get_index_feature(self, idx_feature=None, name_feature=None):
        if idx_feature is None and name_feature is None:
            raise ValueError("idx_feature or name_feature must be specified")

        if (idx_feature is not None) and (name_feature is not None):
            raise ValueError("Only one of the parameters idx_feature or name_feature must be specified")

        if name_feature is not None:
            if self.features is not None:
                return self.features.index(name_feature)
            else:
                raise ValueError("To use the name_feature the feature attribute of the class must be specified first")

        return idx_feature

    def plot_distribution_feature(self, idx_feature=None, name_feature=None, hist=True, ax=None, figsize=(8, 4)):
        """
        Plots the distribution of a given feature for the source and the target datasets.
        The feature can be indicated by the index using the `idx_feature` parameter, or by the name using the
        `name_feature` parameter (only possible if attribute `features` was specified.
        Args:
            idx_feature (int):
                index of the feature
            name_feature (string):
                name of the feature
            hist (boolean):
                whether to show the histograms in the plot or not.
            ax (matplotlib.axes._subplots.AxesSubplot):
                Axes object on which the data will be plotted. If not specified it creates one.
            figsize (tuple):
                figsize to use if `ax` is not specified.
        Returns:
            The axes
        """
        self._check_datasets()
        idx_feature = self._get_index_feature(idx_feature, name_feature)
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        axes = sns.distplot(self.X_src[:, idx_feature], hist=hist, ax=ax, label="dataset source")
        axes = sns.distplot(self.X_target[:, idx_feature], hist=hist, ax=ax, label="dataset target")
        axes.legend()
        return axes

    def plot_histograms_feature(self, idx_feature=None, name_feature=None, figsize=(8, 4)):
        """
        Plots the histograms of a given feature for the source and the target distributions.
        The feature can be indicated by the index using the `idx_feature` parameter, or by the name using the
        `name_feature` parameter (only possible if attribute `features` was specified.
        Args:
            idx_feature (int):
                index of the feature
            name_feature (string):
                name of the feature
            figsize (tuple):
                figsize to use if `ax` is not specified.
        Returns:
            The axes
        """

        self._check_distributions()
        idx_feature = self._get_index_feature(idx_feature, name_feature)
        # Plot distributions using histograms in different plots
        fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=False)

        axes[0].bar(range(len(self.distr_src[idx_feature])), self.distr_src[idx_feature], width=0.8)
        axes[1].bar(range(len(self.distr_target[idx_feature])), self.distr_target[idx_feature], width=0.8)
        axes[0].set_title("histogram source")
        axes[1].set_title("histogram target")
        return axes

    def plot_distribution_drifted_features(self, figsize=(8, 4)):
        """
        Plots the distributions of the features that are considered to have drift. The method `calculate_drift`
        needs to be called first.

        Args:
            figsize (tuple):
                figsize to use if `ax` is not specified.

        """

        drifted_features = self.get_drifted_features(return_as_indices=True)
        for idx in drifted_features:
            ax = self.plot_distribution_feature(idx_feature=idx, figsize=figsize)
            if self.features:
                ax.set_title(self.features[idx])
            else:
                ax.set_title(idx)

    def plot_histograms_drifted_features(self, figsize=(8, 4)):
        """
        Plots the histograms of the features that are considered to have drift. The method `calculate_drift`
        needs to be called first.

        Args:
            figsize (tuple):
                figsize to use if `ax` is not specified.

        """

        drifted_features = self.get_drifted_features(return_as_indices=True)
        for idx in drifted_features:
            axes = self.plot_histograms_feature(idx_feature=idx, figsize=figsize)
            if self.features:
                axes[0].set_title("histogram source " + self.features[idx])
                axes[1].set_title("histogram target " + self.features[idx])
            else:
                axes[0].set_title("histogram source " + str(idx))
                axes[1].set_title("histogram target " + str(idx))

    def get_drifted_features(self, return_as_indices=False):
        """
        Gets the features that are considered to have drift. The method `calculate_drift` needs to be called first.
        The parameter `return_as_indices` controls whether to return the features as indices or names.

        Args:
            return_as_indices (boolean): whether to return the features as indices or names.

        Returns:
            (list): features with drift.
        """
        if "p_vals" in self.drift_metrics.keys() and "threshold" in self.drift_metrics.keys():
            drifted_features = []
            for i in range(len(self.drift_metrics["scores"])):
                if self.drift_metrics["p_vals"][i] < self.drift_metrics["threshold"]:
                    drifted_features.append(i)
        else:
            raise ValueError(
                "individual p-values need to be calculated to obtain features features. "
                "Call calculate_drit() first or use a drift method that calculates p-values per feature.")

        if return_as_indices:
            return drifted_features
        else:
            if self.features is None:
                raise ValueError("Feature indices are not setted")
            else:
                return [self.features[idx] for idx in drifted_features]

    def plot_feature_drift_scores(self, top_k=None, ax=None, figsize=(8, 4)):
        """
        Plots a summary of the scores for each feature. The meaning of the score depends on the subclass that is used.

        Args:
            top_k (int):
                If specified, plots only the top_k features with higher scores.
            ax (matplotlib.axes._subplots.AxesSubplot):
                Axes object on which the data will be plotted. If not specified it creates one.
            figsize (tuple):
                Size of the plotted figure
        """
        if self.drift_metrics is None:
            raise ValueError("Drift must be calculated first to calculate feature drift scores")

        if "scores" not in self.drift_metrics.keys():
            raise ValueError("This drift methods doesn't have the drift scores calculated yet")

        if self.features is None:
            # if feature names are not given, we use the indices
            features = list(range(len(self.drift_metrics["scores"])))
        else:
            features = self.features

        # Sort features
        sorted_pairs = sorted(zip(self.drift_metrics["scores"], features))
        sorted_importances, sorted_features = [list(tuple) for tuple in zip(*sorted_pairs)]

        # Get top k if specified
        if top_k:
            sorted_features = sorted_features[-top_k:]
            sorted_importances = sorted_importances[-top_k:]

        # Create plot
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.barh([f for f in sorted_features], [imp for imp in sorted_importances])
        ax.set_title("feature drift scores")

        return ax