import numpy as np

from ._base_batch_drift_detector import BaseBatchDriftDetector
from ._metrics import hellinger_distance, jeffreys_divergence, psi


class HistogramDistanceDrift(BaseBatchDriftDetector):
    """
    The HistogramDistanceDrift class allows us to detect drift in the source dataset distributions `distr_src` and the
    target dataset distributions `distr_target` by calculating distances between the distributions. For each feature,
    the distance between its distributions in `distr_src` and `distr_target` is calculated. The distance function to
    use is indicated by the `distance_metric` parameter. For each feature, a p-value is also calculated. This is
    calculated by doing a permutation test, where data from `distr_src` and `distr_target` are put together and sampled
    to calculate a new simulated distance `n_permutations` times. The p-value then is obtained by computing
    the fraction of simulated distances equal or greater than the observed distance. The source and target distributions
    are specified when creating the object. Then, the method `calculate_drift()` is used to obtain a dict of metrics.

    Args:
        distr_src (list of np.arrays):
            The source dataset distributions. It's a list of np.arrays, where each np.array represents the counts of
            each bin for that feature. The features and its bins must be in the same order than the `distr_target`
        distr_target (list of np.arrays):
            The target dataset distributions. It's a list of np.arrays, where each np.array represents the counts of
            each bin for that feature. The features and its bins must be in the same order than the `distr_src`
        features (list):
            The name of the features of `X_src` and `X_target`.
        distance_metric (string):
            The distance metric to use. Available metrics are "hellinger", "jeffreys" or "psi".
            Default value is "hellinger"
        p_val (float):
            Threshold to be used with the p-value obtained to decide if there is drift or not (affects
            `drift_detected` metric when calling `calculate_drift()`)
            Lower values will be more conservative when indicating that drift exists in the dataset.
            Default value is 0.05.
        correction (string):
            Since multiple tests are performed, this parameter controls whether to perform a correction of the p-value
            or not (affects `drift_detected` metric when calling `calculate_drift()`). If None, it doesn't perform
            a correction
            Default value is "bonferroni", which makes the bonferroni correction taking into account the number of tests
        n_permutations (int):
            The number of permutations to perform to calculate the p-value. A higher number will calculate a more
            accurate p-value, but it will be computationally more expensive.
            Default value is 50

    """

    def __init__(
        self, distr_src=None, distr_target=None, features=None,
        correction=None, p_val=None, n_permutations=50, distance_metric="hellinger"
    ):
        super().__init__(
            distr_src=distr_src, distr_target=distr_target, features=features, correction=correction, p_val=p_val
        )
        self.n_permutations = n_permutations
        self.distance_metric = distance_metric

    def _generate_bin_samples_from_added_histogram(self, hist_1, hist_2):
        """
        Adds the bin counts of two histograms together.
        Then, it transforms the counts to "bin_samples"
        """
        added_hist_1 = hist_1 + hist_2
        bins = []
        for i in range(len(added_hist_1)):
            bins += int(added_hist_1[i]) * [i]
        return np.array(bins)

    def _permutation_test_from_histograms(
            self, hist_1, hist_2, n_permutations, func_statistic, obs_statistic, sample_with_replacement=False
    ):
        """
        Performs a permutation test for histograms.
        """

        self._check_distributions()

        bin_samples = self._generate_bin_samples_from_added_histogram(hist_1, hist_2)
        n_hist_1 = int(np.sum(hist_1))
        metrics = []
        for _ in range(n_permutations):
            if sample_with_replacement:
                pool = bin_samples[np.random.randint(0, len(bin_samples), len(bin_samples))]
            else:
                pool = bin_samples
            # Split bins randomly with the same size of the original samples
            idx_hist_1 = np.random.choice(len(pool), size=n_hist_1, replace=False)
            idx_hist_2 = np.ones(len(pool), dtype=bool)
            idx_hist_2[idx_hist_1] = False
            new_bins_hist_1 = pool[idx_hist_1]
            new_bins_hist_2 = pool[idx_hist_2]
            # Count how many samples we have on each bin
            bin_counts_1 = dict(enumerate(np.bincount(new_bins_hist_1), 0))
            bin_counts_2 = dict(enumerate(np.bincount(new_bins_hist_2), 0))
            # Convert from sampled bins to histogram (counts)
            new_hist_1 = np.zeros_like(hist_1)
            new_hist_2 = np.zeros_like(hist_2)
            for k, v in bin_counts_1.items():
                new_hist_1[k] = v
            for k, v in bin_counts_2.items():
                new_hist_2[k] = v
            # Compute distance with this new sample
            metrics.append(func_statistic(new_hist_1, new_hist_2))
        # Compute fraction of simulated distances equal or greater than the observed distance
        p_val_estimated = np.mean(np.array(metrics) >= obs_statistic)
        return p_val_estimated

    def _calculate_histogram_distance(self):

        self._check_distributions()
        if self.distance_metric.lower() == "hellinger":
            distance_func = hellinger_distance
        elif self.distance_metric.lower() == "jeffreys":
            distance_func = jeffreys_divergence
        elif self.distance_metric.lower() == "psi":
            distance_func = psi
        else:
            raise ValueError("The specified distance metric doesn't exist. Specify 'hellinger', 'jeffreys' or 'psi'")

        n_features = len(self.distr_src)
        p_vals = np.zeros(n_features, dtype=np.float32)
        distances = np.zeros_like(p_vals)
        for i in range(n_features):
            # Compute observed stiance between the source distribution and the target distribution
            distances[i] = distance_func(self.distr_src[i], self.distr_target[i])
            # Make a permutation test to obtain p-value
            p_vals[i] = self._permutation_test_from_histograms(
                self.distr_src[i],
                self.distr_target[i],
                self.n_permutations,
                distance_func,
                distances[i])

        return p_vals, distances

    def calculate_drift(self):
        """
        Computes drift by calculating histogram-based distances between the source dataset distributions `distr_src`
        and the target dataset distributions `distr_target`. It calculates the distance individually for each feature
        and obtains the p-values and distance for each.
        The returned dictionary contains the next metrics:

        - p_vals: np.array of the p-values returned by the permutation test for each feature.
        - scores: np.array of the distances for each feature.
        - score: average of all the distances.
        - drift_detected: final decision if drift is detected. If any of the p-values of the features is lower than
            the threshold, then this is set to 1. Otherwise is 0. This will depend on the specified `p_val` and
            `correction` values.
        - threshold: final threshold applied to decide if drift is detected. It depends on the specified
            `p_val` and `correction` values.

        Returns:
            (dict): Dictionary with the drift metrics as specified.
        """
        p_vals_estimated, distances = self._calculate_histogram_distance()
        drift_pred, threshold = self._apply_correction(p_vals_estimated)
        self.drift_metrics = {
            "p_vals": p_vals_estimated,
            "score": np.mean(distances),
            "scores": distances,
            "drift_detected": drift_pred,
            "threshold": threshold
        }
        return self.drift_metrics