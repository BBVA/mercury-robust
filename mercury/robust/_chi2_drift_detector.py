import numpy as np

from scipy.stats import chi2_contingency

from ._base_batch_drift_detector import BaseBatchDriftDetector


class Chi2Drift(BaseBatchDriftDetector):
    """
    The Chi2Drift class  allows us to detect drift in the source dataset distributions `distr_src` and the
    target dataset distributions `distr_target` by performing a chi-square test of independence of variables in a
    contingency table for each feature. It does it separately for each feature, calculating the statistic and the
    p-value. The source and target distributions are specified when creating the object. Then, the method
    `calculate_drift()` is used to obtain a dict of metrics.

    Args:
        distr_src (list of np.arrays):
            The source dataset distributions. It's a list of np.arrays, where each np.array represents the counts of
            each bin for that feature. The features and its bins must be in the same order than the `distr_target`
        distr_target (list of np.arrays):
            The target dataset distributions. It's a list of np.arrays, where each np.array represents the counts of
            each bin for that feature. The features and its bins must be in the same order than the `distr_src`
        features (list):
            The name of the features of `X_src` and `X_target`.
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

    """

    def __init__(self, distr_src=None, distr_target=None, features=None, correction="bonferroni", p_val=0.05):
        super().__init__(
            distr_src=distr_src, distr_target=distr_target, features=features, correction=correction, p_val=p_val
        )

    def _remove_zero_cols(self, arr):
        """ Removes cols that have value zero in all the rows for that column"""
        idx = np.argwhere(np.all(arr[..., :] == 0, axis=0))
        arr = np.delete(arr, idx, axis=1)
        return arr

    def _calculate_chi2(self):
        """
        Calculates a chi-square test
        """

        self._check_distributions()

        n_features = len(self.distr_src)
        p_val = np.zeros(n_features, dtype=np.float32)
        dist = np.zeros_like(p_val)
        for f in range(n_features):
            # Construct contingency table
            table = np.array([self.distr_src[f], self.distr_target[f]])
            table = self._remove_zero_cols(table)
            # Calculate chi2 contingency test
            chi2, p, dof, ex = chi2_contingency(table)
            dist[f] = chi2
            p_val[f] = p

        return p_val, dist

    def calculate_drift(self):
        """
        Computes drift by performing a chi-square test of independence between the source dataset distributions
        `distr_src` and the target dataset distributions `distr_target`. It performs the test individually for each
        feature and obtains the p-values and the value of the statistic for each.
        The returned dictionary contains the next metrics:

        - p_vals: np.array of the p-values returned by the Chi2 test for each feature.
        - scores: np.array of the chi2 statistics for each feature.
        - score: average of all the scores.
        - drift_detected: final decision if drift is detected. If any of the p-values of the features is lower than
            the threshold, then this is set to 1. Otherwise is 0. This will depend on the specified `p_val` and
            `correction` values.
        - threshold: final threshold applied to decide if drift is detected. It depends on the specified
            `p_val` and `correction` values.

        Returns:
            (dict): Dictionary with the drift metrics as specified.
        """

        p_vals_test, distances = self._calculate_chi2()
        drift_pred, threshold = self._apply_correction(p_vals_test)
        self.drift_metrics = {
            "p_vals": p_vals_test,
            "score": np.mean(distances),
            "scores": distances,
            "drift_detected": drift_pred,
            "threshold": threshold
        }
        return self.drift_metrics