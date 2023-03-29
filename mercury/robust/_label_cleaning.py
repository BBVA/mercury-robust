import numpy as np
from sklearn.linear_model import LogisticRegression
from cleanlab.count import estimate_cv_predicted_probabilities, compute_confident_joint
from cleanlab.filter import find_label_issues
from typing import Tuple


def get_confident_joint(
    X: np.array,
    y: np.array,
    psx: np.array = None,
    n_folds: int = 5,
    clf: "BaseEstimator" = None,  # noqa: F821
    seed: int = None
) -> np.array:

    """
    Calculates the confident joint matrix, which contains the counts of each one of the samples of a class that could belong to
    each one of the other classes. The diagonal entries of the matrix count correct labels and non-diagonal possible asymmetric
    label error counts. It uses the cleanlab library to compute it. More information in the Confident Learning paper:
    https://arxiv.org/pdf/1911.00068.pdf

    Args:
        X (np.array): array of features of your dataset
        y (np.array): array of (noisy) labels of your dataset, i.e. some labels may be erroneous.
        psx (np.array): matrix with (noisy) model-predicted probabilities for each of the N samples of X. It should be
            computed using cross-validation. If None, then it will be computed using `n_folds` and `clf`
        n_folds (int): number of cross-validation folds used to compute out-of-sample probabilities for each sample
        clf (BaseEstimator): Classifier to calculate out-of-sample probabilities. If None, LogisticRegression is used.
        seed (int): seed to use.

    Returns:
        Confident joint matrix

    """

    if psx is None:
        clf = LogisticRegression() if clf is None else clf
        psx = estimate_cv_predicted_probabilities(
            X=X,
            labels=y,
            clf=clf,
            cv_n_folds=n_folds,
            seed=seed,
        )

    conf_joint = compute_confident_joint(
        labels=y,
        pred_probs=psx
    )

    return conf_joint


def get_label_issues(
    X: np.array,
    y: np.array,
    n_folds: int = 5,
    clf: "BaseEstimator" = None,  # noqa: F821
    frac_noise: float = 1.0,
    num_to_remove_per_class: list = None,
    prune_method: str = 'prune_by_noise_rate',
    sorted_index_method: str = None,
    n_jobs: int = None,
    seed: int = None
) -> Tuple[np.array, np.array]:
    """
    Obtains samples from a dataset which can contain label issues, where issues could be samples that are incorrectly
    labeled, samples that could belong to more than one label category or some other issue. Internally, it uses the
    cleanlab library, which is based on confident learning:
    https://arxiv.org/pdf/1911.00068.pdf
    It also returns the confident joint matrix, which contains the counts of each one of the samples of a class that
    could belong to each one of the other classes


    Args:
        X (np.array): array of features of your dataset
        y (np.array): array of (noisy) labels of your dataset, i.e. some labels may be erroneous.
        n_folds (int): number of cross-validation folds used to compute out-of-sample probabilities for each sample
        clf (BaseEstimator): Classifier to calculate out-of-sample probabilities. If None, LogisticRegression is used.
        num_to_remove_per_class (list): Only used when `prune_method == 'prune_by_class'`. Indicates how many
            samples return for each class (returning the most likely mislabeled samples first).
        prune_method (str): Method used for filtering/pruning out the label issues.
            Possible Values: 'prune_by_class', 'prune_by_noise_rate', 'both', 'confident_learning' or 'predicted_neq_given'.
            Default value is 'prune_by_noise_rate'.
            You can check cleanlab documentation for further details (filter_by argument):
            https://docs.cleanlab.ai/master/cleanlab/filter.html#cleanlab.filter.find_label_issues
        sorted_index_method (str): Indicates how to sort the returned samples with issues. If `None` returns a boolean mask
            (`True` if example at index is label error). If not `None`, returns an array of the label error indices (instead
            of a boolean mask) where error indices are ordered.
            Possible values: None, 'self_confidence', 'normalized_margin', 'confidence_weighted_entropy'
            Default value is `None`.
            You can check cleanlab documentation for further details (return_indices_ranked_by argument):
            https://docs.cleanlab.ai/master/cleanlab/filter.html#cleanlab.filter.find_label_issues
        frac_noise (float): When frac_noise=1.0, return all "confident" estimated noise indices. Value in range (0,1]
            will return fewer samples. It only applies when `prune_method` is 'prune_by_class', 'prune_by_noise_rate' or 'both'
            Default value is `1.0`
            You can check cleanlab documentation for further details.
        n_jobs (int): Number of processing threads used by multiprocessing.
        seed (int): seed to use.

    Returns:
        returns the indexes of the samples that are detected as containing the issues and the confident joint matrix

    """

    clf = LogisticRegression() if clf is None else clf

    psx = estimate_cv_predicted_probabilities(
        X=X,
        labels=y,
        clf=clf,
        cv_n_folds=n_folds,
        seed=seed,
    )

    confident_joint = get_confident_joint(X, y, psx, n_folds, clf, seed)

    cl_label_errors = find_label_issues(
        labels=y,
        pred_probs=psx,
        confident_joint=confident_joint,
        frac_noise=frac_noise,
        num_to_remove_per_class=num_to_remove_per_class,
        filter_by=prune_method,
        return_indices_ranked_by=sorted_index_method,
        n_jobs=n_jobs
    )

    return cl_label_errors, confident_joint