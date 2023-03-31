import numpy as np

from scipy.special import rel_entr

def hellinger_distance(p, q, normalize=True):
    """
    Computes Hellinger distance between two histograms as specified in this paper:
    http://users.rowan.edu/~polikar/RESEARCH/PUBLICATIONS/cidue11.pdf
    Both histograms are represented by numpy arrays with the same number of dimensions,
    where each dimension represents the counts for that particular bin.
    It is assumed that the bin edges are the same.

    Args:
        p (np.array): First histogram. Each dimension represents one bin and bin edges are assumed to be the same as in q.
        q (np.array): Second histogram. Each dimension represents one bin and bin edges are assumed to be the same as in p.
        normalize (bool): Whether to normalize the histograms. If True the proportions of each bin are calculated first
            and then the distance is calculated with the proportions

        Returns:
            (float): float representing the Hellinger distance

    """

    if len(p) != len(q):
        raise ValueError("p and q must have the same size and represent the same bins")

    if normalize:
        p = p / np.sum(p)
        q = q / np.sum(q)

    distances = (np.sqrt(p) - np.sqrt(q)) ** 2
    return np.sqrt(np.sum(distances))


def jeffreys_divergence(p, q, normalize=True):

    """
    Computes Jeffreys divergence between two histograms as specified in this paper:
    https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/rubner-jcviu-00.pdf
    Both histograms are represented by numpy arrays with the same number of dimensions,
    where each dimension represents the counts for that particular bin.
    It is assumed that the bin edges are the same.

    Args:
        p (np.array): First histogram. Each dimension represents one bin and bin edges are assumed to be the same as in q.
        q (np.array): Second histogram. Each dimension represents one bin and bin edges are assumed to be the same as in p.
        normalize (bool): Whether to normalize the histograms. If True the proportions of each bin are calculated first
            and then the distance is calculated with the proportions

        Returns:
            (float): float representing the hellinger distance

    """

    if len(p) != len(q):
        raise ValueError("p and q must have the same size and represent the same bins")

    if normalize:
        p = p / np.sum(p)
        q = q / np.sum(q)

    m = (p + q) / 2
    distances = rel_entr(p, m) + rel_entr(q, m)
    return np.sum(distances)


def psi(p, q, normalize=True, eps=1e-4):

    """
    Calculates the Population Stability Index (PSI). The metric helps to measure the stability between two population
    samples. It assumes that the two population samples have already been splitted in bins, so the histograms are the
    input to this function.

    Args:
        p (np.array): First histogram. Each dimension represents one bin and bin edges are assumed to be the same as in q.
        q (np.array): Second histogram. Each dimension represents one bin and bin edges are assumed to be the same as in p.
        normalize (bool): Whether to normalize the histograms. If True the proportions of each bin are calculated first
            and then the distance is calculated with the proportions

    Returns:
        (float): float representing the PSI

    Example:
        ```python
        >>> a = np.array([12, 11, 14, 12, 12, 10, 12, 6, 6, 5])
        >>> b = np.array([11, 11, 12, 13, 11, 11, 13, 5, 7, 6])
        >>> psi = psi(a, b)
        ```
    """

    if len(p.shape) != 1 or len(q.shape) != 1:
        raise ValueError("p and q must be np.array with len(shape)==1")

    if len(p) != len(q):
        raise ValueError("p and q must have the same size and represent the same bins")

    if normalize is None:
        if np.any(p > 1) or np.any(q > 1):
            normalize = True
        else:
            normalize = False

    if normalize:
        p = p / np.sum(p)
        q = q / np.sum(q)

    # Replace 0's to avoid inf and nans
    p[p == 0] = eps
    q[q == 0] = eps

    psi = (p - q) * np.log(p / q)
    return np.sum(psi)