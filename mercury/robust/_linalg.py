import numpy as np


def lin_combs_in_columns(X, tol=None):
    """
    Returns all the linear combinations that exists among the columns of X by performing Gauss-Jordan elimination.

    Args:
        X (np.array): Any numerical matrix (rank == 2) dimensions (r, c) that is either a numpy ndarray or can be converted into one
                      by passing it to the constructor of a numpy ndarray.
        tol (float):  An optional numerical tolerance. If not used, it will be set according to the number of columns squared times a small
                      constant. Note that numerical rounding errors will inevitably apply after performing reduction operations on the
                      matrix. In that case, deciding if a value is zero, or just very small depends on this tolerance. When you
                      independently compute the rank of X and expect a number of combinations, you can play with this tolerance to match
                      your expectations. Otherwise, it is okay to just leave the default value.

    Returns:
        None, if no linear combinations exist. If linear combinations exist: a matrix of shape (n, c) where n is the number of linear
        combinations found and c is the number of columns of the original X matrix. Each linear combination is defined so the
        **for each row** in the result matrix the product of X·r' is a column of all zeros. r' is the row transposed as a column vector.
    """
    X = np.array(list(X), dtype=np.float64)

    assert len(X.shape) == 2

    m = X.shape[1]

    if tol is None:
        tol = 1e6 * m * m * np.finfo(np.double).eps

    oidx = [i for i in range(m)]
    zX = 1 * (abs(X) > tol)
    cs = np.sum(zX, axis=0)
    clin = None

    for i, s in reversed(list(enumerate(cs))):
        if s == 0:
            cl = np.zeros((1, m), dtype=np.float64)
            cl[0, i] = 1
            if clin is None:
                clin = cl
            else:
                clin = np.vstack((clin, cl))

            if X.shape[1] == 1:
                return clin

            X = np.delete(X, i, axis=1)
            oidx.pop(i)

    rs = np.sum(zX, axis=1)

    for i, s in reversed(list(enumerate(rs))):
        if s == 0:
            X = np.delete(X, i, axis=0)

    nr, n = X.shape
    min_lc = max(0, n - nr)

    comb = np.zeros((n, m), dtype=np.float64)
    for j in range(n):
        comb[j, oidx[j]] = 1

    i = 0
    while i < n - 1 and i < nr:
        last = i == nr - 1
        row = X[i, :].copy()
        zrow = 1 * (abs(row) > tol)
        zpt = zrow[i]

        if zpt == 0:
            col = X[:, i].copy()
            zcol = 1 * (abs(col) > tol)

            j = i + 1
            while zcol[j] == 0:
                j += 1

            X[[i, j], :] = X[[j, i], :]

            row = X[i, :].copy()
            zrow = 1 * (abs(row) > tol)

        col = X[:, i].copy()
        j = i + 1

        while j < n:
            if zrow[j] > 0:
                f = -row[j] / row[i]

                X[:, j] = X[:, j] + f * col

                comb[j, :] = comb[j, :] + comb[i, :] * f

                if (last and n - j <= min_lc) or not np.any(abs(X[:, j]) > tol):
                    if clin is None:
                        clin = np.reshape(comb[j, :].copy(), (1, -1))
                    else:
                        clin = np.vstack((clin, comb[j, :]))

                    min_lc -= 1

                    comb = np.delete(comb, j, axis=0)
                    X = np.delete(X, j, axis=1)
                    row = np.delete(row, j)
                    zrow = np.delete(zrow, j)

                    n -= 1
                    j -= 1

            j += 1

        i += 1

    return clin