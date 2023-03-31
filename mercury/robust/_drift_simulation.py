
from typing import Callable, Union, List

import warnings
import numpy as np
import pandas as pd

from mercury.dataschema import DataSchema


class BatchDriftGenerator():
    """ This is a simulator for applying synthetic perturbations to a given dataset,
    so real life drift situations can be generated.

    It allows generating several typical drift-like perturbations. In particular, it has the
    following ones implemented:

        1) Shift drift: Each one of the numerical features will be shifted by an amount N. With this,
        you will be able to simulate situations where a particular feature increases of decreases.
        2) Recodification drift: This operations simulates the situation where a particular feature
        changes its codification (due to some change in a data ingestion pipeline). For example, imagine
        a feature codified as [A, B, C] and, for some reason, category "C" becomes "A" and vice versa.
        3) Missing value drift: Artificially increases the percentage of NaNs on a given feature or
        list of features.
        4) Hyperplane rotation drift: This method operates by modifying a set of features at once but
        attempting to preserve the relationships they have. In other terms, it tries to change the
        joint distribution of N continuous variables at once.
        5) Outliers drift: This method artificially introduces outliers in the specified columns. You
        can use one of the existing methods to inject outliers or create your own method. Check the
        method documentation for `outliers_drift` for further documentation

    Args:
        X: Dataset features
        names_categorical: An optional list with the names of the categorical variables. If given,
        even as an empty list, only the column names in the list will be treated as categorical.
        Otherwise, it will try to autodetect.

    """
    def __init__(
            self, X: Union[pd.DataFrame, np.ndarray],
            schema: DataSchema = None,
    ):
        self.X = X

        if isinstance(self.X, np.ndarray):
            self.X = pd.DataFrame(self.X, columns=[f'col{i}' for i in range(self.X.shape[-1])])

        self._schema = schema

    @property
    def schema(self):
        if self._schema is None:
            warnings.warn(
                """You haven't specified a schema in the constructor neither `cols`. Autoinferring an schema... """
                """If you wish to suppress this warning provide a schema to the construct or specify the `cols` parameter.""",
                RuntimeWarning
            )
            self._schema = DataSchema().generate(self.X, verbose=True)
        return self._schema

    def hyperplane_rotation_drift(self, cols: List[str] = None, force: float = 10.0) -> "BatchDriftGenerator":
        """ Adds drift to several features at once by rotating them.

        Args:
            cols: columns which will be jointly rotated.
            force: rotation amount in degrees (0 or 360 = no transformation)
        Returns:
            An updated version of the simulator.
        """
        force = abs(force)
        columns = self.schema.continuous_feats if cols is None else cols
        rotmat = self.__get_rotation_matrix(len(columns), force)
        # Rows with NaNs wont receive the transformation
        nan_entries = self.X.loc[:, columns].isna().any(axis=1)
        self.X.loc[~nan_entries, columns] = self.X.loc[~nan_entries, columns].values @ rotmat
        return self

    def shift_drift(self, cols: List[str] = None, force=5.0, noise=0.0) -> "BatchDriftGenerator":
        """ Adds drift to one or several features by adding or substracting a `force`
        quantity. You could also add random noise to these shifts via the `noise`
        parameter. The shifted amount will follow a gaussian distribution with mean
        `force` and std `noise`.

        Args:
            cols: List of columns names where the operation will be performed
            force: Amount shift each column will receive
            noise: Amount of standard deviation the shift of each column will receive.

        Returns:
            An updated version of the simulator.
        """
        drifted_df = self.X
        columns = self.schema.continuous_feats if cols is None else cols
        for i, col in enumerate(columns):
            f = force[i] if type(force) == list else force
            n = noise[i] if type(noise) == list else noise
            shifts = np.random.normal(f, n, len(drifted_df))
            drifted_df.loc[:, col] += shifts

        return self

    def scale_drift(
        self, cols: List[str] = None, mean=1, sd=0.05, iqr=None
    ) -> "BatchDriftGenerator":
        """
        Adds drift to one or several features by multiplying then by some
        random factor (normally close to 1).

        Args:
            cols: List of columns names where the operation will be performed
            mean: The mean value of the random scale (if iqr is not given). Default is 1.
            sd: The standard deviation of the random scale (if iqr is not given). Default is 0.05.
            iqr: An alternative way to define the random scale is by giving the interquartile range as
                a list of 2 elements. E.g. [1, 1.2]

        Returns:
            An updated version of the simulator.
        """
        if iqr is not None:
            # q1, q3 = mean +/- .67449*sd
            q1, q3 = iqr[0], iqr[1]

            mean = (q1 + q3) / 2
            sd = (q3 - q1) / 1.34898

        drifted_df = self.X

        if cols is None:
            cols = self.schema.continuous_feats

        for col in cols:
            scale = np.random.normal(mean, sd, len(drifted_df))
            drifted_df.loc[:, col] *= scale

        return self

    def recodification_drift(self, cols: List[str] = None) -> "BatchDriftGenerator":
        """ Adds drift by randomly remapping the categories of cols.
        For example, if you have a column with domain: ["A", "B", "C"], then, this
        operation would swap all the "A" with "C" (and vice versa).

        Note the swaps are random. The same remapping will not be repeated.

        Args:
              cols: List of columns names to be recoded

        Returns:
            An updated version of the simulator.
        """
        drifted_df = self.X
        cols = self.schema.categorical_feats if cols is None else cols
        for col in cols:
            self.__remap_categorical(drifted_df, col)

        return self

    def missing_val_drift(self, cols: List[str] = None,
                          percent: Union[float, List[float]] = .1) -> "BatchDriftGenerator":
        """ Randomly injects NaN values in a column.

        Args:
            cols: Columns where this operation will be performed
            percent: Percentage of values which will become Null values.
        Returns:
            An updated version of the simulator.
        """
        if type(cols) == list and type(percent) == list and len(cols) != len(percent):
            raise ValueError("If both cols and percent are lists, they must have the same size.")

        drifted_df = self.X
        cols = drifted_df.columns if cols is None else cols
        for i, col in enumerate(cols):
            p = percent[i] if type(percent) == list else percent
            mask = np.random.uniform(0, 1, len(self.X)) < p
            drifted_df.loc[mask, col] = np.nan

        return self

    def outliers_drift(
        self, cols: List[str] = None, method: Union[str, Callable] = "percentile", method_params: dict = None
    ) -> "BatchDriftGenerator":
        """
        Injects outliers in `cols` using the method specified in `method`. Parameters of the method can
        be specified using `method_params`.

        Args:
              cols: List of columns names to generate outliers.
              method: string of the method to generate to use or custom function to generate outliers. If
                using string, the current available method are 'percentile' and 'value'. If using 'percentile'
                values in the dataframe are replaced by a percentile of that column. The percentile to use is
                specified in key 'percentile' in `method_params` dictionary, and the key 'proportion_outliers'
                specifies the proportion of outliers to set in the column. If using 'value', a custom value is
                set in the column. The value is specified with the key 'value' in `method_params` and the
                key `proportion_outliers` specifies the proportion of outliers to set in the column.
                When using a custom function, the interface that should follow is
                `fn(X: np.array, params: dict = None)`, where `X` will be an array of shape==1 to apply
                the outlier generation and `params` is the dictionary where parameters of the method
                are passed.
              method_params: dictionary containing the parameters to use in `method`

        Returns:
            An updated version of the simulator.
        """
        drifted_df = self.X

        if cols is None:
            cols = self.schema.continuous_feats + self.schema.discrete_feats

        if isinstance(method, Callable):
            generate_outliers_fn = method
        else:
            generate_outliers_fn = _get_outlier_generation_fn(method)

        for col in cols:
            outliers = generate_outliers_fn(drifted_df.loc[:, col].values, method_params)
            drifted_df.loc[:, col] = outliers

        return self

    def __get_rotation_matrix(self, n: int, deg: float) -> np.ndarray:
        """ Returns a rotation matrix for a R^n space.

        Args:
            n: Dimensionality of the space
            deg: amount of rotation to be applied. In degrees.

        Returns:
            nxn dimensional rotation matrix
        """
        # https://en.wikipedia.org/wiki/Orthogonal_matrix#:~:text=Canonical%20form%5Bedit%5D
        theta = deg * np.pi / 180
        rot = np.zeros(shape=(n, n))

        for i in range(0, n - 1, 2):
            rot[i, i] = np.cos(theta)
            rot[i, i + 1] = -np.sin(theta)
            rot[i + 1, i] = np.sin(theta)
            rot[i + 1, i + 1] = np.cos(theta)
        if n % 2 == 1:
            rot[-1, -1] = 1
        return rot

    def __remap_categorical(self, df: pd.DataFrame, colname: str):
        """ Randomly shuffles the categories of a particular column
        """
        original_ids = df[colname].dropna().unique()
        remaped_ids = np.array(original_ids)
        np.random.shuffle(remaped_ids)
        remap_dic = {x: y for x, y in zip(original_ids, remaped_ids)}
        df.loc[:, colname] = df.loc[:, colname].map(remap_dic)

    @property
    def data(self) -> pd.DataFrame:
        return self.X


def _replace_values_randomly(X: np.array, value: Union[float, int], percentage: float):
    indices = np.random.choice(range(len(X)), size=int(len(X) * percentage), replace=False)
    X[indices] = value
    return X


def _generate_outliers_percentile(X: np.array, params: dict = None):

    if not np.issubdtype(X.dtype, np.number):
        raise ValueError("generating outliers by perecentile only works with numeric arrays")

    _params = dict(
        percentile=99,
        proportion_outliers=0.5
    )
    if isinstance(params, dict):
        _params.update(params)

    outlier_value = np.percentile(X[~np.isnan(X)], q=_params["percentile"])

    X = _replace_values_randomly(X, value=outlier_value, percentage=_params["proportion_outliers"])
    return X


def _generate_outliers_setting_value(X, params: dict = None):

    _params = dict(
        value=X.max(),
        proportion_outliers=0.5
    )
    if isinstance(params, dict):
        _params.update(params)

    X = _replace_values_randomly(X, value=_params["value"], percentage=_params["proportion_outliers"])
    return X


def _get_outlier_generation_fn(name_fn: str) -> Callable:
    if name_fn.lower() == "percentile":
        return _generate_outliers_percentile
    elif name_fn.lower() == "value":
        return _generate_outliers_setting_value
    else:
        raise ValueError("Invalid method name to generate outliers. Available method: 'percentile' and 'value'")