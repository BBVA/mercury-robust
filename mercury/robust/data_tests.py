from .basetests import RobustDataTest
from .errors import FailedTestError
from typing import Union, Any, List, Callable, Dict, Tuple
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
import psutil

from mercury.dataschema import DataSchema
from mercury.dataschema.feature import DataType, FeatType

import numpy as np
import pandas as pd


class SameSchemaTest(RobustDataTest):
    """
    This test ensures a certain dataset, or new batch of samples, follows a
    previously defined schema.

    On the `run` call, it will make a schema validation. If something isn't right, it
    will raise an Exception, crashing the test.

    Args:
        base_dataset: Dataset which will be evaluated
        schema_ref: Schema the base_dataset will be evaluated against. It can be either a
                    `DataSchema` object or a string. In case of the later, it must be a
                    path to a previously serialized schema.
        custom_feature_map: Dictionary with <column_name: FeatType>. You can provide this in case
                            the DataSchema wrongly infers the types for the new data batch. If
                            this is None, the types will be auto inferred.
        name (str): A name for the test. If not used, it will take the name of the class.

    Example:
        ```python
        >>> from mercury.dataschema import DataSchema
        >>> schema_reference = DataSchema().generate(df_train).calculate_statistics()
        >>> from mercury.robust.data_tests import SameSchemaTest
        >>> test = SameSchemaTest(df_inference, schema_reference)
        >>> test.run()
        ```

    """
    def __init__(self,
                 base_dataset: "pandas.DataFrame",  # noqa: F821
                 schema_ref: Union[str, DataSchema],
                 custom_feature_map: Dict[str, FeatType] = None,
                 name: str = None,
                 *args: Any, **kwargs: Any):
        super().__init__(base_dataset, name, *args, **kwargs)
        self.schema_ref = schema_ref
        self.custom_feature_map = custom_feature_map

        if type(schema_ref) == str:
            self.schema_ref = DataSchema.load(schema_ref)

    def run(self, *args, **kwargs):
        """
        Runs the test.

        Raises FailedTestError if any of the checks fail.
        """
        super().run(*args, **kwargs)
        current_schema = DataSchema().generate(self.base_dataset, force_types=self.custom_feature_map)
        try:
            self.schema_ref.validate(current_schema)
        except RuntimeError as e:
            raise FailedTestError(str(e))


class LinearCombinationsTest(RobustDataTest):
    """
    This test ensures a certain dataset doesn't have any linear combination between
    its numerical columns and no categorical variable is redundant. See the following functions if you want to know further details:

      - For numerical features: _lin_combs_in_columns.lin_combs_in_columns
      - For categorical features: _CategoryStruct.individually_redundant

    In case any combination or redundancy is detected, the test fails.

    Args:
        base_dataset: Dataset which will be evaluated
        schema_custom_feature_map: Internally, this test generates a DataSchema object. In case you find it makes
                                   wrong feature type assignations to your features you can pass here a dictionary which
                                   specify the feature type of the columns you want to fix. (See DataSchema. `force_types`
                                   parameter for more info on this).
        dataset_schema: Pre built schema. This argument is complementary to `schema_custom_feature_map`. In case you want to
                        manually build your own DataSchema object, you can pass it here and the test will internally use it
                        instead of the default, automatically built. If you provide this parameter, `schema_custom_feature_map`
                        will not be used.
        name (str): A name for the test. If not used, it will take the name of the class.

    Example:
        ```python
        >>> from mercury.robust.data_tests import LinearCombinationsTest
        >>> test = LinearCombinationsTest(df_train)
        >>> test.run()
        ```
    """
    def __init__(self,
                 base_dataset: "pandas.DataFrame",  # noqa: F821
                 schema_custom_feature_map: Dict[str, FeatType] = None,
                 dataset_schema: DataSchema = None,
                 name: str = None, *args: Any, **kwargs: Any):
        super().__init__(base_dataset, name, *args, **kwargs)
        self._schema_custom_feature_map = schema_custom_feature_map
        self._dataset_schema = dataset_schema

    def run(self, *args, **kwargs):
        """
        Runs the test.

        Raises FailedTestError if any of the checks fail.
        """
        from ._linalg import lin_combs_in_columns
        from ._category_struct import CategoryStruct

        super().run(*args, **kwargs)

        if self._dataset_schema is None:
            current_schema = DataSchema().generate(self.base_dataset, force_types=self._schema_custom_feature_map)
        else:
            current_schema = self._dataset_schema

        lin_combinations = None
        numeric_feats = current_schema.continuous_feats + current_schema.discrete_feats
        if len(numeric_feats) > 0:
            cont_cols = self.base_dataset.loc[:, numeric_feats].values

            lin_combinations = lin_combs_in_columns(
                np.matmul(cont_cols.T, cont_cols)  # Compress original matrix
            )

        if lin_combinations is not None:
            # Create message with the linear combinations found
            lin_combinations = np.around(lin_combinations, decimals=5)
            lin_combs_found = []
            for i in range(lin_combinations.shape[0]):
                lin_comb_idx = np.where(lin_combinations[i] != 0)[0]
                lin_comb_cols = self.base_dataset.loc[:, numeric_feats].columns[lin_comb_idx].tolist()
                lin_combs_found.append(lin_comb_cols)
            raise FailedTestError(f"Test failed. Linear combinations for continuous features were encountered: {lin_combs_found}.")

        individually_redundant = CategoryStruct.individually_redundant(self.base_dataset, current_schema.categorical_feats)
        if len(individually_redundant) > 0:
            raise FailedTestError((
                f"""Test failed. Any of these categorical variables is redundant: {individually_redundant}. """
                """Try deleting any of them."""
            ))


class DriftTest(RobustDataTest):
    """
    This test ensures the distributions of new dataset, or new batch of data, are
    not too different from a reference dataset (i.e. training data). In other words, it
    checks for no data drift between features.

    If drift is detected on any of the features, it will raise an Exception, crashing the test.

    Args:
        base_dataset: Dataset which will be evaluated
        schema_ref: Schema the base_dataset will be evaluated against. It can be either a
                    `DataSchema` object or a string. In case of the later, it must be a
                    path to a  previously serialized schema.
        drift_detector_args: Dictionary with arguments passed to the internal drift detectors:
                            HistogramDistanceDrift for continuous features and Chi2Drift for
                            the categoricals.
        name (str): A name for the test. If not used, it will take the name of the class.

    Example:
        ```python
        >>> from mercury.dataschema import DataSchema
        >>> schema_reference = DataSchema().generate(df_train).calculate_statistics()
        >>> from mercury.robust.data_tests import DriftTest
        >>> test = DriftTest(df_inference, schema_reference)
        >>> test.run()
        >>> # Check test result details
        >>> test.info()
        ```
    """
    def __init__(self,
                 base_dataset: "pandas.DataFrame",  # noqa: F821
                 schema_ref: Union[str, DataSchema] = None,
                 drift_detector_args: dict = None,
                 name: str = None,
                 *args: Any, **kwargs: Any):
        super().__init__(base_dataset, name, *args, **kwargs)
        self.schema_ref = schema_ref

        # Default args for ALL detectors. Depending on the implementation, only the
        # necessary ones will be used.
        self._detector_args = dict(
            distance_metric="hellinger",
            correction="bonferroni",
            n_runs=3,
            test_size=0.3,
            p_val=0.01,
            n_permutations=100
        )

        if isinstance(drift_detector_args, dict):
            self._detector_args.update(drift_detector_args)

        if type(schema_ref) == str:
            self.schema_ref = DataSchema.load(schema_ref)

        self.continuous_drift_metrics = None
        self.cat_drift_metrics = None

    def info(self):
        if (self.continuous_drift_metrics is None) and (self.cat_drift_metrics is None):
            return None

        dict_info = {}
        if self.continuous_drift_metrics is not None:
            dict_info["continuous_drift_metrics"] = self.continuous_drift_metrics["score"]
        if self.cat_drift_metrics is not None:
            dict_info["cat_drift_metrics"] = self.cat_drift_metrics["score"]

        return dict_info

    def run(self, *args, **kwargs):
        """
        Runs the test.

        Raises FailedTestError if any drift is detected
        """
        from mercury.monitoring.drift.histogram_distance_drift_detector import HistogramDistanceDrift
        from mercury.monitoring.drift.chi2_drift_detector import Chi2Drift

        super().run(*args, **kwargs)

        schema_ref = self.schema_ref

        # Test numerical features with HistogramDistance
        featnames = schema_ref.continuous_feats + schema_ref.discrete_feats

        if len(featnames) > 0:

            src_histograms, tgt_histograms = self._get_features_histograms(featnames)

            detector_args = {
                'distance_metric': self._detector_args['distance_metric'],
                'correction': self._detector_args['correction'],
                'p_val': self._detector_args['p_val'],
                'n_permutations': self._detector_args['n_permutations']
            }

            drift_detector = HistogramDistanceDrift(
                distr_src=src_histograms,
                distr_target=tgt_histograms,
                features=featnames,
                **detector_args
            )

            self.continuous_drift_metrics = drift_detector.calculate_drift()

            if self.continuous_drift_metrics['drift_detected']:
                raise FailedTestError(f"Test failed. Drift was detected on the following features: {drift_detector.get_drifted_features()}")

        # Test categorical feats with Chi2
        featnames = schema_ref.binary_feats + schema_ref.categorical_feats
        if len(featnames) > 0:

            src_histograms, tgt_histograms = self._get_features_histograms(featnames)

            chi2_args = {
                'correction': self._detector_args['correction'],
                'p_val': self._detector_args['p_val'],
            }

            drift_detector = Chi2Drift(
                distr_src=src_histograms,
                distr_target=tgt_histograms,
                features=featnames,
                **chi2_args
            )

            self.cat_drift_metrics = drift_detector.calculate_drift()

            if self.cat_drift_metrics['drift_detected']:
                raise FailedTestError(f"Test failed. Drift was detected on the following features: {drift_detector.get_drifted_features()}")

    def _get_features_histograms(self, featnames: list) -> Tuple[List[np.array], List[np.array]]:
        """
        Returns the source and target histograms of each feature in featnames

        Args:
            featnames: list of feature names to obtain the histograms
        Returns
            Two list src_histograms and tgt_histograms, with the first containing the histograms of the source dataset and
            the latter containing the histograms of the target dataset

        """

        src_histograms = []
        tgt_histograms = []
        for feat in featnames:
            src_histogram, tgt_histogram = self._get_histograms(feat)
            src_histograms.append(src_histogram)
            tgt_histograms.append(tgt_histogram)
        return src_histograms, tgt_histograms

    def _get_histograms(self, feat: str) -> Tuple[np.array, np.array]:
        """
        Returns histograms for a specific feature, feat.
        Obtains the histogram for the reference dataset using the dataschema,
        and the histogram for the target dataset from the base_dataset.

        Args:
            feat: feature name

        Returns:
            Two numpy arrays, the first containing the histogram for the source dataset,
            and the second containing the histogram for the target dataset

        Raises ValueError if the new feature has a new value not present in the schema.
        """

        schema_ref = self.schema_ref
        # Obtain source histogram from dataschema stats
        src_hist = np.array(schema_ref.feats[feat].stats['distribution'])

        # Obtain target histogram, depending on if its a discrete feature or a numeric one
        discrete_feats = self.schema_ref.categorical_feats + self.schema_ref.binary_feats

        if feat in discrete_feats:
            # If features are discrete, we don't need to take numerical ranges into
            # account
            domain = self.base_dataset.loc[:, feat].value_counts()
            values = set(domain.index)
            reference_values = set(schema_ref.feats[feat].stats['distribution_bins'])

            # If new dataset has categories that base doesn't have, raise FailedTestError
            new_values = values - reference_values
            if len(new_values) > 0:
                raise FailedTestError(
                    f"Test failed. Drift was detected on feature '{feat}' because has new categories: {new_values}"
                )

            # If new dataset has missing categories than reference, we add them with 0 count
            missing_values = reference_values - values
            if len(missing_values) > 0:
                for v in missing_values:
                    domain[v] = 0

            domain.index = domain.index.astype(str)
            str_bins = [str(x) for x in schema_ref.feats[feat].stats['distribution_bins']]
            domain = domain[str_bins]
            tgt_hist = domain.values

        else:
            # Feature is  numerical, we need to match source and target bins
            tgt_hist = np.histogram(self.base_dataset.loc[:, feat],
                                    bins=schema_ref.feats[feat].stats['distribution_bins']
                                    )[0]

            # If we have a new minimum, then we will have target samples outside the limits of the source distribution
            # We add a new bin at the beginning of both histograms for these samples
            if self.base_dataset.loc[:, feat].min() < schema_ref.feats[feat].stats['min']:
                source_samples_new_bin = 0
                target_samples_new_bin = (self.base_dataset.loc[:, feat] < schema_ref.feats[feat].stats['min']).sum()
                src_hist = np.insert(src_hist, 0, source_samples_new_bin)
                tgt_hist = np.insert(tgt_hist, 0, target_samples_new_bin)

            # If we have a new maximum, then we will have target samples outside the limits of the source distribution
            # We add a new bin at the end of both histograms for these samples
            if self.base_dataset.loc[:, feat].max() > schema_ref.feats[feat].stats['max']:
                source_samples_new_bin = 0
                target_samples_new_bin = (self.base_dataset.loc[:, feat] > schema_ref.feats[feat].stats['max']).sum()
                src_hist = np.insert(src_hist, len(src_hist), source_samples_new_bin)
                tgt_hist = np.insert(tgt_hist, len(tgt_hist), target_samples_new_bin)

            # Convert counts to densities
            tgt_hist = tgt_hist / tgt_hist.sum()

        # Trick because Drift detectors don't work with densities yet
        src_hist *= 1000
        tgt_hist *= 1000

        return src_hist, tgt_hist


class LabelLeakingTest(RobustDataTest):
    """
    This test ensures the target variable is not being leaked into the predictors
    (i.e. no feature has a strong relationship with the target). For this, it
    uses the TreeSelector class which tries to predict the
    target variable given each one of the features. If, given a particular feature, the
    target is easily predicted, it means that feature is very important. By default,
    performance is measured by ROC-AUC for classification problems and R^2 for regression.

    After the test has been executed, two attributes are made available for further inspection.

        1) `importances_`: dictionary with the the feature importances. That is,
           the closer a feature is to 1 (ideal value), the most important it is. If any
           feature f is f > 1 - threshold, the test will fail.
        2) `_selector`: Fitted TreeSelector used for the test.

    Args:
        base_dataset: Dataset which will be evaluated. Features must be numerical (i.e. no strings/objects)
        label_name: column which represent the target label. If it's multi-categorical, it
                    must be label-encoded.
        task: Task of the dataset. It must be either 'classification' or 'regression'. If None provided
              it will be auto inferred from the `label_name` column.
        threshold: Custom threshold which will be used when considering a particular feature important.
                   If the obtained score for a feature < threshold, it will be considered "too important", and, thus,
                   the test will fail.
        metric: Metric to be used by the internal TreeSelector. If None, ROC AUC will be used for classification and
                R2 for regression.
        ignore_feats: Features which will be not tested.
        schema_custom_feature_map: Internally, this test generates a DataSchema object. In case you find it makes
                                   wrong feature type assignations to your features you can pass here a dictionary which
                                   specify the feature type of the columns you want to fix. (See DataSchema. `force_types`
                                   parameter for more info on this).
        dataset_schema: Pre built schema. This argument is complementary to `schema_custom_feature_map`. In case you want to
                        manually build your own DataSchema object, you can pass it here and the test will internally use it
                        instead of the default, automatically built. If you provide this parameter, `schema_custom_feature_map`
                        will not be used.
        handle_str_cols (str): if the dataset contains string columns, this parameter indicates how to handle them. If 'error'
            then a ValueError exception is raised if the dataset contains string columns. If 'transform' the string columns
            are transformed to integer using a LabelEncoder. Default value is 'error'
        name (str): A name for the test. If not used, it will take the name of the class.

    Example:
        ```python
        >>> from mercury.robust.data_tests import LabelLeakingTest
        >>> test = LabelLeakingTest(
        >>>     train_df,
        >>>     label_name = "y",
        >>>     task = "classification",
        >>>     threshold = 0.05,
        >>> )
        >>> test.run()
        >>> # Check test result details
        >>> test.info()
        ```
    """
    def __init__(self,
                 base_dataset: "pandas.DataFrame",  # noqa: F821
                 label_name: str,
                 task: str = None,
                 threshold: float = None,
                 metric: Callable = None,
                 ignore_feats: List[str] = None,
                 schema_custom_feature_map: Dict[str, FeatType] = None,
                 dataset_schema: DataSchema = None,
                 handle_str_cols: str = "error",
                 name: str = None,
                 *args: Any, **kwargs: Any):
        super().__init__(base_dataset, name, *args, **kwargs)

        self.label_name = label_name
        self.task = task
        self.threshold = threshold
        self.ignore_feats = ignore_feats
        self.metric = metric
        self.importances_ = None
        self._schema_custom_feature_map = schema_custom_feature_map
        self._base_schema = dataset_schema
        self.handle_str_cols = handle_str_cols

    def info(self):
        if self.importances_ is not None:
            return {"importances": self.importances_}
        else:
            return None

    def run(self, *args, **kwargs):
        """
        Runs the test.

        Raises FailedTestError if any of the checks fail.
        """
        from ._treeselector import TreeSelector
        from sklearn.metrics import r2_score, roc_auc_score

        super().run(*args, **kwargs)

        base_dataset = self.base_dataset

        if self.ignore_feats is not None:
            base_dataset = base_dataset.loc[:, [c for c in base_dataset.columns if c not in self.ignore_feats]]

        label_name = self.label_name
        threshold = self.threshold
        metric = self.metric

        if self._base_schema is None:
            schema = DataSchema().generate(base_dataset, force_types=self._schema_custom_feature_map)
        else:
            schema = self._base_schema

        if not self.task:
            if label_name in schema.binary_feats or label_name in schema.categorical_feats:
                self.task = 'classification'
            if label_name in schema.discrete_feats or label_name in schema.continuous_feats:
                self.task = 'regression'

        task = self.task

        if task not in ('classification', 'regression'):
            raise ValueError("Error. 'task' must be either classification or regression")

        # This threshold has been found to be working well in most tested cases for
        # R2 and AUC.
        threshold = 0.05 if not threshold else threshold
        if task == 'classification':
            metric = roc_auc_score if metric is None else metric
        else:
            metric = r2_score if metric is None else metric

        # The TreeSelector used later doesn't support string columns. Raise error or transform them
        str_cols = schema.get_features_by_type(datatype=DataType.STRING)
        if len(str_cols) > 0:
            if self.handle_str_cols.lower() == 'error':
                raise ValueError(
                    "String column found. Set 'handle_str_cols' parameter to 'transform' to automatically transform "
                    "string columns to integers or transform them before using this test."
                )
            elif self.handle_str_cols.lower() == 'transform':
                # We need to create a copy in this case to avoid to modify the original dataset
                base_dataset = base_dataset.copy()
                for col in str_cols:
                    if col in base_dataset.columns:
                        base_dataset.loc[:, col] = LabelEncoder().fit_transform(base_dataset.loc[:, col])
            else:
                raise ValueError("Wrong value for 'handle_str_cols' parameter. Set to 'error' or 'transform'")

        # Separate features from target
        X = base_dataset.loc[:, [f for f in base_dataset.columns if f != label_name]]
        y = base_dataset.loc[:, label_name]

        selector = TreeSelector(task, metric=metric)
        selector.fit(X, y)

        # Store selector for inspection. Mainly debugging purposes
        self._selector = selector

        # We calculate our "importance" scores depending on whether we have a
        # classification or regression dataset.
        importances = selector.feature_importances

        # By default,  we ensure the best performing feature (i.e.
        # the one with highest metric) is not too close to one (the best
        # possible value for classification and regression / AUC or R^2).
        metrics = 1 - np.clip(np.array(importances['metrics']), 0, 1)
        high_importance_feats = metrics < threshold

        # Store computed importances for post-analysis
        ids = X.columns[selector.feature_importances['features']].tolist()
        self.importances_ = dict()
        for i, val in enumerate(ids):
            self.importances_[val] = 1 - metrics[i]  # Recover original metric instead of its inverse

        # If any of the features is too important the test fails.
        if high_importance_feats.any():
            idxs = selector.feature_importances['features'][high_importance_feats]
            names = X.columns[idxs].tolist()
            raise FailedTestError((
                f"Test failed because high importance features were detected: {names}. "
                "Check for possible target leaking."
            ))


class NoisyLabelsTest(RobustDataTest):
    """
    This test looks if the labels of a dataset contain a high level of noise. Internally, it uses the cleanlab
    library to obtain those samples in the dataset where the labels are considered as noisy. Noisy labels can happen because
    the sample is incorrectly labelled, because the sample could belong to several label categories, or some other reason.
    If the percentage of samples with noisy labels is higher than a specified threshold, then the test fails.
    It can be used with tabular datasets and with text datasets. In the case of a text dataset, the argument `text_col` must
    be specified.
    This test only works for classification tasks.

    After the test has been executed, two attributes are made available for further inspection.

        1) `idx_issues_`: indices of the labels detected as issues (only available when `calculate_idx_issues` is True)
        2) `rate_issues_`: percentage of labels detected containing possible issues.

    IMPORTANT: If the Test reports convergence problems or it takes too long, then you can change the model used in the
    algorithm (see example below). Sometimes, the default Logistic Regression used might not converge in some datasets.
    In those cases, changing the solver is usually enough to solve the problem.

    Args:
        base_dataset (pandas.DataFrame): Dataset which will be evaluated.
        label_name (str): column which represents the target label.
        threshold (float): threshold to specify the percentage of noisy labels from which the test will fail. Default value is 0.4
        text_col (str): column containing the text. Only has to be specified when using a text dataset.
        preprocessor (Union[TransformerMixin, BaseEstimator]): Optional preprocessor for the features. If not specified, it will
            create a preprocessor with OneHotEncoder to encode categorical features in case of tabular data
            (text_col not specified) or a CountVectorizer to encode text in case of a text dataset(text_col specified)
        ignore_feats (List[str]): features that will not be used in the algorithm to detect noisy labels.
        calculate_idx_issues (bool): whether to calculate also the index of samples with label issues. If True, the idx will be
            available in `idx_issues_` attribute. Default value is True.
        label_issues_args (dict): arguments for the algorithm to detect noisy labels. You can check documentation from
            `_label_cleaning.get_label_issues` to see all available arguments.
        schema_custom_feature_map: Internally, this test generates a DataSchema object. In case you find it makes
                                   wrong feature type assignations to your features you can pass here a dictionary which
                                   specify the feature type of the columns you want to fix. (See DataSchema. `force_types`
                                   parameter for more info on this).
        dataset_schema: Pre built schema. This argument is complementary to `schema_custom_feature_map`. In case you want to
                        manually build your own DataSchema object, you can pass it here and the test will internally use it
                        instead of the default, automatically built. If you provide this parameter, `schema_custom_feature_map`
                        will not be used.
        name (str): A name for the test. If not used, it will take the name of the class.

    Example:
        ```python
        >>> from mercury.robust.data_tests import NoisyLabelsTest
        >>> test = NoisyLabelsTest(
        >>>     base_dataset=train_df,
        >>>     label_name="label_col",
        >>>     text_col = "text_col,
        >>>     threshold = 0.2,
        >>>     label_issues_args={"clf": LogisticRegression(solver='sag')}
        >>> )
        >>> test.run()
        >>> # Percentage of samples with possible label issues
        >>> test.rate_issues_
        >>> # Access to samples with possible label issues
        >>> train_df.iloc[test.idx_issues_]
        ```
    """

    def __init__(self,
                 base_dataset: "pandas.DataFrame",  # noqa: F821
                 label_name: str,
                 threshold: float = None,
                 text_col: str = None,
                 preprocessor: Union["TransformerMixin", "BaseEstimator"] = None,  # noqa: F821
                 ignore_feats: List[str] = None,
                 calculate_idx_issues: bool = True,
                 label_issues_args: dict = None,
                 name: str = None,
                 schema_custom_feature_map: Dict[str, FeatType] = None,
                 dataset_schema: DataSchema = None,
                 *args: Any, **kwargs: Any):
        super().__init__(base_dataset, name, *args, **kwargs)

        self.label_name = label_name
        self.threshold = 0.4 if not threshold else threshold
        self.preprocessor = preprocessor
        self.text_col = text_col
        self.ignore_feats = ignore_feats
        self.calculate_idx_issues = calculate_idx_issues
        self.rate_issues_ = None
        self._schema_custom_feature_map = schema_custom_feature_map
        self._dataset_schema = dataset_schema

        # Default args for function to get label issues
        self.label_issues_args = dict(
            clf=None,
            n_folds=5,
            frac_noise=1.0,
            num_to_remove_per_class=None,
            prune_method='prune_by_noise_rate',
            sorted_index_method=None,
            n_jobs=None,
            seed=None
        )

        if isinstance(label_issues_args, dict):
            self.label_issues_args.update(label_issues_args)

    def _preprocess_features_from_text(self, X):
        """
        Converts text from text_col column in X to a feature vector.

        Args:
            X (pd.DataFrame): features dataframe containing the text to preprocess

        Returns:
            (np.array): The preprocessed text
        """

        # If preprocessor specified we use it, otherwise create a simple CountVectorizer
        if self.preprocessor:
            preprocessor = self.preprocessor
        else:
            preprocessor = CountVectorizer(max_features=10000).fit(X[self.text_col])
        X = preprocessor.fit_transform(X[self.text_col])

        return X

    def _preprocess_features_from_tabular(self, X, current_schema):
        """
        Preprocesses the features from a tabular dataframe.

        Args:
            X (pd.DataFrame): features dataframe to preprocess
            current_schema (Schema): schema of the dataset

        Returns:
            (np.array): The preprocessed features
        """
        # If preprocessor specified we use it
        if self.preprocessor:
            preprocessor = self.preprocessor
            X = preprocessor.fit_transform(X)
        else:
            # If preprocessor not specified, create one to preprocess string features with OneHotEncoder
            str_feats = [c for c in current_schema.get_features_by_type(DataType.STRING) if c in X.columns]
            cat_feats = list(set.union(set(str_feats), set(current_schema.categorical_feats)))
            if len(cat_feats) > 0:
                preprocessor = ColumnTransformer(
                    transformers=[('ohe', OneHotEncoder(handle_unknown='ignore'), cat_feats)],
                    remainder='passthrough'
                )
                X = preprocessor.fit_transform(X)
            else:
                X = X.values
        return X

    def _preprocess_features(
        self,
        X: "pandas.DataFrame",  # noqa: F821
        current_schema: DataSchema
    ) -> np.array:
        """
        Preprocesses the features dataframe to create a np.array ready for the label issues detection
        algorithm.

        Args:
            X: features dataframe to preprocess
            current_schema: schema of the dataset

        Returns:
            np.array with the preprocessed features
        """

        # If text_col is specified, we have to preprocess (only) the text column
        if self.text_col:
            X = self._preprocess_features_from_text(X)
        else:
            X = self._preprocess_features_from_tabular(X, current_schema)

        return X

    def info(self):
        if self.rate_issues_ is not None:
            return {"rate_issues": self.rate_issues_}
        else:
            return None

    def run(self, *args, **kwargs):
        """
        Runs the test.

        Raises FailedTestError if noise above threshold is detected
        """
        from ._label_cleaning import get_label_issues, get_confident_joint

        super().run(*args, **kwargs)

        base_dataset = self.base_dataset

        if self.ignore_feats is not None:
            base_dataset = base_dataset.loc[:, [c for c in base_dataset.columns if c not in self.ignore_feats]]

        if self._dataset_schema is None:
            current_schema = DataSchema().generate(base_dataset, force_types=self._schema_custom_feature_map)
        else:
            current_schema = self._dataset_schema

        # Separate features from target
        X = base_dataset.loc[:, [f for f in base_dataset.columns if f != self.label_name]]
        y = base_dataset.loc[:, self.label_name].values

        # Preprocess features
        X = self._preprocess_features(X, current_schema)

        if self.calculate_idx_issues:
            # Get label issues
            self.idx_issues_, _ = get_label_issues(
                X=X,
                y=y,
                **self.label_issues_args
            )

            # Obtain number of label issues
            if self.idx_issues_.dtype == bool:
                self.num_issues = self.idx_issues_.sum()
            else:
                self.num_issues = len(self.idx_issues_)

        else:
            # Compute only confident joint and calculate number of issues as the sum of non-diagonal elements
            confident_join = get_confident_joint(
                X, y, n_folds=self.label_issues_args["n_folds"], clf=self.label_issues_args["clf"], seed=self.label_issues_args["seed"]
            )
            self.num_issues = confident_join.sum() - np.trace(confident_join)

        # Calculate percentage of obtained labels with issues
        self.rate_issues_ = self.num_issues / len(base_dataset)

        # If percentage of issues is higher than threshold, throw exception
        if self.rate_issues_ > self.threshold:
            raise FailedTestError(
                f"Test failed. High level of noise detected in labels. Percentage of labels with noise: {self.rate_issues_}"
            )


class CohortPerformanceTest(RobustDataTest):
    """
    This test looks if some metric performs poorly for some cohort of your data when compared with other groups.
    The computed metric is specified with the argument `eval_group_fn`, which the user needs to define (see example).
    The metric is computed for all the groups of the variable specified with the variable `group_col`.
    When the argument `compare_max_diff` is True, then the comparison is between the group which has the maximum
    metric and the group which has the minimum metric. If the difference is higher than the specified threshold
    then the test will fail. When the argument `compare_max_diff` is False, then the metric calculated for each
    group is compared against the mean of the whole dataset. If the difference of any of the groups with the mean
    is higher than the threshold, then the test will fail.
    The argument `threshold_is_percentage` controls if the threshold is compared with the absolute value of the
    difference (`threshold_is_percentage=False`), or if is compared with the percentage of the difference
    (`threshold_is_percentage=True`).

    Args:
        base_dataset (pandas.DataFrame): Dataset which will be evaluated. It must contain the specified `group_col`
            and the necessary columns to calculate the metric defined in `eval_group_fn`
        group_col (str): column name which contains the groups to be evaluated
        eval_fn (Callable): evaluation function which computes the metric to evaluate. It must return a float
        threshold (float): threshold to compare. If `compare_max_diff` is True and the difference between the
            maximum metric and minimum metric is higher than the threshold, then the test fails. If
            `compare_max_diff` is False and the difference between the mean metric of the dataset and the
            metric in any of the groups is higher than the threshold, then the test fails.
        compare_max_diff (bool): If True, then the comparison is between the group which has the max metric and the
            group which has the min metric. If False, then the comparison is between the mean of the whole dataset
            and all the other groups. Default value is True
        threshold_is_percentage (bool): If True, then the comparison with the threshold is with the percentage
            of the computed differences, therefore the threshold represents the percentage.
            If False, then the comparison with the threshold is with the absolute values of the differences,
            therefore the threshold represents the absolute value. Default value is True.
        name (str): A name for the test. If not used, it will take the name of the class.

    Example:
        ```python
        >>> from mercury.robust.data_tests import CohortPerformanceTest
        >>> def eval_acc(df):
        >>>     return accuracy_score(df["y_true"], df["y_pred"])
        >>> test1 = CohortPerformanceTest(
        >>>     base_dataset=dataset, group_col="gender", eval_fn = eval_acc, threshold = 0.2, compare_max_diff=True
        >>> )
        >>> test1.run()
        >>> # Check test result details
        >>> test1.info()
        ```
    """

    def __init__(
        self,
        base_dataset: "pandas.DataFrame",  # noqa: F821
        group_col: str,
        eval_fn: Callable,
        threshold: float,
        compare_max_diff: bool = True,
        threshold_is_percentage: bool = True,
        name: str = None,
        *args: Any,
        **kwargs: Any
    ):
        super().__init__(base_dataset, name, *args, **kwargs)
        self.group_col = group_col
        self.eval_fn = eval_fn
        self.threshold = threshold
        self.compare_max_diff = compare_max_diff
        self.threshold_is_percentage = threshold_is_percentage
        self.metric_by_group = None

    def info(self):
        if self.metric_by_group is not None:
            return {"metric_by_group": self.metric_by_group}
        else:
            return None

    def _compare_max_diff(self):
        max_value = self.metric_by_group.max()
        min_value = self.metric_by_group.min()
        if self.threshold_is_percentage:
            self.diff = (max_value - min_value) / min_value
        else:
            self.diff = max_value - min_value
        if self.diff > self.threshold:
            raise FailedTestError((
                f"Test failed because max difference {round(self.diff,3)} is higher than threshold {self.threshold}"
                f" for groups in {self.group_col}. "
                f" Group {self.metric_by_group.idxmax()} has a metric of {round(max_value,3)}"
                f" and Group {self.metric_by_group.idxmin()} has a metric of {round(min_value,3)}"
            ))

    def _compare_diff_with_mean(self):
        self.mean_metric = self.eval_fn(self.base_dataset)
        if self.threshold_is_percentage:
            self.diff = (self.metric_by_group - self.mean_metric) / self.mean_metric
        else:
            self.diff = (self.mean_metric - self.metric_by_group)
        mean_diff_higher = self.diff.abs() > self.threshold
        if any(mean_diff_higher):
            raise FailedTestError((
                f"Test failed because the metric difference in "
                f"groups {self.metric_by_group[mean_diff_higher].index.values} "
                f"is higher than threshold with respect the mean in the dataset"
            ))

    def run(self, *args, **kwargs):
        """
        Runs the test.

        Raises FailedTestError if metric difference above the threshold is detected
        """
        super().run(*args, **kwargs)

        self.metric_by_group = self.base_dataset.groupby(self.group_col).apply(lambda x: self.eval_fn(x))

        if self.compare_max_diff:
            self._compare_max_diff()
        else:
            self._compare_diff_with_mean()


class SampleLeakingTest(RobustDataTest):
    """
    This test looks if there are samples in the test dataset that are identical to samples in the base/train dataset.
    It considers that a sample is the same if it contains the same values for all the columns. If the number of duplicated
    samples is higher than the allowed by the `threshold` then the test fails.

    Args:
        base_dataset (pandas.DataFrame): Training/Base dataset.
        test_dataset (pandas.DataFrame): Test dataset that we look if we have the same samples than in the
            `base_dataset`. It must contain the same columns as the `base_dataset`
        ignore_feats (List): List of the name of the columns that we want to ignore when comparing the samples
        threshold (Union[float, int]): max percentage (float) or number (int) of samples that we allowed to be duplicated.
            By default is 0
        use_hash (bool): If True, it will create a hash for the rows in the dataframes in order to find duplicates. If False
            it will use `duplicate()` pandas method. Using hash usually results in faster execution in bigger datasets. If
            None it will use one method or other depending of the available memory
        name (str): A name for the test. If not used, it will take the name of the class.

    Example:
        ```python
        >>> from mercury.robust.data_tests import SampleLeakingTest
        >>> test = SampleLeakingTest(base_dataset=my_train_dataframe, test_dataset=my_test_dataset)
        >>> test.run()
        >>> # Check test result details
        >>> test.info()
        ```
    """

    def __init__(
        self,
        base_dataset: "pandas.DataFrame",  # noqa: F821
        test_dataset: "pandas.DataFrame",  # noqa: F821
        ignore_feats: List[str] = None,
        threshold: Union[int, float] = 0,
        name: str = None,
        use_hash: bool = None,
        *args: Any, **kwargs: Any
    ):
        super().__init__(base_dataset, name, *args, **kwargs)
        self.test_dataset = test_dataset
        self.ignore_feats = ignore_feats if ignore_feats else []
        self.threshold = threshold
        self.use_hash = use_hash
        self._is_duplicated_test = None

    def info(self):

        if self._is_duplicated_test is None:
            return None

        return {
            'num_duplicated': self._is_duplicated_test.sum(),
            'percentage_duplicated': self._is_duplicated_test.sum() / len(self.test_dataset)
        }

    def run(self, *args, **kwargs):
        """
        Runs the test.

        Raises FailedTestError if samples in test set existing in train are above threshold
        """
        self._verify_datasets()
        self._is_duplicated_test = self._find_sample_leaking()

        if self._high_number_of_duplicates():
            raise FailedTestError(
                f"Num of samples in test set that appear in train set is {self._is_duplicated_test.sum()} "
                f"(a proportion of {round(self._is_duplicated_test.sum() / len(self.test_dataset), 3)} test samples )"
                f"and the max allowed is {self.threshold}")

    def _find_sample_leaking(self):

        if self.use_hash is None:
            self.use_hash = self._infer_use_hashing()

        if self.use_hash:
            return self._find_sample_leaking_hash_impl()
        else:
            return self._find_sample_leaking_duplicated_impl()

    def _infer_use_hashing(self):
        if psutil.virtual_memory().percent > 20.0:
            return True
        else:
            return False

    def _find_sample_leaking_duplicated_impl(self):

        # Concatenate the datasets without duplicates inside of the same sets
        cols = [c for c in self.base_dataset.columns if c not in self.ignore_feats]
        base_dataset_no_duplicates = self.base_dataset[cols].drop_duplicates()
        test_dataset_no_duplicates = self.test_dataset[cols].drop_duplicates()
        concat_dataset = pd.concat([base_dataset_no_duplicates, test_dataset_no_duplicates])

        # Find the duplicates in the concatenated dataset
        is_duplicated_concat = concat_dataset.duplicated(keep=False).values

        # Return boolean array of elements in test set that appear in train
        return is_duplicated_concat[len(base_dataset_no_duplicates):]

    def _find_sample_leaking_hash_impl(self):
        cols = [c for c in self.base_dataset.columns if c not in self.ignore_feats]

        base_dataset_hash = pd.util.hash_pandas_object(self.base_dataset[cols], index=False)
        base_dataset_hash = base_dataset_hash.drop_duplicates()

        test_dataset_hash = pd.util.hash_pandas_object(self.test_dataset[cols], index=False)
        test_dataset_hash = test_dataset_hash.drop_duplicates()

        concat_hashes = pd.concat([base_dataset_hash, test_dataset_hash])

        is_duplicated_concat = concat_hashes.duplicated(keep=False).values
        return is_duplicated_concat[len(base_dataset_hash):]

    def _verify_datasets(self):

        base_dataset_cols = set(self.base_dataset.columns)
        test_dataset_cols = set(self.test_dataset.columns)
        if not base_dataset_cols == set(self.test_dataset.columns):

            message_error = "Both datasets must contain the same columns. "
            base_cols_not_in_test = base_dataset_cols.difference(test_dataset_cols)
            if len(base_cols_not_in_test) > 0:
                message_error += f"Columns {base_cols_not_in_test} appear in base_dataset but not in test_dataset. "
            test_cols_not_in_base = test_dataset_cols.difference(base_dataset_cols)
            if len(test_cols_not_in_base) > 0:
                message_error += f"Columns {test_cols_not_in_base} appear in test_dataset but not in base_dataset. "

            raise ValueError(message_error)

    def _high_number_of_duplicates(self):

        if isinstance(self.threshold, int) or self.threshold.is_integer():
            return self._is_duplicated_test.sum() > self.threshold
        else:
            return (self._is_duplicated_test.sum() / len(self.test_dataset)) > self.threshold


class NoDuplicatesTest(RobustDataTest):
    """
    This test checks no duplicated samples are present in a dataframe. In case of
    having duplicated samples it could bias your model and/or evaluation metrics.

    Args:
        dataset (pandas.DataFrame): Dataset in which duplicates will be looked for.
        ignore_feats: List of columns that will be ignored when evaluating whether two samples are equal or not
        use_hash (bool): If True, it will create a hash for the rows in the dataframes in order to find duplicates. If False
            it will use `duplicate()` pandas method. Using hash usually results in faster execution in bigger datasets. If
            None it will use one method or other depending of the available memory
        name (str): A name for the test. If not used, it will take the name of the class.

    Example:
        ```python
        >>> from mercury.robust.data_tests import NoDuplicatesTest
        >>> test = NoDuplicatesTest(mydataframe)
        >>> test.run()
        >>> # Check test result details
        >>> test.info()
        ```
    """

    def __init__(
        self,
        dataset: "pandas.DataFrame",  # noqa: F821
        ignore_feats: List[str] = None,
        name: str = None,
        use_hash: bool = None,
        *args: Any, **kwargs: Any
    ):
        super().__init__(dataset, name, *args, **kwargs)
        self.ignore_feats = ignore_feats if ignore_feats else []
        if use_hash is None:
            use_hash = psutil.virtual_memory().percent > 50.0
        self.use_hash = use_hash
        self._num_duplicates = None

    def info(self):
        if self._num_duplicates is None:
            return None

        return {
            'num_duplicated': self._num_duplicates,
            'index_duplicates': self._index_duplicates
        }

    def run(self, *args, **kwargs):
        """
        Runs the test.

        Raises FailedTestError if there exist duplicated samples in the provided dataframe
        """
        dataset = self.base_dataset[[x for x in self.base_dataset.columns if x not in self.ignore_feats]]
        if not self.use_hash:
            duplicates = dataset.duplicated()
        else:
            hashes = pd.util.hash_pandas_object(dataset, index=False)
            duplicates = hashes.duplicated()

        self._num_duplicates = duplicates.sum()
        self._index_duplicates = duplicates[duplicates].index

        if self._num_duplicates > 0:
            raise FailedTestError(
                f"Your dataset has {self._num_duplicates} duplicates. Drop or inspect them via the `info` method"
            )
