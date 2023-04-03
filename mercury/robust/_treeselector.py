from sklearn.tree import (
    DecisionTreeRegressor,
    DecisionTreeClassifier
)
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.base import (
    BaseEstimator,
    TransformerMixin
)
from sklearn.utils.validation import check_is_fitted


import warnings
import numpy as np


class TreeSelector(TransformerMixin, BaseEstimator):
    """
    Feature selection using shallow trees.
    This selector will order the features by its predictive power w.r.t. the
    target variable. The purpose is to select the features which more easily
    divide the target space by fitting a simple tree (usually 3-4 level depth)
    for each feature.

    Args:
        task (string): 'classification' or 'regression'
        max_depth (int): Maximum depth of the fitted trees. Default is three.
        metric (function): Sklearn style metric which accepts (y_true, y_pred) parameters.
            As default, mse will be used for regression task and accuracy for classification.
        use_proba (bool): Whether the passed metric expects probabilities or labels.
            Default is False (Trees will use labels). This is only used for classifiction tasks.
        lower_better(bool): The columns with a lower asscociated metric will be considered more
            important (usually when chosing metrics like mse, mae...). For classification tasks
            this should be False. Default = True.
        tree_kwargs (dict): .

    """

    def __init__(self, task, max_depth=3,
                 metric=None,
                 use_proba=False,
                 lower_better=True,
                 tree_kwargs={}):
        """Constructor"""
        if task != 'classification' and task != 'regression':
            raise ValueError("Choose one of [classification, regression] task")

        if metric is None:
            metric = mean_squared_error if task == 'regression' else accuracy_score

        if metric == accuracy_score and lower_better:
            lower_better = False

        self.task = task
        self.max_depth = max_depth
        self.metric = metric
        self.use_proba = use_proba
        self.tree_kwargs = tree_kwargs
        self.lower_better = lower_better

    def _tree_importance(self, feature, target, random_state=42):
        """ Fits a tree to the feature, trying to predict the target.

        Args:
            feature (np.array): with the feature values.
            target (np.array): target to predict

        Returns:
            The obtained metric for the fit.
        """
        if self.task == 'classification':
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                          random_state=random_state,
                                          **self.tree_kwargs).fit(feature, target)

            if self.use_proba:
                p = tree.predict_proba(feature)
            else:
                p = tree.predict(feature)

            # Needed to suppress warnings in case a tree only predicts one class.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = self.metric(target, p)
        else:
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         random_state=random_state,
                                         **self.tree_kwargs).fit(feature, target)
            p = tree.predict(feature)
            res = self.metric(target, p)
        return res

    def fit(self, X, y, tree_random_state=42) -> "TreeSelector":
        """ Fits the estimator and calculates the features importances

        Args:
            X (np.array or pd.DataFrame): Features
            y (np.array or pd.Series): Target variable
            reverse (bool): Whether to reverse the order of the result

        Returns:
            (dict): dictionary with two keys:

                - features: Ordered feature indices by the obtained metric\n
                - metrics: value of the metric for each feature.
            (TreeSelector): the estimator itself
        """
        num_features = X.shape[-1]
        errors = []
        for feat in range(num_features):
            err = self._tree_importance(np.array(X)[:, feat].reshape(-1, 1), y,
                                        random_state=tree_random_state)
            errors.append(err)

        # Sorted feature indexes
        self.ordered_feat_idx_ = np.argsort(errors) if self.lower_better else np.argsort(errors)[::-1]

        # Sorted feature errors (or the chosen metric)
        self.ordered_feat_importances_ = [errors[e] for e in self.ordered_feat_idx_]

        return self

    def transform(self, X, keep_top_n=None):
        check_is_fitted(self, ["ordered_feat_idx_", "ordered_feat_importances_"])
        filtered_data = None
        if type(X) == "pd.DataFrame":
            filtered_data = X.iloc[:, self.ordered_feat_idx_[:keep_top_n]]
        else:
            filtered_data = X[:, self.ordered_feat_idx_[:keep_top_n]]

        return filtered_data

    @property
    def feature_importances(self):
        return dict(features=self.ordered_feat_idx_, metrics=self.ordered_feat_importances_)