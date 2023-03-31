import numpy as np
import typing as tp

from abc import ABC


class TreeCoverageAnalyzer(ABC):
    def __init__(self):
        pass

class SkLearnTreeCoverageAnalyzer(TreeCoverageAnalyzer):
    """ This class acts as an analyzer for Scikit-Learn Trees (DecisionTreeClassifier/Regressor...)

    At the moment, it is only intended for evaluating the total leaf coverage achieved using a
    particular test data frame.

    Args:
        estimator: Fitted Scikit-Learn tree based classifier/regressor
    """
    def __init__(
        self,
        estimator: tp.Union['sklearn.tree.DecisionTreeClassifier',  # noqa: F821
                            'sklearn.tree.DecisionTreeRegressor']   # noqa: F821
    ):
        self.estimator = estimator

    def analyze(
            self,
            X: tp.Union['np.ndarray', 'pd.DataFrame', 'scipy.sparse.csr_matrix']  # noqa: F821
    ) -> "SkLearnTreeCoverageAnalyzer":
        """ Analyzes the tree. At the moment, only coverage is calculated.
        Analysis stats can be accesed after calling this method.

        Args:
            X: (test) Dataset which will be analyzed.

        Returns:
            self, the "fitted" Tree analyzer.
        """

        if hasattr(self, "all_leaves_"):
            del self.all_leaves_
            del self.covered_leaves_

        _ = self.get_all_leaves()
        coverage = self.get_covered_leaves(X)
        self.coverage_ = coverage
        return self

    def get_percent_coverage(self) -> float:
        """ Given test set (from the analyze method), get the percentage of total
        leaves that recieved at least one sample
        """
        num_unvisited = len(self.unvisited_leaves())
        return 1 - (num_unvisited / self.get_num_leaves())

    def get_all_leaves(self) -> np.array:
        """ Returns the node-id's of the leaves.
        """
        if not hasattr(self, "all_leaves_"):
            self.all_leaves_ = np.argwhere(self.estimator.tree_.children_right == -1).flatten()
        return self.all_leaves_

    def get_num_leaves(self) -> int:
        """ Returns the total number of leaves in the tree
        """
        return self.estimator.tree_.n_leaves

    def get_covered_leaves(
        self,
        X: tp.Union['np.ndarray', 'pd.DataFrame', 'scipy.sparse.csr_matrix']  # noqa: F821
    ) -> dict:
        """ Returns a dictionary in which the key is a node-id of a leaf and the value
        is the total number of samples of X that fell in each leaf.
        """
        if not hasattr(self, "covered_leaves_"):
            preds = self.estimator.apply(X)
            cnt = dict()
            for lid in self.get_all_leaves():
                cnt[lid] = np.sum(preds == lid)
            self.covered_leaves_ = cnt

        return self.covered_leaves_

    def unvisited_leaves(self):
        """ Returns a set with the node-ids of unvisited leafs.
        """
        if not hasattr(self, "unvisited_labels_leaves_"):
            self.unvisited_leaves_ = {key for key, val in self.coverage_.items() if val == 0}

        return self.unvisited_leaves_


class SkLearnTreeEnsembleCoverageAnalyzer(TreeCoverageAnalyzer):
    """ This class acts as an analyzer for Scikit-Learn Ensemble Tree based models
    (RandomForestClassifier/Regressor...) or anything that has an `estimators_` attribute
    where each estimator has a `tree_` attribute once fitted.

    At the moment, it is only intended for evaluating the total leaf coverage achieved using a
    particular test dataset.

    Args:
        estimator: Fitted Scikit-Learn tree based classifier/regressor
    """
    def __init__(
        self,
        estimator: tp.Union['sklearn.ensemble.RandomForestClassifier',  # noqa: F821
                            'sklearn.ensemble.RandomForestRegressor',   # noqa: F821
                            ]
    ):
        self.estimator = estimator
        self.tree_analyzers = [SkLearnTreeCoverageAnalyzer(est) for est in estimator.estimators_]

    def analyze(self, X) -> "SkLearnTreeEnsembleCoverageAnalyzer":
        """ Analyzes the ensemble. At the moment, only coverage is calculated.
        Analysis stats can be accesed after calling this method.

        Args:
            X: (test) Dataset which will be analyzed.

        Returns:
            self, the "fitted" Tree analyzer.
        """
        covs = [t.analyze(X) for t in self.tree_analyzers]
        self.coverage_ = covs
        return self

    def get_unvisited_leaves(self) -> tp.List[set]:
        """ Returns a list of sets in which each item i is the set of unvisited node-ids
        for each internal tree i.
        """
        unvisited_labels_ = list(
            filter(lambda a: len(a) > 0, [a.unvisited_leaves() for a in self.tree_analyzers])
        )
        return unvisited_labels_

    def get_covered_leaves(self) -> tp.List[dict]:
        """ Returns a list of dicts in which each item i is a dict counting how many items fell into one
        of the leaves of the tree i.
        """
        unvisited_labels_ = [a.covered_leaves_ for a in self.tree_analyzers]
        return unvisited_labels_

    def get_all_leaves(self) -> tp.List[np.array]:
        """ Returns a list of lists in which each item contains the node-ids of the leaves of
        each internal tree.
        """
        return [t.get_all_leaves() for t in self.tree_analyzers]

    def get_num_leaves(self) -> int:
        """ Returns the total number of leaves in the ensemble
        """
        return np.sum([t.get_num_leaves() for t in self.tree_analyzers])

    def get_percent_coverage(self) -> float:
        """ Given test set (from the analyze method), gets the percentage of total
        leaves that recieved at least one sample over all the internal trees in the ensemble.
        """
        total_leaves = np.sum(
            [a.get_num_leaves() for a in self.tree_analyzers]
        )

        num_unvisited = np.sum([len(a) for a in self.get_unvisited_leaves()])
        return 1 - (num_unvisited / total_leaves)