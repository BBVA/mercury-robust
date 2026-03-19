from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
from sklearn.tree import DecisionTreeClassifier

from mercury.robust._category_struct import CategoryStruct
from mercury.robust._label_cleaning import get_confident_joint
from mercury.robust._linalg import lin_combs_in_columns
from mercury.robust._tree_analysis import (
    SkLearnTreeCoverageAnalyzer,
    SkLearnTreeEnsembleCoverageAnalyzer,
    TreeCoverageAnalyzer,
)
from mercury.robust._treeselector import TreeSelector
from mercury.robust.basetests import RobustDataTest, RobustModelTest, RobustTest, TestState
from mercury.robust.errors import FailedTestError
from mercury.robust.suite import TestSuite as RobustTestSuite


class PassiveTest(RobustTest):
    pass


class RecordingTest(RobustTest):
    def __init__(self, recorder, behavior="pass", name=None):
        super().__init__(name=name)
        self.recorder = recorder
        self.behavior = behavior

    def info(self):
        return {"behavior": self.behavior}

    def run(self, *args, **kwargs):
        self.recorder.append(self.name)
        if self.behavior == "fail":
            raise FailedTestError(f"{self.name} failed")
        if self.behavior == "error":
            raise RuntimeError(f"{self.name} errored")


def test_basetest_default_methods_and_holders():
    test = PassiveTest()

    assert test.name == "PassiveTest"
    assert str(test) == "PassiveTest"
    assert repr(test) == "PassiveTest"
    assert test.info() is None
    assert test.get_error_message() == ""
    assert test.get_state() == TestState.NOT_EXECUTED
    assert test.run() is None

    data_test = RobustDataTest(base_dataset="dataset")
    model_test = RobustModelTest(model="model")
    assert data_test.base_dataset == "dataset"
    assert model_test.model == "model"


def test_suite_selective_run_and_results_dataframe():
    recorder = []
    tests = [
        RecordingTest(recorder, name="first"),
        RecordingTest(recorder, behavior="fail", name="second"),
        RecordingTest(recorder, behavior="error", name="third"),
    ]

    suite = RobustTestSuite(tests=tests, run_safe=True)
    assert suite._test_should_run(tests[0], 0, None) is True
    assert suite._test_should_run(tests[0], 0, ["first"]) is True
    assert suite._test_should_run(tests[1], 1, [1]) is True
    assert suite._test_should_run(tests[2], 2, ["missing"]) is False

    suite.run(tests_to_run=["second", 2])
    assert recorder == ["second", "third"]

    results = suite.get_results_as_df()
    assert results["name"].tolist() == ["first", "second", "third"]
    assert results["state"].tolist() == [
        TestState.NOT_EXECUTED,
        TestState.FAIL,
        TestState.EXCEPTION,
    ]
    assert results["info"].tolist()[1] == {"behavior": "fail"}

    strict_suite = RobustTestSuite([RecordingTest([], name="only")], run_safe=False)
    strict_suite.run([0])


def test_category_struct_helpers_cover_edge_cases():
    df = pd.DataFrame(
        {
            "city": ["A", "A", "B", "B"],
            "city_copy": ["A", "A", "B", "B"],
            "zone": ["north", "south", "north", "south"],
            "value": [1, 2, 3, 4],
        }
    )

    assert CategoryStruct.categoricals(df) == ["city", "city_copy", "zone"]
    assert CategoryStruct.cardinality_set_of_rows(df, []) == 1
    assert CategoryStruct.synonyms(df, ["city"]) == []
    assert CategoryStruct.synonyms(df) == [("city", "city_copy")]

    complete_sets = CategoryStruct.all_complete_sets(df, ["city", "city_copy", "zone"])
    assert ["city", "city_copy", "zone"] in complete_sets
    assert ["city_copy", "zone"] in complete_sets
    assert ["city", "zone"] in complete_sets

    redundant = CategoryStruct.individually_redundant(df, ["city", "city_copy", "zone"])
    assert redundant == ["city", "city_copy"]
    assert CategoryStruct.individually_redundant(df, ["city"]) == []


def test_category_struct_default_and_nested_complete_set_paths():
    df_single = pd.DataFrame({"only_cat": ["x", "y"], "value": [1, 2]})
    assert CategoryStruct.all_complete_sets(df_single) == [["only_cat"]]
    assert CategoryStruct.individually_redundant(df_single) == []

    df_break = pd.DataFrame(
        {
            "a": ["x", "x", "y", "y"],
            "b": ["u", "v", "w", "u"],
            "c": ["m", "n", "o", "p"],
        }
    )
    assert CategoryStruct.synonyms(df_break, ["a", "b", "c"]) == []

    df_nested = pd.DataFrame(
        {
            "a": ["x", "y", "x", "y"],
            "b": ["x", "y", "x", "y"],
            "c": ["x", "y", "x", "y"],
        }
    )
    assert ["c"] in CategoryStruct.all_complete_sets(df_nested, ["a", "b", "c"])


def test_linalg_detects_zero_columns_and_zero_rows():
    result = lin_combs_in_columns([[0.0], [0.0]])
    assert result.shape == (1, 1)
    assert np.allclose(result, np.array([[1.0]]))

    result = lin_combs_in_columns([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [2.0, 0.0, 2.0]])
    assert result.shape[1] == 3
    assert any(np.allclose(row, np.array([0.0, 1.0, 0.0])) for row in result)

    stacked = lin_combs_in_columns([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    assert stacked.shape[0] >= 2


def test_get_confident_joint_builds_default_classifier(monkeypatch):
    captured = {}
    expected_psx = np.array([[0.9, 0.1], [0.2, 0.8]])
    expected_joint = np.array([[1, 0], [0, 1]])

    def fake_estimate_cv_predicted_probabilities(X, labels, clf, cv_n_folds, seed):
        captured["clf"] = clf
        captured["cv_n_folds"] = cv_n_folds
        captured["seed"] = seed
        return expected_psx

    def fake_compute_confident_joint(labels, pred_probs):
        captured["pred_probs"] = pred_probs
        return expected_joint

    monkeypatch.setattr(
        "mercury.robust._label_cleaning.estimate_cv_predicted_probabilities",
        fake_estimate_cv_predicted_probabilities,
    )
    monkeypatch.setattr(
        "mercury.robust._label_cleaning.compute_confident_joint",
        fake_compute_confident_joint,
    )

    result = get_confident_joint(np.array([[0.0], [1.0]]), np.array([0, 1]), n_folds=3, seed=7)

    assert isinstance(captured["clf"], LogisticRegression)
    assert captured["cv_n_folds"] == 3
    assert captured["seed"] == 7
    assert np.array_equal(captured["pred_probs"], expected_psx)
    assert np.array_equal(result, expected_joint)


def test_tree_analyzers_reset_caches_and_expose_helpers():
    TreeCoverageAnalyzer()
    X, y = make_classification(n_samples=40, n_features=4, random_state=0)

    tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    analyzer = SkLearnTreeCoverageAnalyzer(tree).analyze(X[:5])
    first_coverage = analyzer.coverage_.copy()
    analyzer.analyze(X)
    assert analyzer.coverage_ != first_coverage
    assert set(analyzer.get_covered_leaves(X)) == set(analyzer.get_all_leaves())
    assert 0 <= analyzer.get_percent_coverage() <= 1

    from sklearn.ensemble import RandomForestClassifier

    ensemble = RandomForestClassifier(n_estimators=3, max_depth=2, random_state=0).fit(X, y)
    ensemble_analyzer = SkLearnTreeEnsembleCoverageAnalyzer(ensemble).analyze(X[:5])
    assert isinstance(ensemble_analyzer, TreeCoverageAnalyzer)
    assert len(ensemble_analyzer.get_all_leaves()) == 3
    assert len(ensemble_analyzer.get_covered_leaves()) == 3
    assert ensemble_analyzer.get_num_leaves() >= 3
    assert isinstance(ensemble_analyzer.get_unvisited_leaves(), list)
    assert 0 <= ensemble_analyzer.get_percent_coverage() <= 1


def test_tree_selector_constructor_importance_and_transform_paths():
    with pytest.raises(ValueError):
        TreeSelector("clustering")

    selector_cls = TreeSelector("classification")
    assert selector_cls.metric is accuracy_score
    assert selector_cls.lower_better is False

    selector_reg = TreeSelector("regression")
    assert selector_reg.metric is mean_squared_error
    assert selector_reg.lower_better is True

    X = np.array([[0.0], [1.0], [0.0], [1.0]])
    y = np.array([0, 1, 0, 1])
    selector_proba = TreeSelector(
        "classification",
        metric=log_loss,
        use_proba=True,
        lower_better=True,
    )
    assert selector_proba._tree_importance(X, y) >= 0

    with pytest.raises(NotFittedError):
        selector_cls.transform(np.array([[0.0, 1.0]]))

    selector_cls.fit(np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.5], [1.0, 0.2]]), y)
    transformed = selector_cls.transform(np.array([[0.0, 1.0], [1.0, 0.0]]), keep_top_n=1)
    assert transformed.shape == (2, 1)
