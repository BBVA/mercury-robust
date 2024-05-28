import pytest
import seaborn as sns
import numpy as np
import pandas as pd

from mercury.dataschema import DataSchema
from mercury.dataschema.feature import FeatType
from mercury.robust.data_tests import (
    SameSchemaTest,
    DriftTest,
    LinearCombinationsTest,
    LabelLeakingTest,
    NoisyLabelsTest,
    CohortPerformanceTest,
    SampleLeakingTest,
    NoDuplicatesTest
)
from mercury.robust.errors import FailedTestError

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification


@pytest.fixture(scope="module")
def datasets():
    tips = sns.load_dataset("tips")
    tips["sex"] = tips["sex"].astype(str)
    tips["smoker"] = tips["smoker"].astype(str)
    tips["day"] = tips["day"].astype(str)
    tips["time"] = tips["time"].astype(str)

    titanic = sns.load_dataset("titanic")
    isna_deck = titanic.deck.isna()
    titanic["class"] = titanic["class"].astype(str)
    titanic["deck"] = titanic["deck"].astype(str)
    titanic["who"] = titanic["who"].astype(str)
    titanic.loc[isna_deck, "deck"] = np.nan

    return tips, titanic


def test_sameschema(datasets):
    tips, titanic = datasets
    titanic2 = titanic.copy()
    titanic2 = titanic.drop("deck", axis=1)

    schma = DataSchema().generate(titanic).calculate_statistics()
    schma2 = DataSchema().generate(titanic2).calculate_statistics()

    with pytest.raises(FailedTestError) as exinfo:
        SameSchemaTest(titanic2, schma).run()

    # SameSchemaTest(titanic, schma, schema_cat_threshold=0.01).run()
    SameSchemaTest(titanic, schma).run()

def test_sameschema_custom_feature_map(datasets):

    tips, titanic = datasets
    titanic2 = titanic.copy()

    force_types = {
        'sex': FeatType.CATEGORICAL,
        'pclass': FeatType.DISCRETE
    }
    schma_reference = DataSchema().generate(titanic, force_types=force_types).calculate_statistics()

    # Same types are forced. the test runs fine
    test = SameSchemaTest(
        titanic2, 
        schma_reference,
        custom_feature_map={
            'pclass': FeatType.DISCRETE,
            'sex': FeatType.CATEGORICAL
        }
    )
    test.run()


def test_drift(datasets):
    tips, titanic = datasets

    schma = DataSchema().generate(titanic).calculate_statistics()

    # Simulate a dataset with drift on the age feature
    titanic2 = titanic.copy()
    titanic2["age"] *= titanic2["age"]  # quadratic relation

    drift_test = DriftTest(titanic2, schma)
    assert drift_test.info() is None
    with pytest.raises(FailedTestError) as exinfo:
        drift_test.run()

    assert "age" in str(exinfo.value)

    # Simulate a dataset with drift on the sex feature. Only males are present
    titanic2 = titanic.copy()
    titanic2["sex"] = "male"

    with pytest.raises(FailedTestError) as exinfo:
        DriftTest(titanic2, schma).run()

    assert "sex" in str(exinfo.value)

    # This shouldn't fail
    drift_test = DriftTest(titanic, schma)
    drift_test.run()
    assert 'cat_drift_metrics' in drift_test.info()

    # Dataset without any Continuous Features should work without raising exception (Previous bug)
    titanic_cat = titanic[["sex", "embarked"]]
    schma_cat = DataSchema().generate(titanic_cat).calculate_statistics()
    drift_test = DriftTest(titanic_cat, schma_cat)
    drift_test.run()
    # Dataset without any Categorical Feature should work without raising exception (Previous bug)
    titanic_cont = titanic[["age", "fare"]]
    schma_cont = DataSchema().generate(titanic_cont).calculate_statistics()
    drift_test = DriftTest(titanic_cont, schma_cont)
    drift_test.run()

def test_drift_different_domain_categoricals(datasets):

    tips, titanic = datasets

    # Same number of categories, but one category changes in the test set
    titanic_ref = titanic[["sex", "embarked"]].copy()
    titanic_test = titanic_ref.copy()

    titanic_test = titanic_test.replace({'sex': {'male': 'm'}})

    schma_ref = DataSchema().generate(titanic_ref).calculate_statistics()
    drift_test = DriftTest(titanic_test, schma_ref)
    with pytest.raises(FailedTestError) as exinfo:
        drift_test.run()


    # test set has one category more
    titanic_ref = titanic[["sex", "embarked"]].copy()
    titanic_test = titanic_ref.copy()

    new_records = titanic_test.iloc[0:5].copy()
    new_records['embarked'] = 'new_value'
    titanic_test = pd.concat([titanic_test, new_records])

    schma_ref = DataSchema().generate(titanic_ref).calculate_statistics()
    drift_test = DriftTest(titanic_test, schma_ref)
    with pytest.raises(FailedTestError) as exinfo:
        drift_test.run()


    # Test has one category less, in this case it fails because of drift
    titanic_ref = titanic[["sex", "embarked"]].copy()
    titanic_test = titanic_ref.copy()
    titanic_test = titanic_test[titanic_test['embarked'] != 'Q']

    schma_ref = DataSchema().generate(titanic_ref).calculate_statistics()
    drift_test = DriftTest(titanic_test, schma_ref)
    with pytest.raises(FailedTestError) as exinfo:
        drift_test.run()

def test_drift_target_samples_outside_source_bin_limits():
    # This is is testing the case where the target distribution in a continous variable contains samples 
    # that are outside of the limits of the binning created in the source distribution


    # Case 1: Target samples contain values higher than the max bin
    df_source = pd.DataFrame(
        data=[0, 0.5, 1, 2., 2.3, 5, 6., 7, 8, 9, 9.5, 10],
        columns=["f1"]
    )
    df_target = pd.DataFrame(
        data=[11., 11.3, 12, 13, 14, 15, 16, 17],
        columns=["f1"]
    )
    schema_source = DataSchema().generate(df_source).calculate_statistics()
    drift = DriftTest(df_target, schema_ref=schema_source)
    with pytest.raises(FailedTestError) as exinfo:
        drift.run()

    # Case 2: Target samples contain values lower than min bin and higher than max bin
    df_source = pd.DataFrame(
        data=[0, 0.5, 1, 2., 2.3, 5, 6., 7, 8, 9, 9.5, 10],
        columns=["f1"]
    )
    df_target = pd.DataFrame(
        data=[-4., -3., -1, 3., 5.3, 7, 9, 14, 15, 16, 17],
        columns=["f1"]
    )
    schema_source = DataSchema().generate(df_source).calculate_statistics()
    drift = DriftTest(df_target, schema_ref=schema_source)
    with pytest.raises(FailedTestError) as exinfo:
        drift.run()

    # Case 3: Only one sample (of many) outside de limits of distribution shouldn't produce drift
    df_source = pd.DataFrame(
        data=[0.]*20 + [1.1]*20 + [3.3]*20 + [6.6]*20 + [7.7]*20 + [9.9]*20 + [10.]*20 + [12.]*20 + [15.]*20 + [20.]*20,
        columns=["f1"]
    )
    df_target = pd.DataFrame(
        data=[0.]*20 + [1.1]*20 + [3.3]*20 + [6.6]*20 + [7.7]*20 + [9.9]*20 + [10.]*20 + [12.]*20 + [15.]*20 + [20.]*20 + [23.],
        columns=["f1"]
    )
    schema_source = DataSchema().generate(df_source).calculate_statistics()
    drift = DriftTest(df_target, schema_ref=schema_source)
    drift.run()


def test_label_leaking(datasets):
    tips, titanic = datasets

    titanic = titanic.dropna()
    tips = tips.dropna()

    # Remove category
    tips["sex"] = tips["sex"].astype(str)
    tips["smoker"] = tips["smoker"].astype(str)
    tips["day"] = tips["day"].astype(str)
    tips["time"] = tips["time"].astype(str)

    # Label encoding
    for feat in ["sex", "smoker", "day", "time"]:
        tips.loc[:, feat] = preprocessing.LabelEncoder().fit_transform(
            tips.loc[:, feat]
        )

    # Categoricals need to be label encoded
    le = preprocessing.LabelEncoder()
    categoricals = [
        "sex",
        "class",
        "who",
        "deck",
        "embark_town",
        "alive",
        "alone",
        "embarked",
    ]
    for f in categoricals:
        titanic[f] = le.fit_transform(titanic.loc[:, f])

    # Only classification / regression possible
    with pytest.raises(ValueError) as exinfo:
        test = LabelLeakingTest(titanic, task="wrong", label_name="survived")
        test.run()
    assert "classification" in str(exinfo.value)

    test = LabelLeakingTest(titanic, label_name="survived")

    with pytest.raises(FailedTestError) as exinfo:
        test.run()
    assert "alive" in str(exinfo.value)

    # This will not fail
    titanic2 = titanic.drop("alive", axis=1)

    x, y = make_classification(n_samples=10_000, n_features=30)
    x = pd.DataFrame(x)
    x["target"] = y
    x["injected"] = y + np.random.normal(0, 0.2, len(y))

    test = LabelLeakingTest(
        x,
        label_name="target",
    )

    with pytest.raises(FailedTestError) as exinfo:
        test.run()
    assert "injected" in str(exinfo.value)

    # test with regression and inject a nonlinear relationship
    tips["injected"] = tips["total_bill"] ** 2
    test = LabelLeakingTest(tips, label_name="total_bill", threshold=0.03)

    with pytest.raises(FailedTestError) as exinfo:
        test.run()
    assert "injected" in str(exinfo.value)

    # Test ignoring features.
    # In this case, it will pass because we ignore the injected features
    test = LabelLeakingTest(
        tips, label_name="total_bill", threshold=0.03, ignore_feats=["injected"]
    )
    test.run()


def test_linear_combs_continuous():
    custom_feature_mapping = {
        0: FeatType.CONTINUOUS,
        1: FeatType.CONTINUOUS,
        2: FeatType.CONTINUOUS
    }

    X = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 2.0], [3.0, 2.0, 3.0], [4.0, 3.0, 4.0]])

    with pytest.raises(FailedTestError) as exinfo:
        LinearCombinationsTest(pd.DataFrame(X), schema_custom_feature_map=custom_feature_mapping).run()
    assert "Linear combinations for continuous features were encountered" in str(exinfo.value)

    X = np.array([[-1.0, 4.0, 2.0], [4.0, 6.0, 3.0], [4.0, 8.0, 4.0], [3.0, 10.0, 5.0]])

    with pytest.raises(FailedTestError) as exinfo:
        LinearCombinationsTest(pd.DataFrame(X), schema_custom_feature_map=custom_feature_mapping).run()
    assert "Linear combinations for continuous features were encountered" in str(exinfo.value)

    # This should pass
    X = np.array([[0.0, 1.0, 2.0], [2.0, 2.0, 5.0], [5.0, 1.0, 7.0], [0.0, 9.0, 2.0]])
    LinearCombinationsTest(pd.DataFrame(X), schema_custom_feature_map=custom_feature_mapping).run()


def test_linear_combs_categoricals():
    cps = df = pd.DataFrame({
        "cp": [
            "28701",
            "28015",
            "28710",
            "48015",
            "48001",
            "01001",
            "01002",
            "01003",
            "28701",
        ],
        "prov": [
            "Madrid",
            "Madrid",
            "Madrid",
            "Vizcaya",
            "Vizcaya",
            "Álava",
            "Álava",
            "Álava",
            "Madrid",
        ],
        "num": [5432.1, 243.3, 17.6, 22.4, 97.4, 1.2, -2.2, 56.7, 246.0],
    })

    with pytest.raises(FailedTestError) as exinfo:
        LinearCombinationsTest(cps).run()

    assert "prov" in str(exinfo.value)

def test_noisy_labels_tabular(datasets):
    tips, titanic = datasets
    titanic = titanic[["survived", "pclass", "sex", "age", "fare", "who"]].dropna().copy()

    # this test should pass
    from sklearn.ensemble import RandomForestClassifier
    label_issues_args = dict(
        clf=RandomForestClassifier(max_depth=4)
    )
    test = NoisyLabelsTest(
        titanic,
        label_name='survived',
        threshold=0.25,
        label_issues_args=label_issues_args
    )
    test.run()

    # It should pass also using calculate_idx_issues=True
    test = NoisyLabelsTest(
        titanic,
        label_name='survived',
        threshold=0.25,
        calculate_idx_issues=True,
        label_issues_args=label_issues_args
    )
    test.run()

    # this test shouldn't pass
    titanic2 = titanic.copy()
    titanic2["survived"] = np.random.choice([0, 1], replace=True, size=len(titanic2))
    test = NoisyLabelsTest(
        titanic2,
        label_name='survived',
        threshold=0.25,
        label_issues_args=label_issues_args
    )
    with pytest.raises(FailedTestError) as exinfo:
        test.run()

    # Case using own preprocessor
    preprocessor = ColumnTransformer(
        transformers=[('ohe', preprocessing.OneHotEncoder(handle_unknown='ignore'), ["sex", "who"])],
        remainder='passthrough'
    )
    test = NoisyLabelsTest(
        titanic,
        label_name='survived',
        threshold=0.25,
        preprocessor=preprocessor,
        label_issues_args=label_issues_args
    )
    test.run()

    # Case prprocessingis not needed (only continous features)
    test = NoisyLabelsTest(
        titanic[["age", "fare", "survived"]],
        label_name='survived',
        threshold=0.35,
        label_issues_args=label_issues_args
    )
    test.run()

    # Case using ignore features and different configuration for cleanlab
    label_issues_args = dict(
        clf=RandomForestClassifier(max_depth=4),
        sorted_index_method='normalized_margin'
    )
    test = NoisyLabelsTest(
        titanic,
        label_name='survived',
        ignore_feats=["pclass"],
        threshold=0.25,
        calculate_idx_issues=True,
        label_issues_args=label_issues_args
    )
    test.run()


def test_noisy_labels_text():

    df = pd.DataFrame({
        "text": [
            "positive text",
            "positive text",
            "positive text",
            "positive text",
            "positive text",
            "negative text",
            "negative text",
            "negative text",
            "negative text",
            "negative text"
        ],
        "label": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    })

    # this test should pass
    test = NoisyLabelsTest(
        df,
        label_name='label',
        text_col='text',
        threshold=0.25,
    )
    test.run()

    # this test should fail
    size = 500
    df = pd.DataFrame()
    df["text"] = ["text class0"] * 25 + ["text class1"] * 25 + ["text class2"] * 25 + ["text class3"] * 25
    df["label"] = [0] * 25 + [1] * 25 + [2] * 25 + [3] * 25
    df["label"] = np.random.choice([0,1, 2, 3], size=len(df), replace=True)
    test = NoisyLabelsTest(
        df,
        label_name='label',
        text_col='text',
        threshold=0.25,
    )
    with pytest.raises(FailedTestError) as exinfo:
        test.run()

    # Case creating own text feature extraction
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features=1000)
    test = NoisyLabelsTest(
        df,
        label_name='label',
        text_col='text',
        threshold=0.20,
        preprocessor=cv
    )
    with pytest.raises(FailedTestError) as exinfo:
        test.run()


def test_cohort_performance(datasets):
    tips, titanic = datasets
    titanic = titanic[["survived", "age", "fare", "pclass", "sex"]].dropna()

    label_col = "survived"

    # Create pipeline to train model
    numeric_features = ['age', 'fare']
    numeric_transformer = Pipeline(steps=[("scaler", preprocessing.StandardScaler())])
    categorical_features = ['pclass', 'sex']
    categorical_transformer = preprocessing.OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(solver='newton-cg'))]
    )
    # Train
    X_train = titanic[numeric_features + categorical_features]
    y_train = titanic[label_col]
    pipeline = pipeline.fit(X_train, y_train)
    # Generate Predictions
    titanic2 = titanic.copy()
    titanic2["class_pred"] = pipeline.predict(X_train)
    probs = pipeline.predict_proba(X_train)
    col_probs = []
    for i in range(probs.shape[1]):
        titanic2["prob_" + str(i)] = probs[:,i]

    # Test 1 - Accuracy (compare_max_diff=True) - SHOULD PASS
    def eval_acc(df):
        return accuracy_score(df[label_col], df["class_pred"])
    test1 = CohortPerformanceTest(
        base_dataset=titanic2, group_col="sex", eval_fn = eval_acc, threshold = 0.2, compare_max_diff=True,
        threshold_is_percentage = False
    )
    assert test1.info() == None
    test1.run()
    assert test1.metric_by_group.max() - test1.metric_by_group.min() < test1.threshold
    assert 'metric_by_group' in test1.info()

    # Test 2 - Accuracy (compare_max_diff=False) - SHOULD PASS
    test2 = CohortPerformanceTest(
        base_dataset=titanic2, group_col="sex", eval_fn = eval_acc, threshold=0.2, compare_max_diff=False,
        threshold_is_percentage=False
    )
    test2.run()
    assert test2.mean_metric - test1.metric_by_group.min() < test2.threshold

    # Test 3 - Mean Prediction (compare_max_diff=TRUE) SHOULD NOT PASS
    def eval_mean_prediction(df):
        return df["prob_1"].mean()

    test3 = CohortPerformanceTest(
        base_dataset=titanic2, group_col="sex", eval_fn = eval_mean_prediction, threshold = 0.1,
        compare_max_diff=True, threshold_is_percentage=False
    )
    with pytest.raises(FailedTestError) as exinfo:
        test3.run()

    # Test 4 - Mean Prediction (compare_max_diff=FALSE) SHOULD NOT PASS
    test4 = CohortPerformanceTest(
        base_dataset=titanic2, group_col="sex", eval_fn = eval_mean_prediction, threshold = 0.1,
        compare_max_diff=False, threshold_is_percentage=False
    )
    with pytest.raises(FailedTestError) as exinfo:
        test4.run()

    # Test 5 - Mean Prediction (compare_max_diff=True, threshold_is_percentage=True) - SHOULD FAIL
    test5 = CohortPerformanceTest(
        base_dataset=titanic2, group_col="sex", eval_fn = eval_mean_prediction, threshold = 1., compare_max_diff=True,
        threshold_is_percentage = True
    )
    with pytest.raises(FailedTestError) as exinfo:
        test5.run()

    # Test 6 - Mean Predictin (compare_max_diff=False, threshold_is_percentage=True) - SHOULD PASS
    test6 = CohortPerformanceTest(
        base_dataset=titanic2, group_col="sex", eval_fn = eval_mean_prediction, threshold = 1.5, compare_max_diff=False,
        threshold_is_percentage = True
    )
    test6.run()

def test_label_leaking_test_str(datasets):
    tips, titanic = datasets
    titanic = titanic.dropna()

    # Raises ValueError becuase handle_str_cols indicates to raise Error when finding string columns
    test = LabelLeakingTest(titanic, task="classification", label_name="survived", handle_str_cols='error')
    with pytest.raises(ValueError) as exinfo:
        test.run()

    # String columns are transformed the test is executed. Raises FailedTestError because leaking in alive column
    test = LabelLeakingTest(titanic, task="classification", label_name="survived", handle_str_cols='transform')
    with pytest.raises(FailedTestError) as exinfo:
        test.run()

    # Raises ValueError because hand_src_cols param has a wrong value
    test = LabelLeakingTest(titanic, task="classification", label_name="survived", handle_str_cols='wrong value')
    with pytest.raises(ValueError) as exinfo:
        test.run()

def test_sample_leaking_test_numbers():

    synthetic_train = pd.DataFrame(data=[
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2],
        [1.3, 1.4, 1.5],
        [1.6, 1.7, 1.8],
    ], columns=["f1", "f2", "f3"])


    # Contains 2 repeated
    synthetic_test = pd.DataFrame(data=[
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [3.7, 3.8, 3.9],
        [4.0, 4.1, 4.2]
    ], columns=["f1", "f2", "f3"])

    # Allowing 1 duplicate the test fails
    test = SampleLeakingTest(base_dataset=synthetic_train, test_dataset=synthetic_test, ignore_feats=None, threshold=1)
    with pytest.raises(FailedTestError):
        test.run()
    assert test.info()['num_duplicated'] == 2
    # Check for Hash implementation
    test = SampleLeakingTest(base_dataset=synthetic_train, test_dataset=synthetic_test, ignore_feats=None, threshold=1, use_hash=True)
    with pytest.raises(FailedTestError):
        test.run()
    assert test.info()['num_duplicated'] == 2

    # Allowing 3 duplicates the test passes
    test = SampleLeakingTest(base_dataset=synthetic_train, test_dataset=synthetic_test, ignore_feats=None, threshold=3)
    test.run()
    assert test.info()['num_duplicated'] == 2
    # Check for hash implementation
    test = SampleLeakingTest(base_dataset=synthetic_train, test_dataset=synthetic_test, ignore_feats=None, threshold=3, use_hash=True)
    test.run()
    assert test.info()['num_duplicated'] == 2

    # Allowing a percentage of 0.25 duplicated samples the test fails
    test = SampleLeakingTest(base_dataset=synthetic_train, test_dataset=synthetic_test, ignore_feats=None, threshold=0.25, use_hash=False)
    with pytest.raises(FailedTestError):
        test.run()
    assert test.info()['percentage_duplicated'] >= 0.1
    # Check for hash implementation
    test = SampleLeakingTest(base_dataset=synthetic_train, test_dataset=synthetic_test, ignore_feats=None, threshold=0.25, use_hash=True)
    with pytest.raises(FailedTestError):
        test.run()
    assert test.info()['percentage_duplicated'] >= 0.1

    # Allowing a percentage of 0.75 duplicated samples the test fails
    test = SampleLeakingTest(base_dataset=synthetic_train, test_dataset=synthetic_test, ignore_feats=None, threshold=0.75)
    test.run()
    assert test.info()['percentage_duplicated'] <= 0.75
    # Check for hash implementation
    test = SampleLeakingTest(base_dataset=synthetic_train, test_dataset=synthetic_test, ignore_feats=None, threshold=0.75, use_hash=True)
    test.run()
    assert test.info()['percentage_duplicated'] <= 0.75


def test_sample_leaking_test_ignore_feats_param():

    synthetic_train = pd.DataFrame(data=[
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2],
        [1.3, 1.4, 1.5],
        [1.6, 1.7, 1.8],
    ], columns=["f1", "f2", "f3"])

    # Doesn't contain repeated thanks to f3. If f3 removed all are repeated with synthetic_train
    synthetic_test_1 = pd.DataFrame(data=[
        [0.1, 0.2, 1.3],
        [0.4, 0.5, 1.6],
        [0.7, 0.8, 1.9],
        [1.0, 1.1, 2.2]
    ], columns=["f1", "f2", "f3"])

    # Without ignoring any feature the test passes
    test = SampleLeakingTest(base_dataset=synthetic_train, test_dataset=synthetic_test_1, ignore_feats=None, threshold=0)
    test.run()

    # Ignoring column 3 raises error
    test = SampleLeakingTest(base_dataset=synthetic_train, test_dataset=synthetic_test_1, ignore_feats=['f3'], threshold=0)
    with pytest.raises(FailedTestError):
        test.run()

def test_sample_leaking_test_strings():

    synthetic_train = pd.DataFrame(data=[
        [0.1, "val_1", "value_1"],
        [0.4, "val_2", "value_1"],
        [0.7, "val_3", "value_1"],
        [1.0, "val_4", "value_1"]
    ], columns=["f1", "f2", "f3"]
    )

    # Doesn't contain repeated thanks to f3
    synthetic_test = pd.DataFrame(data=[
        [0.1, "val_1", "value_2"],
        [0.4, "val_2", "value_2"],
        [0.7, "val_3", "value_2"],
        [1.0, "val_4", "value_2"]
    ], columns=["f1", "f2", "f3"]
    )

    # Using all columns the test passes
    test = SampleLeakingTest(base_dataset=synthetic_train, test_dataset=synthetic_test, ignore_feats=None)
    test.run()
    # Check for hash implementation
    test = SampleLeakingTest(base_dataset=synthetic_train, test_dataset=synthetic_test, ignore_feats=None, use_hash=True)
    test.run()

    # Ignoring column f3 raises error
    test = SampleLeakingTest(base_dataset=synthetic_train, test_dataset=synthetic_test, ignore_feats=['f3'])
    with pytest.raises(FailedTestError):
        test.run()
    # Check for hash implementation
    test = SampleLeakingTest(base_dataset=synthetic_train, test_dataset=synthetic_test, ignore_feats=['f3'], use_hash=True)
    with pytest.raises(FailedTestError):
        test.run()

def test_sample_leaking_test_contains_duplicated_only_in_train():

    synthetic_train = pd.DataFrame(data=[
        [0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3],
        [0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2],
        [1.3, 1.4, 1.5],
        [1.6, 1.7, 1.8],
    ], columns=["f1", "f2", "f3"])

    synthetic_test = pd.DataFrame(data=[
        [10.1, 10.2, 10.3],
        [10.4, 10.5, 10.6],
        [13.7, 13.8, 13.9],
        [14.0, 14.1, 14.2]
    ], columns=["f1", "f2", "f3"])

    # test should pass
    test = SampleLeakingTest(base_dataset=synthetic_train, test_dataset=synthetic_test, ignore_feats=None, threshold=0)
    test.run()
    # Check for hash implementation
    test = SampleLeakingTest(base_dataset=synthetic_train, test_dataset=synthetic_test, ignore_feats=None, threshold=0, use_hash=True)
    test.run()

def test_sample_leaking_test_contains_duplicated_only_in_test():

    synthetic_train = pd.DataFrame(data=[
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2],
        [1.3, 1.4, 1.5],
        [1.6, 1.7, 1.8],
    ], columns=["f1", "f2", "f3"])

    synthetic_test = pd.DataFrame(data=[
        [10.1, 10.2, 10.3],
        [10.1, 10.2, 10.3],
        [3.7, 3.8, 3.9],
        [4.0, 4.1, 4.2]
    ], columns=["f1", "f2", "f3"])

    # test should pass
    test = SampleLeakingTest(base_dataset=synthetic_train, test_dataset=synthetic_test, ignore_feats=None, threshold=0)
    test.run()
    # Check for hash implementation
    test = SampleLeakingTest(base_dataset=synthetic_train, test_dataset=synthetic_test, ignore_feats=None, threshold=0, use_hash=True)
    test.run()

def test_sample_leaking_test_different_columns_in_datasets():
    synthetic_test = pd.DataFrame(data=[
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2],
        [1.3, 1.4, 1.5],
        [1.6, 1.7, 1.8],
    ], columns=["f1", "f2", "f4"])


    # Contains 2 repeated
    synthetic_train = pd.DataFrame(data=[
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [3.7, 3.8, 3.9],
        [4.0, 4.1, 4.2]
    ], columns=["f1", "f2", "f3"])

    # Allowing 1 duplicate the test fails
    test = SampleLeakingTest(base_dataset=synthetic_train, test_dataset=synthetic_test, ignore_feats=None,
                            threshold=1)
    with pytest.raises(ValueError):
        test.run()

def test_dataset_with_nans():
    synthetic_train = pd.DataFrame(data=[
        [0.1, 0.2, 0.3],
        [0.4, 0.5, np.nan],
        [0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2],
        [1.3, 1.4, 1.5],
        [1.6, 1.7, 1.8],
    ], columns=["f1", "f2", "f3"])


    # Contains 2 repeated
    synthetic_test = pd.DataFrame(data=[
        [0.1, 0.2, 0.3],
        [0.4, 0.5, np.nan],
        [3.7, 3.8, 3.9],
        [4.0, 4.1, 4.2]
    ], columns=["f1", "f2", "f3"])

    # Allowing 1 duplicate the test fails
    test = SampleLeakingTest(base_dataset=synthetic_train, test_dataset=synthetic_test, ignore_feats=None,
                            threshold=1)
    with pytest.raises(FailedTestError):
        test.run()
    assert test.info()['num_duplicated'] == 2

    # Allowing 3 duplicates the test passes
    test = SampleLeakingTest(base_dataset=synthetic_train, test_dataset=synthetic_test, ignore_feats=None,
                            threshold=3)
    test.run()
    assert test.info()['num_duplicated'] == 2

def test_no_duplicated_test():
    # Assert crashes with duplicated rows
    test_df = pd.DataFrame([
        [1, 2, 3],
        [2, 2, 4],
        [4, 4, 4],
        [4, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
    ])

    test = NoDuplicatesTest(test_df)

    with pytest.raises(FailedTestError):
        test.run()

    # Assert all OK with no duplicates
    test_df = pd.DataFrame([
        [1, 2, 3],
        [2, 2, 4],
        [4, 4, 4],
        [4, 2, 3],
    ])

    test = NoDuplicatesTest(test_df)
    test.run()  # This should run with no problems

    # Force hash sample comparison
    test_df = pd.DataFrame([
        [1, 2, 3],
        [2, 2, 4],
        [4, 4, 4],
        [4, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
    ])
    test = NoDuplicatesTest(test_df, use_hash=True)
    with pytest.raises(FailedTestError):
        test.run()

    # Check it works ignoring some features
    test = NoDuplicatesTest(test_df, ignore_feats=[2])
    with pytest.raises(FailedTestError):
        test.run()

def test_lin_combinations_cont():

    df = pd.DataFrame()
    df["f1"] = np.random.uniform(size=100)
    df["f2"] = df["f1"] * 2
    df["f3"] = np.random.uniform(size=100)
    df["f4"] = df["f1"] + df["f2"]
    df["f5"] = np.random.uniform(size=100)

    df["f6"] = np.random.uniform(size=100)
    df["f7"] = df["f6"] * 3
    df["f8"] = df["f6"] * 0.1

    schma_reference = DataSchema().generate(df).calculate_statistics()
    linear_comb_test = LinearCombinationsTest(df, dataset_schema=schma_reference)
    with pytest.raises(FailedTestError, match="'f1', 'f2'"):
        linear_comb_test.run()
