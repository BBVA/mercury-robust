import pathlib, pytest
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from mercury.dataschema import DataSchema

from mercury.robust.model_tests import (
    ModelReproducibilityTest,
    ModelSimplicityChecker,
    DriftPredictionsResistanceTest,
    ClassificationInvarianceTest,
    TreeCoverageTest,
    DriftMetricResistanceTest,
    FeatureCheckerTest
)
from mercury.robust.errors import FailedTestError


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


# TEST MODEL REPRODUCIBILITY TEST

@pytest.fixture(scope='module')
def classification_dataset():
    # Create Dataset for the model reproducibility test
    X,y = make_classification(n_samples=2000, n_informative=5, flip_y=0.5, random_state=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    df_train = pd.DataFrame(X_train, columns=["x" + str(i) for i in range(X_train.shape[1])])
    df_train["y"] = y_train

    df_test = pd.DataFrame(X_test, columns=["x" + str(i) for i in range(X_test.shape[1])])
    df_test["y"] = y_test

    return df_train, df_test

class CustomModel:

    def __init__(self):
        self.m1 = RandomForestClassifier(max_depth=4, random_state=4)
        self.m2 = RandomForestClassifier(random_state=3)
        self.m3 = RandomForestClassifier(max_depth=3, random_state=2)

    def fit(self, X,y):
        self.m1 = self.m1.fit(X,y)
        self.m2 = self.m2.fit(X,y)
        self.m3 = self.m3.fit(X,y)

    def predict(self, X):
        pred_m1 = self.m1.predict(X).reshape(-1,1)
        pred_m2 = self.m2.predict(X).reshape(-1,1)
        pred_m3 = self.m3.predict(X).reshape(-1,1)
        predictions = np.concatenate([pred_m1, pred_m2, pred_m3], axis=1)
        return np.median(predictions, axis=1)

    def __deepcopy__(self, memo):
        m = CustomModel()
        m.m1 = sklearn.base.clone(self.m1)
        m.m2 = sklearn.base.clone(self.m2)
        m.m3 = sklearn.base.clone(self.m3)
        return m

class NotValidModel:

    def __init__(self):
        pass

    def fit(self, X,y):
        pass

    def predict(self, X):
        return np.ones((X.shape[0]))


def test_model_reproducibility_sklearn(classification_dataset):

    df_train, df_test = classification_dataset

    model_repro = RandomForestClassifier(max_depth=5, random_state=2000)
    model_no_repro = RandomForestClassifier(max_depth=5, random_state=None)

    def train_model(model, X, y, train_params=None):
        model.fit(X,y)
        return model

    def eval_model(model, X, y, eval_params=None):
        y_pred = model.predict(X)
        return accuracy_score(y, y_pred)

    def get_predictions(model, X, predict_params=None):
        return model.predict(X)

    # Test Reproducible sklearn Model - Should pass
    test = ModelReproducibilityTest(
        model = model_repro,
        train_dataset = df_train,
        target="y",
        train_fn = train_model,
        eval_fn = eval_model,
        train_params = None,
        eval_params = None,
        predict_fn = get_predictions,
        threshold_eval = 0,
        threshold_yhat = 0,
        test_dataset = df_test
    )
    test.run()

    # Test non-reproducible sklearn Model - Should fail(Diff Metric)
    test = ModelReproducibilityTest(
        model = model_no_repro,
        train_dataset = df_train,
        target="y",
        train_fn = train_model,
        eval_fn = eval_model,
        train_params = None,
        eval_params = None,
        predict_fn = get_predictions,
        threhsold_eval = 0,
        threshold_yhat = 0,
        test_dataset = df_test
    )
    with pytest.raises(FailedTestError) as exinfo:
        test.run()

    # Test non-reproducible sklearn model - Should fail (Diff predictions)
    test = ModelReproducibilityTest(
        model = model_no_repro,
        train_dataset = df_train,
        target="y",
        train_fn = train_model,
        eval_fn = None,
        train_params = None,
        eval_params = None,
        predict_fn = get_predictions,
        threshold_eval = 0,
        threshold_yhat = 0,
        test_dataset = df_test
    )
    with pytest.raises(FailedTestError) as exinfo:
        test.run()

    # If we don't specify train function, an exception must be raised
    test = ModelReproducibilityTest(
        model = model_repro,
        train_dataset = df_train,
        target="y",
        train_fn = None,
        eval_fn = eval_model,
        train_params = None,
        eval_params = None,
        predict_fn = get_predictions,
        threshold_eval = 0,
        threshold_yhat = 0,
        test_dataset = df_test
    )
    with pytest.raises(ValueError) as exinfo:
        test.run()

    # If we don't specify the eval function neither the predict function, an expection must be raised
    test = ModelReproducibilityTest(
        model = model_repro,
        train_dataset = df_train,
        target="y",
        train_fn = train_model,
        eval_fn = None,
        train_params = None,
        eval_params = None,
        predict_fn = None,
        threshold_eval = 0,
        threshold_yhat = 0,
        test_dataset = df_test
    )
    with pytest.raises(ValueError) as exinfo:
        test.run()

    # Only check the eval metric - This Test should pass
    test = ModelReproducibilityTest(
        model = model_repro,
        train_dataset = df_train,
        target="y",
        train_fn = train_model,
        eval_fn = eval_model,
        train_params = None,
        eval_params = None,
        predict_fn = None,
        threshold_eval = 0,
        threshold_yhat = 0,
        test_dataset = df_test
    )
    test.run()

    # Only Check a a training Set - This Test should pass
    test = ModelReproducibilityTest(
        model = model_repro,
        train_dataset = df_train,
        target="y",
        train_fn = train_model,
        eval_fn = eval_model,
        train_params = None,
        eval_params = None,
        predict_fn = get_predictions,
        threshold_eval = 0,
        threshold_yhat = 0
    )
    test.run()

    # Setting a high threshold_eval can make a non-exactly reproducible model pass the test
    test = ModelReproducibilityTest(
        model = model_no_repro,
        train_dataset = df_train,
        target="y",
        train_fn = train_model,
        eval_fn = eval_model,
        train_params = None,
        eval_params = None,
        predict_fn = None,
        threshold_eval = 100,
        threshold_yhat = 0,
        test_dataset = df_test
    )
    test.run()

    # Setting a high threshold_yhat can make a non-exactly reproducible model pass the test
    test = ModelReproducibilityTest(
        model = model_no_repro,
        train_dataset = df_train,
        target="y",
        train_fn = train_model,
        eval_fn = None,
        train_params = None,
        eval_params = None,
        predict_fn = get_predictions,
        threshold_eval = 0,
        threshold_yhat = 100,
        test_dataset = df_test
    )
    test.run()

    # Similarly, we can set yhat_allowed_diff to allow a non-exactly reproducible model pass
    test = ModelReproducibilityTest(
        model = model_no_repro,
        train_dataset = df_train,
        target="y",
        train_fn = train_model,
        eval_fn = None,
        train_params = None,
        eval_params = None,
        predict_fn = get_predictions,
        threshold_eval = 0,
        threshold_yhat = 0,
        test_dataset = df_test,
        yhat_allowed_diff = 100
    )
    test.run()

def test_model_reproducibility_tfkeras(classification_dataset):

    df_train, df_test = classification_dataset

    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras import Sequential
    model = Sequential()
    model.add(Input(shape=(df_train.shape[1]-1,)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    def train_model(model, X, y, train_params):
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        hist = model.fit(X, y, batch_size=train_params["batch_size"], epochs=train_params["epochs"])
        return model

    def eval_model(model, X, y, eval_params=None):
        y_pred = np.argmax(model.predict(X), axis=1)
        return accuracy_score(y, y_pred)

    def get_predictions(model, X):
        return np.argmax(model.predict(X), axis=1)

    # Model is not reproducible - this test should fail
    test = ModelReproducibilityTest(
        model = model,
        train_dataset = df_train,
        target="y",
        train_fn = train_model,
        eval_fn = eval_model,
        train_params = {"batch_size": 8, "epochs": 2},
        eval_params = None,
        predict_fn = get_predictions,
        threshold_eval = 0,
        threshold_yhat = 0,
        test_dataset = df_test
    )
    with pytest.raises(FailedTestError) as exinfo:
        test.run()


def test_model_reproducibility_custom_model(classification_dataset):

    df_train, df_test = classification_dataset

    # TEST CUSTOM MODEL
    m = CustomModel()
    def train_model(model, X, y, train_params=None):
        model.fit(X,y)
        return model

    def eval_model(model, X, y, eval_params=None):
        y_pred = model.predict(X)
        return accuracy_score(y, y_pred)

    def get_predictions(model, X, predict_params=None):
        return model.predict(X)

    # The model is reproducible - The Test should Pass
    test = ModelReproducibilityTest(
        model = m,
        train_dataset = df_train,
        target="y",
        train_fn = train_model,
        eval_fn = eval_model,
        train_params = None,
        eval_params = None,
        predict_fn = get_predictions,
        threshold_eval = 0,
        threshold_yhat = 0,
        test_dataset = df_test
    )
    test.run()


    # TEST CUSTOM MODEL NOT COMPATIBLE WITH REPRODUCIBLITY TEST
    m = NotValidModel()

    def train_model(model, X, y, hyperparams=None):
        model.fit(X,y)
        return model

    def eval_model(model, X, y, eval_params=None):
        y_pred = model.predict(X)
        return accuracy_score(y, y_pred)

    def get_predictions(model, X):
        return model.predict(X)

    # The model is not compatible with the test - An exception must be raised
    test = ModelReproducibilityTest(
        model = m,
        train_dataset = df_train,
        target="y",
        train_fn = train_model,
        eval_fn = eval_model,
        train_params = None,
        eval_params = None,
        predict_fn = get_predictions,
        threshold_eval = 0,
        threshold_yhat = 0,
        test_dataset = df_test
    )
    with pytest.raises(ValueError) as exinfo:
        test.run()


# MODEL SIMPLICITY CHECKER

def _create_classification_pipeline(num_feats=None, cat_feats=None):

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_feats),
            ("cat", categorical_transformer, cat_feats),
        ], remainder='drop'
    )

    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor),
               ("classifier", RandomForestClassifier(n_estimators=200, max_depth=100, random_state=2000))]
    )

    return pipeline

def _create_regression_pipeline(num_feats=None, cat_feats=None):

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_feats),
            ("cat", categorical_transformer, cat_feats),
        ], remainder='drop'
    )

    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor),
               ("classifier", RandomForestRegressor(n_estimators=200, max_depth=100, random_state=2000))]
    )

    return pipeline

def test_model_simplicity_checker_sk_classification(datasets):

    tips, titanic = datasets
    titanic2 = titanic.copy()
    titanic2 = titanic2.drop(['deck', 'alive'], axis=1).dropna()

    target = 'survived'
    num_feats = ['age', 'fare']
    cat_feats = ['pclass', 'sex', 'sibsp', 'parch', 'embarked', 'class', 'who', 'adult_male', 'embark_town', 'alone']

    df_train, df_test = train_test_split(titanic2, test_size=0.3, random_state=2000)
    X_train = df_train.loc[:, [f for f in df_train.columns if f != target]]
    y_train = df_train.loc[:, target]
    X_test = df_test.loc[:, [f for f in df_test.columns if f != target]]
    y_test = df_test.loc[:, target]

    # create and train Model
    model = _create_classification_pipeline(num_feats, cat_feats)
    model = model.fit(X_train, y_train)

    # With LogReg as baseline, the test fails
    test = ModelSimplicityChecker(
        model = model,
        X_train = X_train,
        y_train = y_train,
        X_test = X_test,
        y_test = y_test,
        threshold=0.01,
        baseline_model=LogisticRegression(solver='liblinear')
    )
    assert test.info() is None
    with pytest.raises(FailedTestError) as exinfo:
        test.run()
    assert (test.metric_model - 0.01) <= test.metric_baseline_model
    assert 'metric_model' in test.info()

    # With DecTree as baseline, the test passes
    test = ModelSimplicityChecker(
        model = model,
        X_train = X_train,
        y_train = y_train,
        X_test = X_test,
        y_test = y_test,
        threshold=0.01,
        baseline_model=DecisionTreeClassifier(random_state=2000),
        task='classification'  # task not inferred in this case
    )
    test.run()
    assert (test.metric_model - 0.01) > test.metric_baseline_model

    # Run with ignore_feats
    X_train["random_feature"] = np.random.uniform(size=len(X_train))
    X_test["random_feature"] = np.random.uniform(size=len(X_test))
    test = ModelSimplicityChecker(
        model = model,
        X_train = X_train,
        y_train = y_train,
        X_test = X_test,
        y_test = y_test,
        ignore_feats=["random_feature"],
        threshold=0.01,
        baseline_model=DecisionTreeClassifier(random_state=2000),
        task='classification'  # task not inferred in this case
    )
    test.run()

def test_model_simplicity_checker_sk_regression(datasets):

    tips, titanic = datasets

    target = 'tip'
    num_feats = ['total_bill']
    cat_feats = ['sex', 'smoker', 'day', 'time', 'size']

    df_train, df_test = train_test_split(tips, test_size=0.3, random_state=2000)
    X_train = df_train.loc[:, [f for f in df_train.columns if f != target]]
    y_train = df_train.loc[:, target]
    X_test = df_test.loc[:, [f for f in df_test.columns if f != target]]
    y_test = df_test.loc[:, target]

    # create and train the model
    model = _create_regression_pipeline(num_feats, cat_feats)
    model = model.fit(X_train, y_train)

    # test with default baseline
    test = ModelSimplicityChecker(
        model = model,
        X_train = X_train,
        y_train = y_train,
        X_test = X_test,
        y_test = y_test,
        threshold=0.02
    )
    with pytest.raises(FailedTestError) as exinfo:
        test.run()
    assert (test.metric_model - 0.01) <= test.metric_baseline_model

def test_model_simplicity_checker_tfkeras():

    # create dataset
    X,y = make_classification(n_samples=2000, n_informative=5, flip_y=0.5, random_state=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    # Create tf.keras model and train
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.layers import Dense, Input

    model = tf.keras.Sequential([
        Input(shape=(X_train.shape[1],)),
        layers.Dense(30, input_dim=8, activation='relu'),
        layers.Dense(20, activation='relu'),
        layers.Dense(2, activation='softmax',  name='output')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=5)

    # Test using custom predict_fn
    def custom_predict(model, X):
        y_hat = np.argmax(model.predict(X), axis=1)
        return y_hat

    test = ModelSimplicityChecker(
        model = model,
        X_train = X_train,
        y_train = y_train,
        X_test = X_test,
        y_test = y_test,
        threshold=0.1,  # high threhsold to ensure randomness doesn't make different outcome in executions
        predict_fn=custom_predict
    )
    with pytest.raises(FailedTestError) as exinfo:
        test.run()
    assert (test.metric_model - 0.1) <= test.metric_baseline_model

    # Test with already given predictions and eval_fn
    # Calculate predictions
    y_hat_test = custom_predict(model, X_test)
    test = ModelSimplicityChecker(
        model = model,
        X_train = X_train,
        y_train = y_train,
        X_test = X_test,
        y_test = y_test,
        threshold=0.1,  # high threhsold to ensure randomness doesn't make different outcome in executions
        eval_fn=roc_auc_score,
        test_predictions=y_hat_test,
        encode_cat_feats=False,
        scale_num_feats=False
    )
    with pytest.raises(FailedTestError) as exinfo:
        test.run()
    assert (test.metric_model - 0.1) <= test.metric_baseline_model

def test_model_simplicity_checker_errors(datasets):

    tips, titanic = datasets

    target = 'tip'
    num_feats = ['total_bill']
    cat_feats = ['sex', 'smoker', 'day', 'time', 'size']

    df_train, df_test = train_test_split(tips, test_size=0.3, random_state=2000)
    X_train = df_train.loc[:, [f for f in df_train.columns if f != target]]
    y_train = df_train.loc[:, target]
    X_test = df_test.loc[:, [f for f in df_test.columns if f != target]]
    y_test = df_test.loc[:, target]

    # create and train the model
    model = _create_regression_pipeline(num_feats, cat_feats)
    model = model.fit(X_train, y_train)

    # Wrong task raises exception
    test = ModelSimplicityChecker(
        model = model,
        X_train = X_train,
        y_train = y_train,
        X_test = X_test,
        y_test = y_test,
        threshold=0.02,
        task='image_classification'
    )
    with pytest.raises(ValueError) as exinfo:
        test.run()

    # Wrong target type (datetime)
    X = np.array([1,2,3])
    y = np.array([np.datetime64('2005-02-25'), np.datetime64('2005-02-25'), np.datetime64('2005-02-25')])
    test = ModelSimplicityChecker(
        model = None,
        X_train = X,
        y_train = y,
        X_test = X,
        y_test = y,
        threshold=0.1,  # high threhsold to ensure randomness doesn't make different outcome in executions
    )
    with pytest.raises(ValueError) as exinfo:
        test.run()

def test_model_simplicity_multiclass_classification():

    # Generate multiclass classification dataset
    X, y = make_classification(n_samples=200, n_features=10, n_informative=5, n_classes=3)
    y_ohe = np.zeros((y.size, y.max()+1))
    y_ohe[np.arange(y.size),y] = 1

    # Test 1: Labels encoded as labels
    m1 = RandomForestClassifier()
    m1.fit(X, y)
    test = ModelSimplicityChecker(
        model = m1,
        X_train = X.copy(),
        y_train = y.copy(),
        X_test = X.copy(),
        y_test = y.copy(),
        threshold=0.01,
    )
    test.run()
    assert (test.metric_model - 0.01) > test.metric_baseline_model

    # Test 2: Lbaels encoded as ohe
    m2 = RandomForestClassifier()
    m2.fit(X, y_ohe)
    test = ModelSimplicityChecker(
        model = m2,
        X_train = X.copy(),
        y_train = y_ohe.copy(),
        X_test = X.copy(),
        y_test = y_ohe.copy(),
        baseline_model =  DecisionTreeClassifier(max_depth=1),
        threshold=0.01,
    )
    test.run()
    assert (test.metric_model - 0.01) > test.metric_baseline_model


def test_drift_prediction_resistance_test():
    df = pd.read_csv("mercury/robust/tutorials/data/credit/UCI_Credit_card.csv")

    df = df.loc[:, df.columns != 'ID']

    col_keys = list(df.columns)
    col_vals = [x.lower() if x != 'default.payment.next.month' else 'Y' for x in col_keys]

    df = df.rename(columns = dict(zip(col_keys, col_vals)))

    X = df.loc[:, df.columns != 'Y'].copy()
    Y = df.loc[:, 'Y']

    rf = RandomForestClassifier()

    model = rf.fit(X, Y)

    drift_args = {'cols' : ['pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6'], 'iqr' : [1, 1.1]}
    test = DriftPredictionsResistanceTest(model = model, X = X, drift_type = 'scale_drift', drift_args = drift_args, tolerance = X.shape[0]*0.05)
    test.run()
    assert test.info()["loss"] < test.tolerance

    drift_args = {'force' : 200, 'noise' : 100}
    test = DriftPredictionsResistanceTest(model			 = model,
                               X				 = X,
                               drift_type		 = 'shift_drift',
                               drift_args		 = drift_args,
                               names_categorical = ['sex', 'education', 'marriage'],
                               tolerance		 = X.shape[0]*0.05)

    with pytest.raises(FailedTestError):
        test.run()

    drift_args = {'cols' : ['pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6'], 'iqr' : [1, 1.1]}

    with pytest.raises(RuntimeError):
        test = DriftPredictionsResistanceTest(model = model, X = X, drift_type = 'not_a_method', drift_args = drift_args, tolerance = X.shape[0]*0.05)

    def eval_fun(Y_obs, Y_hat):
        return sum((Y_obs - Y_hat)**2)

    test = DriftPredictionsResistanceTest(model	  = model,
                               X		  = X,
                               drift_type = 'scale_drift',
                               drift_args = drift_args,
                               eval		  = eval_fun,
                               tolerance  = X.shape[0]*0.05)

    test.run()

    schema = DataSchema().generate_manual(dataframe		   = X,
                                          categ_columns    = ['sex', 'education', 'marriage'],
                                          discrete_columns = [],
                                          binary_columns   = [])

    test = DriftPredictionsResistanceTest(model          = model,
                               X              = X,
                               drift_type     = 'scale_drift',
                               drift_args     = drift_args,
                               dataset_schema = schema,
                               tolerance      = X.shape[0]*0.05)

    test.run()

def test_drift_prediction_resistance_test_hard_labels_multiclass():
    X, y = make_classification(n_samples=300, n_features=3, n_informative=3, n_redundant=0, n_classes=3)
    X = pd.DataFrame(X, columns=["f"+str(i) for i in range(X.shape[1])])

    model = RandomForestClassifier()
    model = model.fit(X, y)

    drift_args = {'cols' : ['f0', 'f1', 'f2'], 'iqr' : [1, 1.1]}
    test = DriftPredictionsResistanceTest(model = model, X = X, drift_type = 'scale_drift', drift_args = drift_args, tolerance = X.shape[0]*0.5)
    test.run()
    assert test.info()["loss"] < test.tolerance

def test_drift_resistance_test():
    X, y = make_classification(n_samples=300, n_features=3, n_informative=3, n_redundant=0, n_classes=3)
    X = pd.DataFrame(X, columns=["f"+str(i) for i in range(X.shape[1])])

    model = RandomForestClassifier()
    model = model.fit(X, y)

    drift_args = {'cols' : ['f0', 'f1', 'f2'], 'iqr' : [1, 1.1]}
    test = DriftPredictionsResistanceTest(model = model, X = X, drift_type = 'scale_drift', drift_args = drift_args, tolerance = X.shape[0]*0.5)
    test.run()
    assert test.info()["loss"] < test.tolerance


CLASS_NEGATIVE = 0
CLASS_NEUTRAL = 1
CLASS_POSITIVE = 2
class SimpleModel():

    def __init__(self):
        pass

    def predict(self, sentences):
        """predicts the class with the mayority of words"""
        preds = []
        for sentence in sentences:
            if sentence.count("pos_word") > sentence.count("neg_word"):
                preds.append(CLASS_POSITIVE)
            elif sentence.count("neg_word") > sentence.count("pos_word"):
                preds.append(CLASS_NEGATIVE)
            else:
                preds.append(CLASS_NEUTRAL)
        return preds

def test_invariance_test_basic():

    # Create simple sample
    original = [
        "neutral pos_word",
        "neutral neg_word",
        "neutral neutral"
    ]

    perturbed = [
        ["neutral pos_word", "neutral pos_word"],
        ["neutral neutral", "neutral pos_word"],
        ["neutral pos_word", "irrelevant neutral"]
    ]

    model = SimpleModel()

    # Low threshold raises error
    test = ClassificationInvarianceTest(
        original_samples=original,
        perturbed_samples=perturbed,
        model=model,
        threshold=0.,
        check_total_errors_rate=True,
        name="test name"
    )
    assert test.info() is None

    with pytest.raises(FailedTestError) as exinfo:
        test.run()
    assert_array_equal(test.pred_is_different[0] , np.array([False, False]))
    assert_array_equal(test.pred_is_different[1] , np.array([True, True]))
    assert_array_equal(test.pred_is_different[2] , np.array([True, False]))
    assert test.num_failed_per_sample == [0,2,1]
    assert test.num_perturbed_per_sample == [2,2,2]
    assert_array_equal(test.samples_with_errors, np.array([False , True, True]))
    assert round(test.rate_samples_with_errors,2) == 0.67
    assert round(test.info()["rate_samples_with_errors"],2) == 0.67
    assert len(test.get_examples_failed(n_samples = 2, n_perturbed = 1)) == 2

    # High threhsold doesn't raise error (Let's pass the predict_fn this time)
    test = ClassificationInvarianceTest(
        original_samples=original,
        perturbed_samples=perturbed,
        predict_fn=model.predict,
        threshold=0.9,
        check_total_errors_rate=False,
        name="test name"
    )
    test.run()
    assert_array_equal(test.pred_is_different[0] , np.array([False, False]))
    assert_array_equal(test.pred_is_different[1] , np.array([True, True]))
    assert_array_equal(test.pred_is_different[2] , np.array([True, False]))
    assert test.num_failed_per_sample == [0,2,1]
    assert test.num_perturbed_per_sample == [2,2,2]
    assert_array_equal(test.samples_with_errors, np.array([False , True, True]))
    assert round(test.rate_samples_with_errors,2) == 0.67


def test_invariance_test_bad_arguments():
    original = ["sample1", "sample2", "sample3"]
    perturbed = ["sample1_pertubed"] # Only contains 1 of the samples
    test = ClassificationInvarianceTest(
        original_samples=original,
        perturbed_samples=perturbed,
        predict_fn=lambda x: 1,
        threshold=0.,
        check_total_errors_rate=True,
        name="test name"
    )

    with pytest.raises(ValueError) as exinfo:
        test.run()

    # Neither predict function nor model are specified
    original = ["sample1"]
    perturbed = ["sample1_pertubed"] # Only contains 1 of the samples
    test = ClassificationInvarianceTest(
        original_samples=original,
        perturbed_samples=perturbed,
        threshold=0.,
        check_total_errors_rate=True,
        name="test name"
    )

    with pytest.raises(ValueError) as exinfo:
        test.run()

def test_tree_coverage_ok():
    x, y = make_classification(200)

    # Test ensemble sklearn
    rf = RandomForestClassifier().fit(x, y)
    test = TreeCoverageTest(rf, x)
    test.run()

    # Test decision tree
    t = DecisionTreeClassifier().fit(x, y)
    test = TreeCoverageTest(t, x)
    test.run()
    # nothing should have failed 'till this point

def test_tree_coverage_fail():
    x, y = make_regression(200)
    rf = DecisionTreeRegressor().fit(x, y)
    test = TreeCoverageTest(rf, np.zeros_like(x))

    with pytest.raises(FailedTestError):
        test.run()

    t = DecisionTreeRegressor().fit(x, y)
    test = TreeCoverageTest(t, np.zeros_like(x))

    with pytest.raises(FailedTestError):
        test.run()

def test_tree_coverage_fail():
    x, y = make_regression(200)
    rf = DecisionTreeRegressor().fit(x, y)
    test = TreeCoverageTest(rf, np.zeros_like(x))

    with pytest.raises(FailedTestError):
        test.run()

    t = DecisionTreeRegressor().fit(x, y)
    test = TreeCoverageTest(t, np.zeros_like(x))

    with pytest.raises(FailedTestError):
        test.run()

def test_tree_coverage_pipeline():
    x, y = make_classification(200)

    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier())
    ])
    pipeline = pipeline.fit(x, y)
    test = TreeCoverageTest(pipeline, x)
    test.run()

def test_tree_coverage_pipeline_without_tree_fails():
    x, y = make_classification(200)

    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression())
    ])
    pipeline = pipeline.fit(x, y)
    with pytest.raises(ValueError):
        test = TreeCoverageTest(pipeline, x)
        test.run()



def test_drift_metric_resistance_test_regression():

    X, y = make_regression(n_samples=200, n_features=3, n_informative=3)
    X = pd.DataFrame(X, columns=["f"+str(i) for i in range(X.shape[1])])

    model_lr = LinearRegression().fit(X, y)
    model_rf = RandomForestRegressor().fit(X, y)

    drift_args = {
        'cols' : ['f0'],
        'method' : 'value',
        'method_params': {'value': 99999}
    }

    # Linear Regression is highly affected by extreme values
    test_lr = DriftMetricResistanceTest(
        model = model_lr,
        X = X,
        Y = y,
        drift_type = "outliers_drift",
        drift_args = drift_args,
        tolerance= 500,
    )
    with pytest.raises(FailedTestError):
        test_lr.run()

    # Random Forest is less afected
    test_rf = DriftMetricResistanceTest(
        model = model_rf,
        X = X,
        drift_type = "outliers_drift",
        drift_args = drift_args,
        tolerance= 500,
        Y = y
    )
    test_rf.run()

    assert test_lr.info()['metric_diff'] > test_rf.info()['metric_diff']

def test_drift_metric_resistance_test_classification():

    X, y = make_classification(n_samples=300, n_features=3, n_informative=3, n_redundant=0)
    X = pd.DataFrame(X, columns=["f"+str(i) for i in range(X.shape[1])])

    model = LogisticRegression().fit(X, y)

    drift_args_1 = {
        'cols' : ['f0', 'f1', 'f2'],
        'method' : 'value',
        'method_params': {'value': 1e9, 'proportion_outliers': 0.99}
    }

    test_1 = DriftMetricResistanceTest(
        model = model,
        X = X,
        drift_type = "outliers_drift",
        drift_args = drift_args_1,
        tolerance= 1.0,
        Y = y
    )
    test_1.run()
    assert test_1.info()['metric_diff'] <= 1.

    # Test custom eval metric
    def custom_eval_metric(y_true, y_hat):
        return 5.

    test = DriftMetricResistanceTest(
        model = model,
        X = X,
        drift_type = "outliers_drift",
        drift_args = drift_args_1,
        tolerance= 10.0,
        Y = y,
        eval=custom_eval_metric
    )
    test.run()
    assert test.info()['metric_diff'] == 0.

def test_drift_metric_resistance_test_invalid_inputs():

    X, y = make_classification(n_samples=300, n_features=3, n_informative=3, n_redundant=0)
    X = pd.DataFrame(X, columns=["f"+str(i) for i in range(X.shape[1])])

    model = LogisticRegression().fit(X, y)

    drift_args_1 = {
        'cols' : ['f0', 'f1', 'f2'],
        'method' : 'value',
        'method_params': {'value': 1e9, 'proportion_outliers': 0.99}
    }

    # invalid task
    test = DriftMetricResistanceTest(
        model = model,
        X = X,
        drift_type = "outliers_drift",
        drift_args = drift_args_1,
        tolerance= 1.0,
        Y = y,
        task='invalid'
    )
    with pytest.raises(ValueError):
        test.run()

def test_feature_checker_test():
    df = pd.read_csv('%s/csv/test_twt.csv' % pathlib.Path(__file__).parent.resolve(), sep = '\t')

    train, test = train_test_split(df, test_size = 0.2)

    X_train = train.loc[:, train.columns != 'Y']
    Y_train = train.loc[:, 'Y']

    rf = RandomForestClassifier()
    rf.fit(X_train, Y_train)

    fct = FeatureCheckerTest(rf, train, 'Y', tolerance = 100)
    with pytest.raises(FailedTestError):
        fct.run()

    fct = FeatureCheckerTest(rf, train, 'Y', test, tolerance = 100)
    with pytest.raises(FailedTestError):
        fct.run()

    def eval_fun(Y_obs, Y_hat):
        return sum((Y_obs - Y_hat)**2)

    fct = FeatureCheckerTest(rf, train, 'Y', test, eval = eval_fun, tolerance = 100)
    with pytest.raises(FailedTestError):
        fct.run()

    fct = FeatureCheckerTest(rf, train, 'Y', test, importance = 'feature_importances_', tolerance = 100)
    with pytest.raises(FailedTestError):
        fct.run()

    fct = FeatureCheckerTest(rf, train, 'Y', test, importance = 'feature_importances_', eval = eval_fun, tolerance = 100, remove_num = 2, num_tries = 1)
    with pytest.raises(FailedTestError):
        fct.run()


def test_feature_checker_test_model_function():

    # Create Dataset
    X, y = make_classification(200, n_features=5, n_informative=5, n_redundant=0)
    df = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4", "x5"])
    df["x6"] = np.random.choice(["cat_0", "cat_1", "cat_2"], size=len(df)) # make categorical feature feature
    df["y"] = y
    df_train, df_test = train_test_split(df, test_size = 0.2)

    # Define function to create the model (pipeline)
    def create_pipeline_fn(X, model_args):

        num_feats = [f for f in model_args['num_feats'] if f in X.columns]
        cat_feats = [f for f in model_args['cat_feats'] if f in X.columns]

        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        categorical_transformer = Pipeline(steps=[("ohe", OneHotEncoder(handle_unknown="ignore"))])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_feats),
                ("cat", categorical_transformer, cat_feats),
            ], remainder='drop'
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", DecisionTreeClassifier())
            ]
        )

        return pipeline

    # Create and run test
    num_feats = ["x1", "x2", "x3", "x4", "x5"]
    cat_feats = ["x6"]
    model_fn_args = {
        'num_feats': num_feats,
        'cat_feats': cat_feats
    }
    test = FeatureCheckerTest(
        model=create_pipeline_fn,
        train=df_train,
        target="y",
        test=df_test,
        tolerance=-1,
        num_tries=3,
        remove_num=1,
        model_fn_args = model_fn_args,
    )
    test.run()