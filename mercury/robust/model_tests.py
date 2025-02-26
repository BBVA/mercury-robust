import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import inspect
import copy
from typing import Union, Callable, List, Dict
from mercury.robust.basetests import RobustModelTest, TaskInferrer
from mercury.robust.errors import FailedTestError

from mercury.dataschema.feature import FeatType
from mercury.dataschema import DataSchema
from mercury.monitoring.drift.drift_simulation import BatchDriftGenerator
from ._tree_analysis import SkLearnTreeCoverageAnalyzer, SkLearnTreeEnsembleCoverageAnalyzer



class ModelReproducibilityTest(RobustModelTest):
    """
    This test checks if the training of a model is reproducible. It does so by training the model two times
    and checking whether they give the same evaluation metric and predictions. If the difference in
    the evaluation metric is higher than `threshold_eval` parameter then the test fails. Similarly, if
    the percentage of different predictions is higher than `threshold_yhat` then the test fails.
    You can check only one of the checks (the evaluation metric and predictions) or only one of them.

    Args:
        model (BaseEstimator, tf.keras.Model): Unfitted model that we are checking reproducibility
        train_dataset (pd.DataFrame): The pandas dataset used for training the model.
        target (str): The name of the target variable predicted by the model which must be one column in
            the train dataset.
        train_fn (Callable): function called to train the model. The interface of the function is
            train_fn(model, X, y, train_params) and returns the fitted model.
        train_params (dict): Params to use for training. It is passed as a parameter to the `train_fn`
        eval_fn (Callable): function called to evaluate the model. The interface of the function is
            eval_fn(model, X, y, eval_params) and returns a float. If None, then the check of looking if
            training two times produces the same evaluation metric won't be performed.
        eval_params (dict): Params to use for evaluation. It is passed as a parameter to the `eval_fn`.
        threshold_eval (float): difference that we are able to tolerate in the evaluation function in order
            to pass the test. If the difference of the evaluation metric when training the model two times
            is higher than the threshold, then the test fails. Default value is 0
        predict_fn (Callable): function called to get the predictions of a dataset once the model is trained.
            The interface of the function is predict_fn(model, X, predict_params) and returns the predictions.
        predict_params (dict): Params to use for prediction. It is passed as a parameter to the `predict_fn`
        threshold_yhat (float): If `predict_fn` is given, this is the percentage of different predictions that we are
            can tolerate without making the test fail. A prediction from the model trained two times is considered
            different according to the parameter `yhat_allowed_diff`. Default value for `threshold_yhat` is 0,
            meaning that with just one prediction different the test will fail.
        yhat_allowed_diff (float): difference that we can tolerate in order to consider that a sample has the
            same prediction from two models. If a prediction of the model trained two times differ by more than
            `yhat_allowed_diff` then that prediction is considered to be different. Default value is 0.
        test_dataset (pd.DataFrame):  If given, a separate dataset with identical column structure used for the
            evaluation parts.
        name (str): A name for the test. If not used, it will take the name of the class.

    Example:
        ```python
        >>> # Define necessary functions for model training, evaluation, and getting predictions
        >>> def train_model(model, X, y, train_params=None):
        >>>     model.fit(X, y)
        >>>     return model
        >>> def eval_model(model, X, y, eval_params=None):
        >>>     y_pred = model.predict(X)
        >>>     return accuracy_score(y, y_pred)
        >>> def get_predictions(model, X, pred_params=None):
        >>>     return model.predict(X)
        >>> # Create and run test
        >>> from mercury.robust.model_tests import ModelReproducibilityTest
        >>> test = ModelReproducibilityTest(
        >>>     model = model,
        >>>     train_dataset = df_train,
        >>>     target = "label_col",
        >>>     train_fn = train_model_fn,
        >>>     eval_fn = eval_model_fn,
        >>>     threshold_eval = 0,
        >>>     predict_fn = get_predictions_fn,
        >>>     epsilon_yhat = 0,
        >>>     test_dataset = df_test
        >>> )
        >>> test.run()
        ```
    """

    def __init__(
        self, model: Union["BaseEstimator", "tf.keras.Model"],  # noqa: F821
        train_dataset: "pd.DataFrame",  # noqa: F821,
        target: str,
        train_fn: Callable, train_params: dict = None,
        eval_fn: Callable = None, eval_params: dict = None, threshold_eval: float = 0.,
        predict_fn: Callable = None, predict_params: dict = None,
        threshold_yhat: float = 0., yhat_allowed_diff: float = 0.,
        test_dataset: "pd.DataFrame" = None,  # noqa: F821,
        name: str = None,
        *args, **kwargs
    ):
        super().__init__(model, name, *args, **kwargs)
        self.train_dataset = train_dataset
        self.train_fn = train_fn
        self.train_params = train_params
        self.target = target
        self.eval_fn = eval_fn
        self.eval_params = eval_params
        self.threshold_eval = threshold_eval
        self.predict_fn = predict_fn
        self.predict_params = predict_params
        self.threshold_yhat = threshold_yhat
        self.yhat_allowed_diff = yhat_allowed_diff
        self.test_dataset = test_dataset

    def _check_diff_in_eval(self, model_1, model_2, X, y):
        self.eval_1 = self.eval_fn(model_1, X, y, self.eval_params)
        self.eval_2 = self.eval_fn(model_2, X, y, self.eval_params)
        if abs(self.eval_1 - self.eval_2) > self.threshold_eval:
            return True
        return False

    def _check_diff_in_predictions(self, model_1, model_2, X):
        yhat_1 = self.predict_fn(model_1, X, self.predict_params)
        yhat_2 = self.predict_fn(model_2, X, self.predict_params)
        self.diff = (np.abs(yhat_1 - yhat_2) > self.yhat_allowed_diff).mean()
        if self.diff > self.threshold_yhat:
            return True
        return False

    def _clone_unfitted_model(self, model):
        """
        Returns a new copy of the model
        """
        if isinstance(model, sklearn.base.BaseEstimator):
            return sklearn.base.clone(model)
        else:
            # import tensorflow only if we need it
            import tensorflow as tf
            if isinstance(model, (tf.keras.Model, tf.estimator.Estimator)):
                return tf.keras.models.clone_model(model)
            else:
                # Check if we have deepcopy method
                members, _ = zip(*inspect.getmembers(model))
                if '__deepcopy__' in members:
                    return copy.deepcopy(model)
                # Otherwise, we cannot copy the model
                raise ValueError("The type of model used is not compatible with the test")

    def run(self, *args, **kwargs):
        """
        Runs the test.

        Raises FailedTestError if training is not reproducible
        """

        super().run(*args, **kwargs)

        X_train = self.train_dataset.loc[:, self.train_dataset.columns != self.target]
        y_train = self.train_dataset.loc[:, self.target]
        if self.test_dataset is not None:
            X_test = self.test_dataset.loc[:, self.test_dataset.columns != self.target]
            y_test = self.test_dataset.loc[:, self.target]

        # Clone the model
        model_1 = self._clone_unfitted_model(self.model)
        model_2 = self._clone_unfitted_model(self.model)

        # Train the models
        if self.train_fn is None:
            raise ValueError("You must provide a valid train_fn to train the model")
        model_1 = self.train_fn(model_1, X_train, y_train, self.train_params)
        model_2 = self.train_fn(model_2, X_train, y_train, self.train_params)

        # Check that at least one of eval_fn or predict_fn are specified
        if self.eval_fn is None and self.predict_fn is None:
            raise ValueError("At least one of eval_fn or predict_fn must be specified")

        if self.eval_fn is not None:
            # Check evaluation on train
            if self._check_diff_in_eval(model_1, model_2, X_train, y_train):
                raise FailedTestError(
                    f"Eval metric different in train dataset when training two times ({round(self.eval_1,4)} vs {round(self.eval_2,4)}). "
                    f"The max difference allowed is {round(self.threshold_eval, 4)} "
                    f"The model is not reproducible."
                )
            # Check evaluation on test
            if (self.test_dataset is not None) and self._check_diff_in_eval(model_1, model_2, X_test, y_test):
                raise FailedTestError(
                    f"Eval metric different in test dataset when training two times ({round(self.eval_1,4)} vs {round(self.eval_2,4)}). "
                    f"The max difference allowed is {round(self.threshold_eval, 4)} "
                    f"The model is not reproducible."
                )

        if self.predict_fn is not None:
            # Check evaluation on train
            if self._check_diff_in_predictions(model_1, model_2, X_train):
                raise FailedTestError(
                    f"Percentage of different predictions in training set is {round(self.diff,4)} when training two times and the "
                    f"maximum allowed is {round(self.threshold_yhat,4)} "
                    f"The model is not reproducible."
                )
            # Check evaluation on test
            if (self.test_dataset is not None) and self._check_diff_in_predictions(model_1, model_2, X_test):
                raise FailedTestError(
                    f"Percentage of different predictions in test set is {round(self.diff,4)} when training two times and the "
                    f"maximum allowed is {round(self.threshold_yhat,4)} "
                    f"The model is not reproducible."
                )


class ModelSimplicityChecker(RobustModelTest, TaskInferrer):

    """
    This test looks if a trained model has a simple baseline which trained in the same dataset
    gives better or similar performance on a test dataset. If not specified, the baseline is considered
    a LogisticRegression model for classification tasks and LinearRegression model for regression tasks.

    Args:
        model (Union["BaseEstimator", "tf.keras.Model"]): the trained model which we will compare against
            a baseline model
        X_train (Union["pandas.DataFrame", np.ndarray]): features of train dataset used to train the `model`.
            This same dataset will be used to train the baseline.
        y_train (Union["pandas.DataFrame", np.ndarray]): targets of train dataset used to train the `model`.
            This same dataset will be used to train the baseline.
        X_test (Union["pandas.DataFrame", np.ndarray]): features of test dataset which will be used to evaluate
            the `model` and the baseline.
        y_test (Union["pandas.DataFrame", np.ndarray]): targets of test dataset which will be used to evaluate
            the `model` and the baseline.
        ignore_feats: Features which won't be used in the `baseline_model`. Only use when `X_train` and `X_test`
            are pandas dataframes.
        baseline_model ("BaseEstimator"): Optional model that will be used as a baseline. It doesn't have to be
            an sklearn model, however, it needs to implement the `fit()` and `predict()` methods. If not specified,
            a LogisticRegression is used in case of classification and LinearRegression in case of regression.
        task (str): Task of the dataset. It must be either 'classification' or 'regression'. If None provided
              then it will be auto inferred from the `target` column.
        eval_fn (Callable[[np.ndarray], np.ndarray]): function which returns a metric to compare the performance of
            the `model` against the baseline. The interface of the function is eval_fn(y_true, y_pred). Note that in
            this test is assumed that the higher the metric the better the model, therefore if you use a metric which
            lower means better then the `eval_fn` should return the negative of the metric. If not specified, the
            accuracy score will be used in case of classification and R2 in case of regression.
        predict_fn (Callable[["BaseEstimator"], np.ndarray]): Custom predict function to obtain predictions from `model`.
            The interface of the function is predict_fn(model, X_test). Note that by default this is None, and in that case,
            the test will try to obtain the predictions using the `predict()` method of the `model`. That works in many cases
            where you are using scikit-learn models and the predict function returns what the `eval_fn` is expecting.
            However, in some case you might need to define a custom `predict_fn`. For example, when using a tf.keras
            model and the accuracy_score as eval_fn, the `predict()` method from tf.keras model returns the
            probabilities and the accuracy_score expects the classes, therefore you can use the `predict_fn` to obtain
            the classes. Another alternative is to pass the already computed predictions using the `test_predictions`
            parameter.
        test_predictions (np.array): array of predictions of the test set obtained by the `model`. If given, the
            test will use them instead of computing them using the `predict()` method of the `model` or the `predict_fn`
            This might be useful in cases where you are creating multiple tests and you want to avoid to compute
            the predictions each time.
        threshold (float): The threshold to use when comparing the `model` and the baseline. It is used to establish
            the limit to consider that the baseline performs similar or better than the `model`. Concretely, if the
            baseline model performs worse than the `model` with a difference equal or higher than this threshold
            then the test passes. Otherwise, if the baseline model performs better or performs worse but with a
            difference lower than this threshold, then the test fails.
        name (str): A name for the test. If not used, it will take the name of the class.
        encode_cat_feats (bool): bool to indicate whether to encode categorical features as
            one-hot-encoding. Note that if you specify it as False and your dataset has string column then the test will
            raise an exception since the default baseline models won't be able to deal with string columns. Default
            value is True
        scale_num_feats (bool): bool to indicate whether to scale the numeric features. If True, a StandardScaler
            is used. Default value is True.
        schema_custom_feature_map: Internally, this test generates a DataSchema object. In case you find it makes
                                   wrong feature type assignations to your features you can pass here a dictionary which
                                   specify the feature type of the columns you want to fix. (See DataSchema. `force_types`
                                   parameter for more info on this).
        dataset_schema: Pre built schema. This argument is complementary to `schema_custom_feature_map`. In case you want to
                        manually build your own DataSchema object, you can pass it here and the test will internally use it
                        instead of the default, automatically built. If you provide this parameter, `schema_custom_feature_map`
                        will not be used.

    Example:
        ```python
        >>> from mercury.robust.model_tests import ModelSimplicityChecker
        >>> test = ModelSimplicityChecker(
        >>>     model = model,
        >>>     X_train = X_train,
        >>>     y_train = y_train,
        >>>     X_test = X_test,
        >>>     y_test = y_test,
        >>>     threshold = 0.02,
        >>>     eval_fn = roc_auc_score
        >>> )
        >>> test.run()
        ```
    """

    def __init__(
        self, model: Union["BaseEstimator", "tf.keras.Model"],  # noqa: F821
        X_train: Union["pandas.DataFrame", np.ndarray],  # noqa: F821
        y_train: Union["pandas.Series", np.ndarray],  # noqa: F821
        X_test: Union["pandas.DataFrame", np.ndarray],  # noqa: F821
        y_test: Union["pandas.Series", np.ndarray],  # noqa: F821
        baseline_model: "BaseEstimator" = None,  # noqa: F821
        ignore_feats: List[str] = None,
        task: str = None,
        eval_fn: Callable[[np.ndarray], np.ndarray] = None,
        predict_fn: Callable[["BaseEstimator"], np.ndarray] = None,  # noqa: F821
        threshold: float = None,
        name: str = None,
        encode_cat_feats: bool = True,
        scale_num_feats: bool = True,
        test_predictions: np.array = None,
        schema_custom_feature_map: Dict[str, FeatType] = None,
        dataset_schema: DataSchema = None,
        *args, **kwargs
    ):

        super().__init__(model, name, *args, **kwargs)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.ignore_feats = ignore_feats if ignore_feats is not None else []

        self.task = task
        self.baseline_model = baseline_model

        self.threshold = threshold
        self.eval_fn = eval_fn
        self.predict_fn = predict_fn
        self.test_predictions = test_predictions

        self.encode_cat_feats = encode_cat_feats
        self.scale_num_feats = scale_num_feats

        self._schema_custom_feature_map = schema_custom_feature_map
        self._dataset_schema = dataset_schema

        self.metric_model = None
        self.metric_baseline_model = None

    def _get_pipeline_with_preprocessing(
        self,
        num_feats: List[str],
        cat_feats: List[str],
        model: "BaseEstimator"  # noqa: F821
    ):

        l_transformers = []
        if (self.scale_num_feats) and (len(num_feats) > 0):
            l_transformers.append(("num", StandardScaler(), num_feats))
        if (self.encode_cat_feats) and (len(cat_feats) > 0):
            l_transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats))

        if len(l_transformers) > 0:
            preprocessor = ColumnTransformer(transformers=l_transformers, remainder='drop')
            return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
        else:
            return Pipeline(steps=[("classifier", model)])

    def _get_baseline_model(self, schema):
        """
        Obtain eval_fn and model_baseline, depending on the task
        """
        if self.task == 'classification':
            baseline_model = LogisticRegression() if self.baseline_model is None else self.baseline_model
        else:
            baseline_model = LinearRegression() if self.baseline_model is None else self.baseline_model

        # Create pipeline with feature preprocessing
        cat_feats = schema.categorical_feats + schema.binary_feats
        cat_feats = [f for f in cat_feats if f not in self.ignore_feats]
        num_feats = schema.continuous_feats + schema.discrete_feats
        num_feats = [f for f in num_feats if f not in self.ignore_feats]

        baseline_model = self._get_pipeline_with_preprocessing(num_feats, cat_feats, baseline_model)

        return baseline_model

    def info(self):
        if self.metric_model is not None:
            return {"metric_model": self.metric_model, "metric_baseline_model": self.metric_baseline_model}
        else:
            return None

    def run(self, *args, **kwargs):

        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_test = self.y_test

        # Obtain Task if not specified
        if not self.task:
            self.task = self.infer_task(y_train)

        if self.task not in ('classification', 'regression'):
            raise ValueError("Error. 'task' must be either classification or regression")

        # Generate Schema (if Xs are not pandas, transform them first)
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)

        if self._dataset_schema is None:
            schema_train = DataSchema().generate(X_train, force_types=self._schema_custom_feature_map)
        else:
            schema_train = self._dataset_schema

        # Obtain model_baseline, depending on the task
        baseline_model = self._get_baseline_model(schema_train)
        baseline_name = str(baseline_model[-1])

        # Obtain eval function, depending on the task
        if self.task == 'classification':
            eval_fn = accuracy_score if self.eval_fn is None else self.eval_fn
        else:
            eval_fn = r2_score if self.eval_fn is None else self.eval_fn

        # Train the baseline model
        baseline_model = baseline_model.fit(X_train, y_train)

        # Evaluate on test set
        if self.test_predictions is not None:
            self.metric_model = eval_fn(y_test, self.test_predictions)
        elif self.predict_fn is not None:
            self.metric_model = eval_fn(y_test, self.predict_fn(self.model, X_test))
        else:
            self.metric_model = eval_fn(y_test, self.model.predict(X_test))
        self.metric_baseline_model = eval_fn(y_test, baseline_model.predict(X_test))

        # Compare difference
        if (self.metric_model - self.threshold) < self.metric_baseline_model:
            raise FailedTestError(
                f"""a {baseline_name} baseline model gives an evaluation metric of {round(self.metric_baseline_model, 4)} while """
                f"""the original model gives an evaluation metric of {round(self.metric_model, 4)}. The difference is lower than """
                f"""the threshold {round(self.threshold, 4)}"""
            )


class DriftPredictionsResistanceTest(RobustModelTest):
    """
    This test checks the robustness of a trained model to drift in the X dataset. It uses the model to predict the Y from the given X and
    uses that `Y` as a ground truth. Then, it applies some drift to the data in X by using a `BatchDriftGenerator` object and does a new
    prediction `drifted_Y` using the drifted dataset. If both the `Y` and `drifted_Y` diverge by more that some given tolerance value,
    the test fails. This test does only one verification. If we need doing more than one drift check, just apply multiple tests with
    appropriate names to simplify following up the results.

    Args:
        model (Any): The model being evaluated. The model must be already trained and will not be trained again by this test. It is assumed
        	to have a sklearn-like compliant predict() method that works on the dataset and returns a vector that is accepted by the
            evaluation function.
        X (pd.DataFrame): A pandas dataset that can be used by the model's predict() method and whose predicted values will be used as the
        	ground truth drift measurement.
        drift_type (str): The name of the method of a BatchDriftGenerator specifying the type of drift to be applied. E.g., "shift_drift",
            "scale_drift", ... You can check the class BatchDriftGenerator in _drift_simulation to see all available types
        drift_args (dict): A dictionary with the argument expected by the drift method. E.g., {cols: ['a', 'b'], iqr: [1.12, 1.18]} for
            "scale_drift".
        names_categorical (List[str], optional): An optional list with the names of the categorical variables. If this is used, the
        	internal `BatchDriftGenerator` will use a `DataSchema` object to fully define the variables in X as either categorical (if
            in the list) or continuous (otherwise). This allows automatically selecting the columns without using the `cols` argument
            in `drift_args`. If this parameter is not given, the `DataSchema` is not initially defined and either you select the columns
            manually by declaring a `cols` argument in `drift_args` or the `BatchDriftGenerator` will create a `DataSchema` that
            automatically infers the column types.
        dataset_schema (DataSchema, optional): Alternatively, you can provide a pre built schema for an even higher level of control. If
        	you use this argument, `names_categorical` is not used. The schema fully defines binary, categorical, discrete or continuous.
            If you still define the `cols` argument in `drift_args`, that selection will prevail over whatever is in `dataset_schema`.
        eval (Callable, optional): If given, an evaluation function that defines how "different" the predictions are. The function must
        	accept two vectors returned by model.predict() and return some positive value that indicates the difference in the predictions
            and is compared with `tolerance`. If not given, then a sum of squared differences will be used unless the `model.predict()`
            method generates the hard labels for a multi-class classification problem. In this last case, the `eval` function will be a
            function to compute the number of different predictions.
        tolerance (float): A real value to be compared with the result of the evaluation function. Note that the purpose of the test is to
        	check if the model is robust to the introduced drift. Therefore, the test will fail when the result (named as `loss`) is
            higher than the tolerance, meaning the model predictions considerably change with the introduced drift. When the test fails,
            you can see the value returned by the `eval` function in the RuntimeError message displayed as `loss`.
        name (str, optional): A name for the test. If not used, it will take the name of the class.

    Example:
        ```python
        >>> from mercury.robust.model_tests import DriftPredictionsResistanceTest
        >>> test = DriftPredictionsResistanceTest(
        >>>     model = trained_model,
        >>>     X = X,
        >>>     drift_type = "shift_drift",
        >>>     drift_args = {'cols': ['feature_1'], 'force': 100.},
        >>>     tolerance = 5,
        >>> )
        >>> test.run()
        ```
    """

    def __init__(
        self,
        model,
        X,
        drift_type,
        drift_args,
        names_categorical=None,
        dataset_schema=None,
        eval=None,
        tolerance=1e-3,
        name=None,
    ):
        super().__init__(model, name)

        self.model = model
        self.X = X.copy()
        self.Y = model.predict(X)

        if dataset_schema is not None:
            self.gen = BatchDriftGenerator(X=self.X, schema=dataset_schema)
        elif names_categorical is not None:
            self.gen = BatchDriftGenerator(
                X=self.X,
                schema=DataSchema().generate_manual(
                    dataframe=self.X,
                    categ_columns=names_categorical,
                    discrete_columns=[],
                    binary_columns=[],
                ),
            )
        else:
            self.gen = BatchDriftGenerator(X=self.X)

        self.fun = getattr(self.gen, drift_type, None)

        if not callable(self.fun):
            raise RuntimeError(
                "drift_type = %s must be a method of BatchDriftGenerator."
            )

        self.drift_args = drift_args
        self.eval = eval
        self.tolerance = tolerance
        self.loss = None

    def run(self, *args, **kwargs):
        """
        Runs the test.

        Raises `FailedTestError` with a descriptive message if any of the attempts fail.
        """

        super().run(*args, **kwargs)

        drifted_Y = self._calculate_drifted_Y()

        if self.eval is None:
            eval_fn = self._get_default_eval_fn()
            self.loss = eval_fn(self.Y, drifted_Y)
        else:
            self.loss = self.eval(self.Y, drifted_Y)

        if self.loss > self.tolerance:
            raise FailedTestError(
                "Test failed. Prediction loss drifted above tolerance = %.3f (loss = %.3f)"
                % (self.tolerance, self.loss)
            )

    def info(self) -> dict:
        return {'loss': self.loss}

    def _calculate_drifted_Y(self):
        drifted_gen = self.fun(**self.drift_args)
        drifted_Y = self.model.predict(drifted_gen.data)
        return drifted_Y

    def _get_default_eval_fn(self):

        if self._contains_hard_labels(self.Y):
            return lambda y, drifted_y: np.sum(y != drifted_y)
        else:
            return lambda y, drifted_y: np.sum((y - drifted_y) ** 2)

    def _contains_hard_labels(self, y):
        return np.all(np.mod(y, 1)) == 0 and np.any(y > 1)


class DriftMetricResistanceTest(DriftPredictionsResistanceTest, TaskInferrer):
    """
    This test checks the robustness of a trained model to drift in the X dataset. It uses the model to predict the Y_no_drifted
    from the given X. Then, it applies some drift to the data in X by using a `BatchDriftGenerator` object and calculates
    Y_drifted. Then calculates a metric using Y_true with Y_no_drifted on the one hand and using Y_true with Y_drifted on the other
    hand. If the difference between these two metrics diverge more than some given tolerance value, the test fails.
    This test does only one verification. If we need doing more than one drift check, just apply multiple tests with
    appropriate names to simplify following up the results.

    Args:
        model (Any): The model being evaluated. The model must be already trained and will not be trained again by this test. It is
        	assumed to have a sklearn-like compliant predict() method that works on the dataset and returns a vector that is accepted by
            the evaluation function.
        X (pd.DataFrame): A pandas dataset that can be used by the model's predict() method and whose predicted values will be used as
        	the ground truth drift measurement.
        Y (Union["pandas.DataFrame", np.ndarray]): array with the ground truth values. It will be used to calculate the metric for the
        	non-drifted dataset and for the drifted dataset
        drift_type (str): The name of the method of a BatchDriftGenerator specifying the type of drift to be applied. E.g., "shift_drift",
            "scale_drift", ... You can check the class BatchDriftGenerator in _drift_simulation to see all available types
        drift_args (dict): A dictionary with the argument expected by the drift method. E.g., {cols: ['a', 'b'], iqr: [1.12, 1.18]} for
            "scale_drift".
        names_categorical (List[str], optional): An optional list with the names of the categorical variables. If this is used, the
        	internal `BatchDriftGenerator` will use a `DataSchema` object to fully define the variables in X as either categorical
            (if in the list) or continuous (otherwise). This allows automatically selecting the columns without using the `cols` argument
            in `drift_args`. If this parameter is not given, the `DataSchema` is not initially defined and either you select the columns
            manually by declaring a `cols` argument in `drift_args` or the `BatchDriftGenerator` will create a `DataSchema` that
            automatically infers the column types.
        dataset_schema (DataSchema, optional): Alternatively, you can provide a pre built schema for an even higher level of control. If
        	you use this argument, `names_categorical` is not used. The schema fully defines binary, categorical, discrete or continuous.
            If you still define the `cols` argument in `drift_args`, that selection will prevail over whatever is in `dataset_schema`.
        eval (Callable, optional):: the evaluation function to use to calculate the metric. If passed, the interface of the function must
        	be `eval_fn(y_true, y_hat)`. If not used, the mean absolute error will be used for regression and the accuracy for
            classification.
        tolerance (float): A real value to be compared with the difference of the computed metric with the non-drifted dataset and with the
            drifted dataset.
        task (str): 'classification' or 'regression'. If not given, the test will try to infer it from `Y`
        name (str): A name for the test. If not used, it will take the name of the class.

    Example:
        ```python
        >>> testing_dataset = pd.DataFrame(...)
        >>> rf = RandomForestClassifier().fit(train_data)
        >>> drift_args = {'cols': 'feature_1', 'method': 'percentile', 'method_params': {'percentile': 95}}
        >>> test = DriftMetricResistanceTest(
        ...    model = rf,
        ...    X = testing_dataset[features],
        ...    Y = testing_dataset[target],
        ...    drift_type = 'outliers_drift',
        ...    drift_args = drift_args,
        ...    tolerance = 0.05
        ... )
        >>> test.run()  # The test will fail if the difference in the metric is more than 0.05
        ```
    """

    def __init__(
        self,
        model,
        X: pd.DataFrame,
        Y: np.array,
        drift_type,
        drift_args,
        names_categorical=None,
        dataset_schema=None,
        eval=None,
        tolerance=None,
        task=None,
        name=None
    ):
        super().__init__(model, X, drift_type, drift_args, names_categorical, dataset_schema, eval, tolerance, name)
        self.Y_true = Y
        self.task = task
        self.metric_no_drifted = None
        self.metric_drifted = None
        self.metric_diff = None

    def run(self, *args, **kwargs):

        drifted_Y = self._calculate_drifted_Y()
        self.metric_diff = self._evaluate_performance_degradation(drifted_Y)
        if self.metric_diff > self.tolerance:
            raise FailedTestError(
                "Test failed. The metric of drifted dataset has changed above tolerance = %.3f (diff = %.3f)"
                % (self.tolerance, self.metric_diff)
            )

    def info(self) -> dict:
        return {
            'metric_no_drifted': self.metric_no_drifted,
            'metric_drifted': self.metric_drifted,
            'metric_diff': self.metric_diff,
        }

    def _evaluate_performance_degradation(self, drifted_Y):

        if self.eval is None:
            eval_fn = self._get_default_eval_fn()
        else:
            eval_fn = self.eval

        self.metric_no_drifted = eval_fn(self.Y_true, self.Y)
        self.metric_drifted = eval_fn(self.Y_true, drifted_Y)

        return np.abs(self.metric_no_drifted - self.metric_drifted)

    def _get_default_eval_fn(self):
        task = self.task if self.task is not None else self.infer_task(self.Y_true)
        if task == "classification":
            return accuracy_score
        elif task == "regression":
            return mean_absolute_error
        else:
            raise ValueError("Invalid task. It must be 'classification' or 'regression'")


class ClassificationInvarianceTest(RobustModelTest):
    """
    The idea of the `ClassificationInvarianceTest` is to check that if we apply a label-preserving perturbation the prediction of the
    model shouldn't change.

    This class helps to check this by checking the number of samples where the conditional of preserving the label doesn't hold and raising
    an error if the percentage of samples where the label is not preserved is higher than a specified threshold. We must pass to the test
    the original samples and the already generated perturbed samples.

    When calling run(), a exception `FailedTestError` is raised if the test fails. Additionally, the next attributes are filled:

        - preds_original_samples: stores the predictions for the original samples
        - preds_perturbed_samples: stores the predictions for perturbed samples
        - pred_is_different: stores for each sample a boolean array indicating if the predictions for the perturbed samples are different
            to the original sample
        - num_failed_per_sample: stores for each sample the number of perturbations where the prediction is different to the original sample
        - num_perturbed_per_sample: stores for each samples the number of perturbations
        - samples_with_errors: boolean array containing which samples contain errors
        - rate_samples_with_errors: the percentage of samples that contains at least one perturbed sample that the model predicted
            a different label.
        - total_rate_errors: the total percentage of perturbed samples that the model predicted a different label

    This test is based on the paper:
        ['Beyond Accuracy: Behavioral Testing of NLP Models with CheckList'](https://homes.cs.washington.edu/~marcotcr/acl20_checklist.pdf)

    Args:
        original_samples (Union(List[str], np.array)): List or array containing the original samples
        perturbed_samples (Union(List[List[str]], np.array)): List or array containing the perturbed samples. Each element
            of the list or each row of the array contains one or several perturbed samples corresponding to the sample in the
            same position/index in `original_samples`
        model: The model being evaluated. The model must be already trained. It is assumed to have a sklearn-like compliant
            predict() method that works on the `original_samples` and `perturbed_samples` and returns a vector with the
            the predictions. Alternatively, you can pass a `predict_fn`
        predict_fn (Callable): function that given the samples returns the predicted labels. Only used if `model` argument is
            None.
        threshold (float): if the percentage of samples with errors is higher than this threshold, then a `FailedTestError`
            will be raised. Default value is 0.05
        check_total_errors_rate (bool): this indicates what to consider as percentage of errors. If True, then each perturbed
            sample counts to calculate the rate. If False, then the rate is calculated with the number of samples independently
            of how many perturbations each sample has. Default value is True
        name (str): A name for the test. If not used, it will take the name of the class.

    Example:
        ```python
        >>> original_samples = ["sample1", "sample2"]
        >>> perturbed_samples = [
        ...    "perturbed_sample_1 for sample1", "perturbed_sample_2 for sample1",
        ...    "perturbed_sample_1 for sample2", "perturbed_sample2 for sample2"
        ... ]
        >>> test = ClassificationInvarianceTest(
        ...    original_samples,
        ...    perturbed_samples,
        ...    predict_fn=my_model.predict,
        ...    threshold=0.1,
        ...    check_total_errors_rate=True,
        ...    name="Invariance Test"
        ... )
        >>> test.run()
        ```
    """

    def __init__(
        self,
        original_samples: Union[List[str], np.array],
        perturbed_samples: Union[List[List[str]], np.array],
        model: "BaseEstimator" = None,  # noqa: F821
        predict_fn: Callable = None,
        threshold: float = 0.05,
        check_total_errors_rate: bool = True,
        name: str = None,
        *args, **kwargs
    ):
        super().__init__(model, name, *args, **kwargs)
        self.original_samples = original_samples
        self.perturbed_samples = perturbed_samples
        self.predict_fn = predict_fn
        self.threshold = threshold
        self.check_total_errors_rate = check_total_errors_rate
        self.num_samples = len(self.original_samples)

        # Attributes for results
        self.preds_original_samples = None
        self.preds_perturbed_samples = None
        self.pred_is_different = None
        self.num_failed_per_sample = None
        self.num_perturbed_per_sample = None
        self.samples_with_errors = None
        self.rate_samples_with_errors = None
        self.total_rate_errors = None

    def run(self, *args, **kwargs):
        """run the test"""

        self._check_args()
        self.preds_original_samples = self._get_predictions(self.original_samples)
        self._check_changes_in_perturbed_samples_predictions()
        self._calculate_errors_rate()
        self._fail_test_if_high_error()

    def info(self):
        if self.preds_original_samples is not None:
            return {
                "rate_samples_with_errors": self.rate_samples_with_errors,
                "total_rate_errors": self.total_rate_errors
            }
        else:
            return None

    def get_examples_failed(self, n_samples: int = 5, n_perturbed: int = 1):
        """
        Returns examples of samples that failed.

        Args:
            n_samples (int): number of samples to recover.
            n_perturbed (int): for each sample, how many failed perturbations to recover.
        """

        selected_failed_samples = self._select_random_failed_samples(n_samples)

        # Get perturbations that failed for each selected sample
        examples = []
        for idx_selected in selected_failed_samples:
            selected_perturbed = self._select_random_failed_perturbations(idx_selected, n_perturbed)
            for idx_perturbed in selected_perturbed:
                sample_original = self.original_samples[idx_selected]
                sample_perturbed = self.perturbed_samples[idx_selected][idx_perturbed]
                pred_original = self.preds_original_samples[idx_selected]
                pred_perturbed = self.preds_perturbed_samples[idx_selected][idx_perturbed]
                examples.append((sample_original, sample_perturbed, pred_original, pred_perturbed))

        return pd.DataFrame(examples, columns=["original", "perturbed", "pred_original", "pred_perturbed"])

    def _check_args(self):
        if len(self.original_samples) != len(self.perturbed_samples):
            raise ValueError("Arguments original_samples and perturbed_samples must have the same length")

        if self.model is None and self.predict_fn is None:
            raise ValueError("At least one of the arguments 'model' or 'predict_fn must be specified")

    def _get_predictions(self, samples):
        if self.model is not None:
            return self.model.predict(samples)
        else:
            return self.predict_fn(samples)

    def _check_changes_in_perturbed_samples_predictions(self):
        """For each one of the samples, check in which perturbed samples the prediction changes"""
        self.preds_perturbed_samples = []
        self.pred_is_different = []
        self.num_failed_per_sample = []
        self.num_perturbed_per_sample = []
        for i in range(self.num_samples):
            # Note if improvement in speed is need: flatten perturbed samples, and make the predictions in bigger batches
            pred_is_different = self._check_changes_in_predictions(self.preds_original_samples[i], self.perturbed_samples[i])
            self.pred_is_different.append(pred_is_different)
            self.num_failed_per_sample.append(np.sum(pred_is_different))
            self.num_perturbed_per_sample.append(len(self.perturbed_samples[i]))

    def _check_changes_in_predictions(self, original_pred, perturbed_samples):
        """For a particular sample, checks in which perturbed samples the predictions changes"""
        self.preds_perturbed_samples.append(self._get_predictions(perturbed_samples))
        pred_is_different = np.array(self.preds_perturbed_samples[-1]) != original_pred
        return pred_is_different

    def _calculate_errors_rate(self):
        self.samples_with_errors = np.array(self.num_failed_per_sample) > 0
        self.rate_samples_with_errors = self.samples_with_errors.sum() / len(self.original_samples)
        self.total_rate_errors = np.sum(self.num_failed_per_sample) / np.sum(self.num_perturbed_per_sample)

    def _fail_test_if_high_error(self):
        """raises a `FailedTestError` if the error is higher than the threshold"""
        if self.check_total_errors_rate:
            error_rate = self.total_rate_errors
        else:
            error_rate = self.rate_samples_with_errors
        if error_rate > self.threshold:
            raise FailedTestError(f"Error rate {error_rate} is higher than threshold {self.threshold}")

    def _select_random_failed_samples(self, n_samples):
        """Randomly, select samples that failed"""
        idx_failed_samples = np.where(np.array(self.num_failed_per_sample) > 0)[0]
        selected_failed_samples = np.random.choice(idx_failed_samples, size=min(n_samples, len(idx_failed_samples)), replace=False)
        return selected_failed_samples

    def _select_random_failed_perturbations(self, idx_sample, n_perturbed):
        """Randomly, select perturbations that failed of a specific sample"""
        preds_is_diff = np.where(self.pred_is_different[idx_sample])[0]
        selected_perturbed = np.random.choice(preds_is_diff, size=min(n_perturbed, len(preds_is_diff)), replace=False)
        return selected_perturbed


class TreeCoverageTest(RobustModelTest):
    """
    This test checks whether a given test_dataset covers a minimum percentage of all the branches of
    a tree. Use this in case you want to make sure no leaves are left unexplored when testing your model.
    In case the percentage of coverage is less than the required `threshold_coverage`, the test will fail.

    Right now, this test only supports scikit-learn tree models, including sklearn pipelines with one tree
    model in one of its steps.

    TODO: Add support for other frameworks such as lightgbm, catboost or xgboost.

    Args:
        model: Fitted tree-based model (or sklearn pipeline with a tree-based model) to inspect
        test_dataset: Dataset for testing the coverage.
        threshold_coverage: this threshold represents the minimum percentage that the `test_dataset` needs
            to cover in order to pass the test. Eg. if `threshold_coverage=0.7` then the `test_dataset`
            needs to cover at least 70% of the branches to pass the test.
        name (str): A name for the test. If not used, it will take the name of the class.

    Example:
        ```python
        >>> testing_dataset = pd.DataFrame(...)
        >>> rf = RandomForestClassifier().fit(train_data)
        >>> test = TreeCoverageTest(
        ...    rf,
        ...    testing_dataset,
        ...    threshold_coverage=.8
        ...    name="My Tree Coverage Test"
        ... )
        >>> test.run()  # The test will fail if the obtained coverage is less than 80%
        ```
    """

    def __init__(
        self, model: Union["DecisionTreeClassifier", "DecisionTreeRegressor",
                           "RandomForestClassifier", "RandomForestRegressor",
                           "sklearn.pipeline.Pipeline"],  # noqa: F821
        test_dataset: "pd.DataFrame",  # noqa: F821,
        threshold_coverage: float = .7,
        name: str = None,
        *args, **kwargs
    ):
        super().__init__(model, name, *args, **kwargs)
        self.test_dataset = test_dataset
        self.threshold_coverage = threshold_coverage
        if isinstance(model, sklearn.pipeline.Pipeline):
            index_model = self._identify_tree_in_pipeline(model)
            self.test_dataset = self._execute_pipeline_until_step(model, index_model, self.test_dataset)
            model = model.steps[index_model][1]
        self._analyzer = self._get_analyzer(model)
        self.coverage = None

    def _get_analyzer(self, model):
        """
        Builds an analyzer according to the model type.
        """
        if isinstance(model, (RandomForestRegressor, RandomForestClassifier)):
            return SkLearnTreeEnsembleCoverageAnalyzer(model)
        elif isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
            return SkLearnTreeCoverageAnalyzer(model)
        else:
            raise ValueError("Error. This model type is not supported by this test.")

    def _identify_tree_in_pipeline(self, pipeline):
        supported_tree_types = (
            RandomForestRegressor,
            RandomForestClassifier,
            DecisionTreeClassifier,
            DecisionTreeRegressor
        )
        index_model = -1
        for i, stage in enumerate(pipeline.steps):
            if isinstance(stage[1], supported_tree_types):
                index_model = i
        if index_model == -1:
            raise ValueError("A pipeline without any supported model types by this test has been provided")
        return index_model

    def _execute_pipeline_until_step(self, pipeline, index_step, data):
        return pipeline[:index_step].transform(data)

    def run(self, *args, **kwargs):
        """Run the test"""
        self._analyzer.analyze(self.test_dataset)
        coverage = self._analyzer.get_percent_coverage()
        self.coverage = coverage

        if coverage < self.threshold_coverage:
            raise FailedTestError(f"Achieved a coverage of {coverage} while the minimum required was {self.threshold_coverage}")

    def info(self) -> dict:
        return {'coverage': self.coverage}

class FeatureCheckerTest(RobustModelTest):
    """
    This model robustness test checks if training the models using less columns in the dataframe can achieve identical results.
    To do so, it uses the variable importance taken from the model itself or estimated using a mercury.explainability explainer
    (ShuffleImportanceExplainer). It does a small number of attempts at removing unimportant variables and "fails if it succeeds",
    since success implies that a smaller, therefore more efficient, dataset should be used instead. The purpose of this test is
    not to find that optimal dataset. That can be achieved by removing the columns identified as unimportant and iterating.

    **NOTE**: This class will retrain (fit) the model several times resulting in the model being altered as a side effect. Make
    copies of your model before using this tool. This tool is intended as a diagnostic tool.

    Args:
        model: The model being evaluated. The model is assumed to comply to a minimalistic sklearn-like interface. More precisely:
            1. It must have a fit() method that works on the dataset and the dataset with some columns removed. It is important that each
            time the method fit() is called the model is trained from scratch (ie does not perform incremental training).
            2. It must have a predict() method that works on the dataset and returns a vector that is accepted by the evaluation function.
            3. If the argument `importance` is used, it must have an attribute with that name containing a list of (value, column_name)
            tuples which is consistent with many sklearn models.
            Alternatively, you can provide a function that creates a model or pipeline. This is useful in models or pipelines where
            removing columns from a dataframe raises an error because it expects the removed column. In this case, the interface of
            the function is model_fn(dataframe, model_args), where dataframe is the input pandas dataframe with the features already
            removed and model_fn_args is a dictionary with optional parameters that can be passed to the function. Importantly,
            this function just needs to create the model (unfitted) instance, but not to perform the training
        model_fn_args: if you are using a function in `model` parameter, you can use this argument to provide arguments to that function.
        train: The pandas dataset used for training the model, possibly with some columns removed.
        target: The name of the target variable predicted by the model which must be one columns in the train dataset.
        test: If given, a separate dataset with identical column structure used for the evaluation parts. Otherwise, the train dataset
            will be used instead.
        importance: If given, the name of a property in the model is updated by a fit() call. It must contain the importance of the
            columns as a list of (value, column_name) tuples. Otherwise, the importance of the variables will be estimated using a
            mercury.explainer ShuffleImportanceExplainer.
        eval: If given, an evaluation function that defines what "identical" results are. The function must accept two vectors returned
            by model.predict() and return some positive value that is smaller than `tolerance` if "identical". Otherwise a sum of squared
            differences will be used instead.
        tolerance: A real value to be compared with the result of the evaluation function. Note that the purpose of the test is finding
            unimportant variables. Therefore, the test will fail when the result (named as `loss`) is smaller than the tolerance, meaning
            the model could work "identically" well with less variables. When the test fails, you can see the value returned by the `eval`
            function in the RuntimeError message displayed as `loss`.
        num_tries: The total number of column removal tries the test should do before passing. This value times `remove_num` must be
            smaller than the number of columns (Y excluded).
        remove_num: The number of columns removed at each try.
        name: A name for the test. If not used, it will take the name of the class.

    Example:
        ```python
        >>> from mercury.robust.model_tests import FeatureCheckerTest
        >>> test = FeatureCheckerTest(
        >>>     model=model,
        >>>     train=df_train,
        >>>     target="label_col",
        >>>     test=df_test,
        >>>     num_tries=len(df_train.columns)-1,
        >>>     remove_num=1,
        >>>     tolerance=len(df_test)*0.01
        >>> )
        >>> test.run()
        ```
    """

    def __init__(
        self,
        model: Union["Estimator", Callable],  # noqa: F821
        train: pd.DataFrame,
        target: str,
        test: pd.DataFrame = None,
        model_fn_args: dict = None,
        importance: str = None,
        eval: Callable = None,
        tolerance: float = 1e-3,
        num_tries: int = 3,
        remove_num: int = 1,
        name: str = None
    ):
        super().__init__(model, name)
        self.model_fn_args = model_fn_args
        self.train = train
        self.target = target
        self.test = test if test is not None else train
        self.importance = importance
        self.eval = eval
        self.remove_num = remove_num
        self.num_tries = num_tries
        self.tolerance = tolerance
        self.losses = {}

    def info(self):
        return self.losses

    def _get_least_important(self, N):

        names = []

        if self.importance is None:

            try:
                from mercury.explainability.explainers import ShuffleImportanceExplainer
            except:
                raise ImportError(
                    'Install mercury-explainability to use the FeatureCheckerTest with this type of model'
                )

            def model_inference_function(data, target):
                assert target == self.target

                X = data.loc[:, data.columns != self.target]
                Y = data.loc[:, self.target]
                Yh = self.fitted_model.predict(X)

                if self.eval is None:
                    return sum((Y - Yh) ** 2)

                return self.eval(Y, Yh)

            explainer = ShuffleImportanceExplainer(model_inference_function)

            explanation = explainer.explain(self.test, target=self.target)

            for name, _ in sorted(explanation.get_importances(), key=lambda x: x[1]):
                names.append(name)
                N -= 1
                if N == 0:
                    break

            return names

        try:
            importance = getattr(self.model, self.importance)

        except AttributeError:
            return names

        finally:
            columns = self.test.columns[self.test.columns != self.target]
            if len(columns) != len(importance):
                return names

            for _, name in sorted(zip(importance, columns)):
                names.append(name)
                N -= 1
                if N == 0:
                    break

            return names

    def run(self, *args, **kwargs):
        """
        Runs the test.

        Raises `FailedTestError` with a descriptive message if any of the attempts fail.
        """

        super().run(*args, **kwargs)

        N = self.remove_num * self.num_tries

        X_train = self.train.loc[:, self.train.columns != self.target]
        Y_train = self.train.loc[:, self.target]

        # Fit model with all the features
        self._fit_model(X_train, Y_train)

        X_test = self.test.loc[:, self.test.columns != self.target]
        Y_hat = self.fitted_model.predict(X_test)

        remove = self._get_least_important(N)

        if len(remove) != N:
            raise RuntimeError(
                "Wrong arguments. Not enough columns for remove_num*num_tries."
            )

        for t in range(self.num_tries):
            exclude = remove[0:self.remove_num]
            remove = remove[self.remove_num:]

            exclude.append(self.target)

            x_cols = [c for c in self.train.columns if c not in exclude]

            X_alt_train = self.train.loc[:, x_cols]

            # Fit model with the removed features
            self._fit_model(X_alt_train, Y_train)

            X_alt_test = self.test.loc[:, x_cols]
            Y_alt_hat = self.fitted_model.predict(X_alt_test)

            if self.eval is None:
                loss = sum((Y_hat - Y_alt_hat) ** 2)
            else:
                loss = self.eval(Y_hat, Y_alt_hat)
            self.losses[", ".join(exclude[:-1])] = loss

            if loss < self.tolerance:
                raise FailedTestError(
                    "Test failed. A model fitted removing columns [%s] is identical within tolerance %.3f (loss = %.3f)"
                    % (", ".join(exclude[:-1]), self.tolerance, loss)
                )

    def _fit_model(self, X, Y):
        if callable(self.model):
            self.fitted_model = self.model(X, self.model_fn_args)
            self.fitted_model.fit(X, Y)
        else:
            self.fitted_model = self.model.fit(X, Y)
