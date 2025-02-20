# mercury-robust

[![](https://github.com/BBVA/mercury-robust/actions/workflows/test.yml/badge.svg)](https://github.com/BBVA/mercury-robust)
![](https://img.shields.io/badge/latest-1.1.4-blue)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-3816/)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-3916/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31011/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3128/)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3131/)
[![Apache 2 license](https://shields.io/badge/license-Apache%202-blue)](http://www.apache.org/licenses/LICENSE-2.0)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://github.com/BBVA/mercury-robust/issues)

## Mercury project at BBVA

Mercury is a collaborative library that was developed by the Advanced Analytics community at BBVA. Originally, it was created as an [InnerSource](https://en.wikipedia.org/wiki/Inner_source) project but after some time, we decided to release certain parts of the project as Open Source.
That's the case with the `mercury-robust` package.

If you're interested in learning more about the Mercury project, we recommend reading this blog [post](https://www.bbvaaifactory.com/mercury-acelerando-la-reutilizacion-en-ciencia-de-datos-dentro-de-bbva/) from www.bbvaaifactory.com

## Introduction to mercury-robust

`mercury-robust` is a Python library designed for performing robust testing on machine learning models and datasets. It helps ensure that data workflows and models are robust against certain conditions, such as data drift, label leaking or the input data schema, by raising an exception when they fail. This library is intended for data scientists, machine learning engineers and anyone interested in ensuring the performance and robustness of their models in productive environments.

## Importance of ML Robustness

Errors or misbehaviours in machine learning models and datasets can have significant consequences, especially in sensitive domains such as healthcare or finance. It is important to ensure that model performance is align to what was measured in testing environemnts and robust to prevent harm to individuals or organizations that rely on them. `mercury-robust` helps ensure the robustness of machine learning models and datasets by providing a modular framework for performing tests.

## Types of Tests
`mercury-robust` provides two main types of tests: `Data Tests` and `Model Tests`. In addition, all tests can be added to a container class called `TestSuite`

### Data Tests
**Data Tests** receive a dataset as the main input argument and check different conditions. For example, the `CohortPerformanceTest checks whether some metrics perform poorly for some cohorts of data when compared to other groups. This is particularly relevant for measuring fairness in sensitive variables.

### Model Tests
**Model Tests** involve data in combination with a machine learning model. For example, the `ModelSimplicityChecker` evaluates if a simple baseline, trained in the same dataset, gives better or similar performance to a given model. It is used to check if added complexity contributes significantly to improve the model.

if the complexisty of the model is adecuate to the model  measures the importance of every input feature and fails if the model has input features that add very marginal contribution.

### TestSuite
This class provides an easy way to group tests and execute them together. Here's an example of a `TestSuite` that checks for input features that add very marginal importance to the model, the existence of linear combinations in those features, or some kind of data drift:

```python
from mercury.robust.model_tests import ModelSimplicityChecker
from mercury.robust.data_tests import LinearCombinationsTest, DriftTest
from mercury.robust.suite import TestSuite

# Create some tests
complexity_test = ModelSimplicityChecker(
    model = model,
    X_train = X_train,
    y_train = y_train,
    X_test = X_test,
    y_test = y_test,
    threshold = 0.02,
    eval_fn = roc_auc_score
)
drift_test = DriftTest(df_test, train_schma, name="drift_train_test")
lin_comb_test = LinearCombinationsTest(df_train)

# Create the TestSuite with the tests
test_suite = TestSuite(
    tests=[complexity_test, drift_test, lin_comb_test], run_safe=True
)
# Obtain results
test_results = test_suite.get_results_as_df()
```

## Cheatsheet
To help you get started with using `mercury-robust`, we've created a cheatsheet that summarizes the main features and methods of the library. You can download the cheatsheet from here: [RobustCheatsheet.pdf](https://github.com/BBVA/mercury-robust/files/11260658/RobustCheatsheet.pdf)


## User installation

The easiest way to install `mercury-robust` is using ``pip``:

    pip install -U mercury-robust

## Help and support

This library is currently maintained by a dedicated team of data scientists and machine learning engineers from BBVA.

### Documentation
website: https://bbva.github.io/mercury-robust/site/

### Email
mercury.group@bbva.com
