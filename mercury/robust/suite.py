from .basetests import RobustTest

import pandas as pd
from typing import List, Union


class TestSuite():
    """
    A TestSuite runs a list of tests. It has two modes of running them. By setting the `run_safe=True`
    all the tests are run and the results of each test are available at the end of the execution using
    the method `get_results_as_df()`. If we set `run_safe=False` a `FailedTestError` exception will
    be raised if a test fails.

    Args:
        tests (List[RobustTest]): A list of RobustTest objects that will be executed
        run_safe (bool): If True, all the tests will be executed independently if some of them fail or
            not. If False, a FailedTestError will be raised when the first test fails and the rest
            won't be executed

    Example:
        ```python
        >>> # Create tests
        >>> drift_test = DriftTest(df_test, train_schma, name="drift_train_test")
        >>> lin_comb_test = LinearCombinationsTest(df_train)
        >>> # Create the TestSuite with the tests
        >>> test_suite = TestSuite(
        >>>     tests=[schema_test, drift_test, lin_comb_test, label_leak_test, noisy_label_test], run_safe=True
        >>> )
        >>> # Obtain results
        >>> test_results = test_suite.get_results_as_df()
        ```
    """

    def __init__(self, tests: List[RobustTest], run_safe: bool = True, *args, **kwargs):
        self._tests = tests
        self._run_safe = run_safe

    def run(self, tests_to_run: List[Union[str, int]] = None):
        """
        Executes all the tests. If attribute `run_safe` is True, an exception is raised when the first
        tests fails. If attribute `run_safe` is False, it executes all the tests even if some of them
        fail.

        Args:
            tests_to_run: If we don't want to run all the test, we can provide this argument with the list
                of index or names of the tests what we want to run.
        """
        for i, test in enumerate(self._tests):
            if self._test_should_run(test, i, tests_to_run):
                if self._run_safe:
                    test.run_safe()
                else:
                    test.run()

    def _test_should_run(self, test, i, tests_to_execute = None):

        if tests_to_execute is None:
            return True
        elif (test.name in tests_to_execute) or (i in tests_to_execute):
            return True
        else:
            return False

    def get_results_as_df(self):
        """
        Obtains a dataframe with a summary of the tests and their execution states

        Returns:
            Pandas DataFrame with the test results

        """
        df_results = pd.DataFrame(
            data={
                "name": [test.name for test in self._tests],
                "state": [test.get_state() for test in self._tests],
                "error": [test.get_error_message() for test in self._tests],
                "info": [test.info() for test in self._tests],
            }
        )
        return df_results