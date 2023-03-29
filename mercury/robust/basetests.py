from abc import ABC
from enum import Enum
import pandas as pd

from mercury.robust.errors import FailedTestError

from mercury.dataschema.feature import (
    FeatureFactory,
    ContinuousFeature,
    DiscreteFeature,
    CategoricalFeature,
    BinaryFeature
)


class TestState(Enum):
    """
    Enumeration class with the different states that a Test can be.
    """
    SUCCESS = 'SUCCESS'
    FAIL = 'FAIL'
    NOT_EXECUTED = 'NOT_EXECUTED'
    EXCEPTION = 'EXCEPTION'


class RobustTest(ABC):
    """
    Base class for the robust testing implementations. It keeps three attributes:

        1) `_name` (str): name of the test. It can be specified when constructing the test, otherwise
            it will take the name of the class.
        2) `_state `(TestState): state in which the test is found. Initially, NOT_EXECUTED
        3) `_error_message` (str): After running the test, this contains an error message if the
            test fails or runs an exception

    """
    def __init__(self, name=None, *args, **kwargs):
        self._name = name
        self._state = TestState.NOT_EXECUTED
        self._error_mesage = ""

    @property
    def name(self) -> str:
        if self._name is None:
            return self.__class__.__name__
        else:
            return self._name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def info(self) -> dict:
        return None

    def get_error_message(self) -> str:
        return self._error_mesage

    def get_state(self) -> TestState:
        return self._state

    def run(self, *args, **kwargs):
        """
        Run the test, raising FailedTestError exception if it fails
        It must be implemented in subclasses
        """
        pass

    def run_safe(self, *args, **kwargs):
        """
        Runs the test without raising FailedTestError exception. Instead, the attribute
        `_state` will store the state that has reached the test, and the attribute
        `_error_message` will contain a message in case that an error happens.
        """
        try:
            self.run(*args, **kwargs)
            self._state = TestState.SUCCESS
            self._error_mesage = ""
        except FailedTestError as e:
            self._state = TestState.FAIL
            self._error_mesage = str(e)
        except Exception as e:
            self._state = TestState.EXCEPTION
            self._error_mesage = e.__repr__()


class RobustDataTest(RobustTest):
    def __init__(self, base_dataset, name=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.base_dataset = base_dataset


class RobustModelTest(RobustTest):
    def __init__(self, model, name=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.model = model


class TaskInferrer():
    def infer_task(
        self,
        y: "np.array"  # noqa: F821
    ) -> str:

        # Case multiclass classification with labels encoded as vectors
        if (len(y.shape) == 2) and (y.shape[1] > 1):
            return 'classification'

        feature_factory = FeatureFactory()
        y_feature = feature_factory.build_feature(pd.Series(y))
        if isinstance(y_feature, BinaryFeature) or isinstance(y_feature, CategoricalFeature):
            return 'classification'
        elif isinstance(y_feature, ContinuousFeature) or isinstance(y_feature, DiscreteFeature):
            return 'regression'
        else:
            raise ValueError(
                "Unable to infer task from 'y'. Provide a valid 'y' or specify "
                "the task with 'task' param "
            )