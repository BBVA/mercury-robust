import pytest

from mercury.robust.basetests import RobustTest, TestState
from mercury.robust.errors import FailedTestError

class DummyTest(RobustTest):

    def __init__(self, test_param="succeed", name=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.test_param = test_param

    def run(self, *args, **kwargs):
        super().run(*args, **kwargs)

        if self.test_param == "fail":
            raise FailedTestError("test has failed")
        elif self.test_param == "exception":
            raise Exception("test has raised exception")

def test_robust_test():

    # Test that succeeds
    dummy_test = DummyTest(name="test1")
    assert dummy_test.name == "test1"
    assert dummy_test._state == TestState.NOT_EXECUTED
    dummy_test.run_safe()
    assert dummy_test._state == TestState.SUCCESS

    # Test that fails
    dummy_test = DummyTest(test_param="fail")
    assert dummy_test._state == TestState.NOT_EXECUTED
    dummy_test.run_safe()
    assert dummy_test._state == TestState.FAIL

    # Test that raises exception
    dummy_test = DummyTest(test_param="exception")
    assert dummy_test._state == TestState.NOT_EXECUTED
    dummy_test.run_safe()
    assert dummy_test._state == TestState.EXCEPTION
