import os, shutil

from mercury.robust.create_tutorials import create_tutorials


def test_create_tutorials():
    # Silently remove the full tree './robust_tutorials/'
    shutil.rmtree('./robust_tutorials', ignore_errors = True)

    create_tutorials('./')

    assert os.path.isfile('./robust_tutorials/RobustTestingExample.ipynb')

    # Clean up
    shutil.rmtree('./robust_tutorials', ignore_errors = True)


if __name__ == "__main__":
    test_create_tutorials()
