import numpy as np

import myriade


def test_predict():
    A = 0
    B = 1
    C = 2

    class Dummy:
        def __init__(self, label):
            self.label = label

        def predict(self, x):
            return x == self.label

    #  / \
    # A  /\
    #   B  C
    tree = myriade.multiclass.ManualHierarchyClassifier(
        classifier=None,
        tree=myriade.Branch(A, myriade.Branch(B, C, Dummy(B)), Dummy(A)),
    )

    x = np.array([A, B, C, A, B, C])
    tree.fit(x, x)
    assert (tree.predict(x) == x).all()
