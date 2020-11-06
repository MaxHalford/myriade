import numpy as np

import myriad


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
    tree = myriad.LabelTreeClassifier(
        classifier=None,
        prior_tree=myriad.Branch(A, myriad.Branch(B, C, Dummy(B)), Dummy(A))
    )

    x = np.array([A, B, C, A, B, C])
    tree.fit(x, x)
    assert (tree.predict(x) == x).all()
