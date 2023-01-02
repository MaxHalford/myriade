import random
import myriade
import numpy as np
from sklearn import utils


class RandomBalancedHierarchyClassifier(myriade.base.HierarchyClassifier):
    """Random balanced hierarchy classifier.

    Parameters
    ----------
    classifier
        The base classifier.
    seed
        The random seed.

    """

    def __init__(self, classifier, seed=None):
        super().__init__(classifier)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _build_tree(self, X, y):
        self.classes_ = np.unique(y)
        labels = self.classes_.copy()
        self.rng.shuffle(labels)
        return make_balanced_tree(labels=labels)


def split_in_half(l: list):
    """Split a list in half.

    >>> split_in_half([1, 2, 3, 4])
    ([1, 2], [3, 4])

    >>> split_in_half([1, 2, 3])
    ([1, 2], [3])

    """
    cut = (len(l) + 1) // 2
    return l[:cut], l[cut:]


def make_balanced_tree(labels: list):
    if len(labels) == 1:
        return labels[0]
    l, r = split_in_half(labels)
    return myriade.Branch(make_balanced_tree(l), make_balanced_tree(r))
