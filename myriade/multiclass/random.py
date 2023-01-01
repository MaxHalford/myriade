import random
import myriade


class RandomHierarchyClassifier(myriade.base.HierarchyClassifier):
    """Random hierarchy classifier.

    For those who are feeling lucky.

    """

    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.tree_ = make_balanced_tree(labels=self.classes_)
        return self


def split_in_half(l: list, shuffle=False):
    """Split a list in half.

    >>> split_in_half([1, 2, 3, 4])
    ([1, 2], [3, 4])

    >>> split_in_half([1, 2, 3])
    ([1, 2], [3])

    """
    if shuffle:
        random.shuffle(l)
    cut = (len(l) + 1) // 2
    return l[:cut], l[cut:]


def make_balanced_tree(labels: list, shuffle=False):
    if len(labels) == 1:
        return labels[0]
    l, r = split_in_half(labels, shuffle=shuffle)
    return Branch(
        make_balanced_tree(l, shuffle=shuffle), make_balanced_tree(r, shuffle=shuffle)
    )
