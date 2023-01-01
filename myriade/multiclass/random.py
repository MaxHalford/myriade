import random
import myriade
import numpy as np
from sklearn import utils


class RandomBalancedHierarchyClassifier(myriade.base.HierarchyClassifier):
    """Random balanced hierarchy classifier.

    For those who are feeling lucky.

    Examples
    --------

    >>> import myriade
    >>> from sklearn import datasets
    >>> from sklearn import linear_model
    >>> from sklearn import model_selection
    >>> from sklearn import pipeline
    >>> from sklearn import preprocessing

    >>> X, y = datasets.load_digits(return_X_y=True)

    >>> X_train, X_test, y_train, y_test = model_selection.train_test_split(
    ...     X, y, test_size=0.4, random_state=42
    ... )

    >>> model = pipeline.make_pipeline(
    ...     preprocessing.StandardScaler(),
    ...     myriade.multiclass.RandomBalancedHierarchyClassifier(
    ...         classifier=linear_model.LogisticRegression(),
    ...         seed=42
    ...     )
    ... )
    >>> model = model.fit(X_train, y_train)
    >>> print(f"{model.score(X_test, y_test):.2%}")
    89.43%

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
