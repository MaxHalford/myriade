import itertools
import operator
import numpy as np
from scipy import special
from sklearn import metrics, model_selection
import myriade


class OptimalHierarchyClassifier(myriade.base.HierarchyClassifier):
    """Optimal hierarchy classifier.

    Parameters
    ----------
    classifier
        The base classifier.
    cv
        The cross-validation strategy.
    scorer
        The scorer.

    """

    def __init__(self, classifier, cv=None, scorer=None):
        super().__init__(classifier)
        self.cv = cv or model_selection.KFold(n_splits=2, shuffle=True)
        self.scorer = scorer or metrics.make_scorer(metrics.accuracy_score)

    def _score_tree(self, X, y, tree):
        """Score a tree by cross-validation."""

        model = myriade.multiclass.ManualHierarchyClassifier(
            classifier=self.classifier, tree=tree
        )
        cv_scores = model_selection.cross_val_score(
            model, X, y, cv=self.cv, scoring=self.scorer
        )

        return np.mean(cv_scores)

    def _build_tree(self, X, y):
        best_score = 0
        best_tree = None
        op = operator.gt if self.scorer._sign else operator.lt

        for tree in iter_trees(set(y)):
            score = self._score_tree(X, y, tree)
            if op(score, best_score):
                best_score = score
                best_tree = tree

        return best_tree


def set_splits(s, r):
    """Iterates over all the binary splits of size `r`."""

    s = set(s)
    combos = map(set, itertools.combinations(s, r))

    if len(s) == 2 * r:
        n = special.comb(2 * (r - 1) + 1, r, exact=True)  # OEIS A001700
        combos = itertools.islice(combos, n)

    for combo in combos:
        other = s - combo
        yield combo, other


def pick(iterable, indexes):
    """

    >>> list(pick(['A', 'B', 'C', 'D'], [1, 2]))
    ['B', 'C']

    >>> list(pick(iter(['A', 'B', 'C', 'D']), [1, 2]))
    ['B', 'C']

    """

    indexes = sorted(indexes)
    iterable = enumerate(iterable)

    for index in indexes:
        while True:
            i, el = next(iterable)
            if i == index:
                yield el
                break


def iter_trees(labels, k: int = None):
    """Iterate over every possible labeled binary trees.

    As an example, for n = 3, the possible trees are:

      /\     /\     /\
     /\ 3   /\ 2   /\ 1
    1  2   1  3   2  3

    Therefore, structural symmetries as well as label permutations are not ignored. The reason why
    is that we only want to consider trees that lead to a different hierarchy of models, and thus a
    different predictive power. Indeed, all of the following trees are equivalent in our case:

      /\     /\     /\     /\
     /\ 3   /\ 3   3 /\   3 /\
    1  2   2  1     1  2   2  1

    The number of results is equal to the double factorial of odd numbers, which corresponds
    to sequence A001147 on the Online Encyclopedia of Integer Sequences (OEIS).

    Parameters
    ----------
    labels
        A set of labels.
    k
        Determines the number of trees to sample at random. All the trees are returned
        when `k = None`, which is the default behaviour. If `k` is a `float`, then it is
        interpreted as the percentage of trees to sample.

    >>> sum(1 for _ in iter_trees(range(1)))
    1

    >>> sum(1 for _ in iter_trees(range(2)))
    1

    >>> sum(1 for _ in iter_trees(range(3)))
    3

    >>> sum(1 for _ in iter_trees(range(4)))
    15

    >>> sum(1 for _ in iter_trees(range(5)))
    105

    """

    if k is not None:
        n = special.factorial2(
            2 * (len(labels) - 1) - 1, exact=True
        )  # number of possible trees
        if isinstance(k, float):
            k = int(k * n)
        if k > n:
            raise ValueError(f"k={k} is higher than the total number of trees ({n})")
        choices = set(np.random.choice(n, size=k, replace=False))
        yield from pick(iter_trees(labels, k=None), choices)
        return
        yield

    if len(labels) == 1:
        yield list(labels)[0]

    for l_size in range((len(labels) + 1) // 2, len(labels)):
        for l_labels, r_labels in set_splits(labels, l_size):
            for l_branch, r_branch in itertools.product(
                iter_trees(l_labels), iter_trees(r_labels)
            ):
                yield myriade.Branch(l_branch, r_branch)
