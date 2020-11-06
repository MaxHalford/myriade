import copy
import itertools
import warnings

import numpy as np
from scipy import special
from sklearn import base


__all__ = [
    'Branch',
    'iter_trees',
    'LabelTreeClassifier'
]


class Branch:

    def __init__(self, left, right, model=None):
        self.left = left
        self.right = right
        self.model = model

    def to_html(self):
        """Render an HTML tree representation."""

        def _to_html(el):
            if isinstance(el, Branch):
                return f'''
                <details open>
                    <summary></summary>
                    {_to_html(el.left)}
                    {_to_html(el.right)}
                </details>
                '''
            return f'<div>{el}</div>'

        css = '<style type="text/css">details > *:not(summary){margin-left: 1.5em;}</style>'
        return _to_html(self) + css

    def _iter_edges(self):

        counter = 0

        def iterate(node, depth):

            nonlocal counter
            no = counter

            if isinstance(node, Branch):
                for child in (node.left, node.right):
                    counter += 1
                    yield no, counter, node, child, depth + 1
                    if isinstance(child, Branch):
                        yield from iterate(child, depth=depth + 1)

        yield from iterate(self, depth=0)

    def to_graphviz(self, **kwargs):
        """Return a Graphviz representation.

        Parameters:
            kwargs: Keyword arguments are passed to `graphviz.Graph`.

        """

        import graphviz
        G = graphviz.Graph(**kwargs)

        for i, j, _, child, _ in self._iter_edges():
            G.node(str(i), shape='point')
            if not isinstance(child, Branch):
                G.node(str(j), label=str(child))
            G.edge(str(i), str(j))

        return G

    def _repr_html_(self):
        """Render output in a Jupyter notebook."""
        from IPython.core.display import display, SVG
        svg = self.to_graphviz(format='svg').pipe().decode('utf-8')
        return display(SVG(svg))


class LabelTreeClassifier(base.BaseEstimator, base.ClassifierMixin):

    def __init__(self, classifier: base.ClassifierMixin, prior_tree: Branch = None,
                 n_rounds=1):
        self.classifier = classifier
        self.prior_tree = prior_tree
        self.n_rounds = n_rounds

    def fit(self, X, y):

        self.classes_ = np.unique(y)

        # Build a balanced tree if no prior tree is provided
        if self.prior_tree is None:
            tree = make_balanced_tree(labels=self.classes_)
        else:
            tree = copy.deepcopy(self.prior_tree)

        mask = train(tree, self.classifier, X, y)

        # mask necessarily contains only Trues if the tree contains each label, therefore we
        # know that labels are missing if mask contains Falses
        if not np.all(mask):
            warnings.warn('Not all classes were specified in the prior tree. This might ' +
                          'hinder performance.')

        # TODO: use self.n_rounds

        # Make predictions and establish the confusion matrix
        #y_pred = predict(X, tree)
        #cm = metrics.confusion_matrix(y, y_pred, labels=self.classes_)

        # Build smarter tree
        #self.tree_ = build_smart_tree(self.classes_, cm)
        #train(X, y, self.tree_, self.classifier)

        self.tree_ = tree

        return self

    def predict(self, X):
        y_pred = np.empty(len(X), dtype=self.classes_.dtype)
        predict(self.tree_, X, y_out=y_pred)
        return y_pred

    def predict_proba(self, X):
        y_pred = np.zeros(len(X), dtype=np.dtype([(str(c), 'f') for c in self.classes_]))
        predict_proba(self.tree_, X, y_out=y_pred)
        return y_pred.view(('f', len(self.classes_)))


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
    return Branch(make_balanced_tree(l), make_balanced_tree(r))


def set_splits(s, r):
    """Iterates over all the binary splits of size `r`.

    """

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
    r"""Iterate over every possible labeled binary trees.

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

    Parameters:
        labels: A set of labels.
        k: Determines the number of trees to sample at random. All the trees are returned
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
        n = special.factorial2(2 * (len(labels) - 1) - 1, exact=True)  # number of possible trees
        if isinstance(k, float):
            k = int(k * n)
        if k > n:
            raise ValueError(f'k={k} is higher than the total number of trees ({n})')
        choices = set(np.random.choice(n, size=k, replace=False))
        yield from pick(iter_trees(labels, k=None), choices)
        return
        yield

    if len(labels) == 1:
        yield list(labels)[0]

    for l_size in range((len(labels) + 1) // 2, len(labels)):
        for l_labels, r_labels in set_splits(labels, l_size):
            for l_branch, r_branch in itertools.product(iter_trees(l_labels), iter_trees(r_labels)):
                yield Branch(l_branch, r_branch)


def train(tree, model, X, y):
    """Train a binary tree of models.

    This function navigates through the tree in a depth-first manner. Each leaf returns
    a mask that indicates what part of the data pertains to said leaf. A model is trained
    at each branch. The training data at each branch is determined by the union of the masks
    from the left and right children.

    Note that the assumption is that a True label corresponds to going left down the tree.

    """

    if not isinstance(tree, Branch):
        return y == tree

    lmask = train(tree.left, model, X, y)
    rmask = train(tree.right, model, X, y)
    mask = lmask | rmask

    # TODO: the following could be handled by a pool of workers
    if tree.model is None:
        tree.model = base.clone(model).fit(X[mask], lmask[mask])

    return mask


def predict(tree, X, y_out, y_idx=None):

    if y_idx is None:
        y_idx = np.arange(len(y_out))

    if len(X) == 0:
        return

    # If we're in a leaf, we set the appropriate y rows to the value of the leaf
    if not isinstance(tree, Branch):
        y_out[y_idx] = tree
        return

    # Go left and right according to the branch's model output for each row
    mask = tree.model.predict(X)
    predict(tree.left, X[mask], y_out, y_idx[mask])
    predict(tree.right, X[~mask], y_out, y_idx[~mask])


def predict_proba(tree, X, y_out, p_parent=None):

    if p_parent is None:
        p_parent = np.ones(len(y_out), dtype=float)

    if len(X) == 0:
        return

    if not isinstance(tree, Branch):
        y_out[str(tree)] = p_parent
        return

    p_pred = tree.model.predict_proba(X)

    predict_proba(tree.left, X, y_out, p_parent * p_pred[:, 1])
    predict_proba(tree.right, X, y_out, p_parent * p_pred[:, 0])
