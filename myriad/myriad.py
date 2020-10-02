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

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def to_html(self):
        """Render an HTML tree representation.

        >>> tree = myriad.Branch(
        ...     myriad.Branch(1, 2),
        ...     myriad.Branch(3, 4)
        ... )
        >>> tree.to_html()

        """

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

        mask = train(X, y, self.classifier, tree)

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
        return predict(X, self.tree_)


def split_in_half(l: list):
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


def train(X, y, model, tree):
    """Train a binary tree of models.

    This function navigates through the tree in a depth-first manner. Each leaf returns
    a mask that indicates what part of the data pertains to said leaf. A model is trained
    at each branch. The training data at each branch is determined by the union of the masks
    from the left and right children.

    """

    if not isinstance(tree, Branch):
        return y == tree

    mask = train(X, y, model, tree.left)
    rmask = train(X, y, model, tree.right)
    mask |= rmask

    # TODO: the following could be handled by a pool of workers
    tree.model = base.clone(model).fit(X[mask], rmask[mask])

    return mask


def predict(X, tree):

    if not isinstance(tree, Branch):
        return np.full(X.shape[0], fill_value=tree)

    mask = tree.model.predict(X)

    y_left = predict(X[~mask], tree.left)
    y_pred = np.empty(mask.shape[0], dtype=y_left.dtype)
    y_pred[~mask] = y_left
    y_pred[mask] = predict(X[mask], tree.right)

    return y_pred
