import abc
import warnings
import numpy as np
from sklearn import base

from .branch import Branch


class HierarchyClassifier(base.BaseEstimator, base.ClassifierMixin, abc.ABC):
    def __init__(self, classifier: base.ClassifierMixin):
        self.classifier = classifier

    @abc.abstractmethod
    def _build_tree(self, X, y):
        ...

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.tree_ = self._build_tree(X, y)
        mask = train(self.tree_, self.classifier, X, y)

        # mask necessarily contains only Trues if the tree contains each label, therefore we
        # know that labels are missing if mask contains Falses
        if not np.all(mask):
            warnings.warn(
                "Not all classes were specified in the prior tree. This might "
                + "hinder performance."
            )

        return self

    def predict(self, X):
        y_pred = np.empty(len(X), dtype=self.classes_.dtype)
        predict(self.tree_, X, y_out=y_pred)
        return y_pred

    def predict_proba(self, X):
        y_pred = np.zeros(
            len(X), dtype=np.dtype([(str(c), "f") for c in self.classes_])
        )
        predict_proba(self.tree_, X, y_out=y_pred)
        return y_pred.view(("f", len(self.classes_)))


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
