import numpy as np
import pandas as pd
from sklearn import metrics, model_selection
import myriade


class BalancedHierarchyClassifier(myriade.base.HierarchyClassifier):
    """Balanced hierarchy classifier."""

    def __init__(self, classifier, cv=None, base_model=None):
        super().__init__(classifier)
        self.cv = cv or model_selection.KFold(n_splits=2)
        self.base_model = (
            base_model
            or myriade.multiclass.RandomBalancedHierarchyClassifier(classifier)
        )

    def _build_tree(self, X, y, cm=None):
        # If we don't have a confusion matrix, we need to compute one.
        if cm is None:
            y_pred = model_selection.cross_val_predict(
                self.base_model, X, y, cv=self.cv
            )
            cm = metrics.confusion_matrix(y, y_pred, labels=self.classes_)

        confusion = pd.DataFrame(cm, index=self.classes_, columns=self.classes_)
        return pair_labels(confusion)


def make_tree_from_pairs(pairs):
    return myriade.Branch(
        make_tree_from_pairs(pairs[0]) if isinstance(pairs[0], tuple) else pairs[0],
        make_tree_from_pairs(pairs[1]) if isinstance(pairs[1], tuple) else pairs[1],
    )


def pair_labels(confusion):
    labels = confusion.index.tolist()

    errors = np.triu(1 + confusion + confusion.T, k=1)

    # Here we find the pairs of labels that are most confused with each other. We keep going until
    # there are no more pairs to compare.
    pairs_idx = []
    while errors.any():
        i, j = np.unravel_index(np.argmax(errors), errors.shape)
        pairs_idx.append((i, j))
        errors[i, :] = 0
        errors[:, i] = 0
        errors[j, :] = 0
        errors[:, j] = 0
    pairs = [(labels[i], labels[j]) for i, j in pairs_idx]

    # If the number of labels is odd, there will be one label left over. We call this an orphan.
    orphans_idx = list(
        set(range(len(labels)))
        - set(i for i, _ in pairs_idx)
        - set(j for _, j in pairs_idx)
    )
    orphans = [labels[idx] for idx in orphans_idx]

    # We now create a new confusion matrix that combines the pairs of labels.
    confusion_arr = (
        confusion.iloc[
            [p[0] for p in pairs_idx] + orphans_idx,
            [p[0] for p in pairs_idx] + orphans_idx,
        ].to_numpy()
        + confusion.iloc[
            [p[0] for p in pairs_idx] + orphans_idx,
            [p[1] for p in pairs_idx] + orphans_idx,
        ].to_numpy()
        + confusion.iloc[
            [p[1] for p in pairs_idx] + orphans_idx,
            [p[0] for p in pairs_idx] + orphans_idx,
        ].to_numpy()
        + confusion.iloc[
            [p[1] for p in pairs_idx] + orphans_idx,
            [p[1] for p in pairs_idx] + orphans_idx,
        ].to_numpy()
    )

    # HACK: If there are orphans, we need to divide the corresponding rows and columns by 2 to
    # avoid double counting.
    if orphans:
        confusion_arr[-1, :] = confusion_arr[-1, :] / 2
        confusion_arr[:, -1] = confusion_arr[:, -1] / 2

    confusion = pd.DataFrame(
        confusion_arr, index=pairs + orphans, columns=pairs + orphans
    )

    # Termination condition
    if len(confusion.columns) == 1:
        return make_tree_from_pairs(pairs[0])

    return pair_labels(confusion)
