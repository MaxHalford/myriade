import numpy as np
from sklearn import metrics, model_selection
import myriade


class BalancedHierarchyClassifier(myriade.base.HierarchyClassifier):
    """Balanced hierarchy classifier.

    Parameters
    ----------
    classifier
        The base classifier.
    base_model
        The base model to use for computing the confusion matrix.
    cv
        The cross-validation strategy applied to the base model.

    """

    def __init__(self, classifier, base_model=None, cv=None):
        super().__init__(classifier)
        self.base_model = (
            base_model
            or myriade.multiclass.RandomBalancedHierarchyClassifier(classifier)
        )
        self.cv = cv or model_selection.KFold(
            n_splits=2,
            shuffle=True,
        )

    def _build_tree(self, X, y, cm=None):
        # If we don't have a confusion matrix, we need to compute one.
        if cm is None:
            y_pred = model_selection.cross_val_predict(
                self.base_model, X, y, cv=self.cv
            )
            cm = metrics.confusion_matrix(y, y_pred, labels=self.classes_)

        return pair_labels(cm, self.classes_)


def make_tree_from_pairs(pairs):
    return myriade.Branch(
        make_tree_from_pairs(pairs[0]) if isinstance(pairs[0], tuple) else pairs[0],
        make_tree_from_pairs(pairs[1]) if isinstance(pairs[1], tuple) else pairs[1],
    )


def pair_labels(confusion, labels):
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
    confusion = (
        confusion[[p[0] for p in pairs_idx] + orphans_idx][
            :, [p[0] for p in pairs_idx] + orphans_idx
        ]
        + confusion[[p[0] for p in pairs_idx] + orphans_idx][
            :, [p[1] for p in pairs_idx] + orphans_idx
        ]
        + confusion[[p[1] for p in pairs_idx] + orphans_idx][
            :, [p[0] for p in pairs_idx] + orphans_idx
        ]
        + confusion[[p[1] for p in pairs_idx] + orphans_idx][
            :, [p[1] for p in pairs_idx] + orphans_idx
        ]
    )

    # HACK: If there are orphans, we need to divide the corresponding rows and columns by 2 to
    # avoid double counting.
    if orphans:
        confusion[-1, :] = confusion[-1, :] / 2
        confusion[:, -1] = confusion[:, -1] / 2

    # Termination condition
    if confusion.shape == (1, 1):
        return make_tree_from_pairs(pairs[0])

    return pair_labels(confusion, pairs + orphans)
