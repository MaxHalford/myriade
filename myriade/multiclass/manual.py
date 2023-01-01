import myriade


class ManualHierarchyClassifier(myriade.base.HierarchyClassifier):
    """Manual hierarchy classifier.

    For the lucky ones who already have a hierarchy.

    """

    def __init__(self, classifier, tree):
        super().__init__(classifier)
        self.tree = tree

    def _build_tree(self, X, y):
        return self.tree
