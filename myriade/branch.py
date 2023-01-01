class Branch:
    def __init__(self, left, right, model=None):
        self.left = left
        self.right = right
        self.model = model

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
            G.node(str(i), shape="point")
            if not isinstance(child, Branch):
                G.node(str(j), label=str(child))
            G.edge(str(i), str(j))

        return G

    def _repr_html_(self):
        """Render output in a Jupyter notebook."""

        return self.to_graphviz(format="svg").pipe().decode("utf-8")
