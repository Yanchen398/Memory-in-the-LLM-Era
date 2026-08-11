class GraphMem:
    """Placeholder for the optional graph-memory backend.

    The upstream LightMem implementation does not currently provide graph
    operations, but the core class conditionally instantiates this object when
    ``graph_mem`` is enabled. Keeping a valid placeholder preserves that API
    without breaking imports for the standard (non-graph) pipeline.
    """

    def __init__(self, config=None):
        self.config = config
