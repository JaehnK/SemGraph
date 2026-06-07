__all__ = [
    'GraphService',
    'NodeFeatureHandler'
]


def __getattr__(name):
    if name == 'GraphService':
        from .GraphService import GraphService
        return GraphService
    if name == 'NodeFeatureHandler':
        from .NodeFeatureHandler import NodeFeatureHandler
        return NodeFeatureHandler
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
