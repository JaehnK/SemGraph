__all__ = [
    'GraphService',
    'NodeFeatureHandler',
    'CrossAttentionFusion',
    'BiDirectionalFusion',
    'LearnedWeightedFusion',
    'GatedFusion',
    'AttentionFusionFactory'
]


def __getattr__(name):
    if name == 'GraphService':
        from .GraphService import GraphService
        return GraphService
    if name == 'NodeFeatureHandler':
        from .NodeFeatureHandler import NodeFeatureHandler
        return NodeFeatureHandler
    if name in (
        'CrossAttentionFusion',
        'BiDirectionalFusion',
        'LearnedWeightedFusion',
        'GatedFusion',
        'AttentionFusionFactory',
    ):
        from . import AttentionFusion
        return getattr(AttentionFusion, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
