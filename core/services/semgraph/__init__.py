from .SemGraphConfig import SemGraphConfig

__all__ = [
    'SemGraphPipeline',
    'SemGraphConfig',
    'TraditionalGraphClusteringService'
]


def __getattr__(name):
    if name == 'SemGraphPipeline':
        from .SemGraphPipeline import SemGraphPipeline
        return SemGraphPipeline
    if name == 'TraditionalGraphClusteringService':
        from .TraditionalGraphClusteringService import TraditionalGraphClusteringService
        return TraditionalGraphClusteringService
    if name == 'SemGraphConfig':
        return SemGraphConfig
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
