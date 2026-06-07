from .SemGraphConfig import SemGraphConfig

GRACEConfig = SemGraphConfig

__all__ = [
    'SemGraphPipeline',
    'SemGraphConfig',
    'GRACEPipeline',
    'GRACEConfig',
    'TraditionalGraphClusteringService'
]


def __getattr__(name):
    if name in ('SemGraphPipeline', 'GRACEPipeline'):
        from .SemGraphPipeline import SemGraphPipeline
        return SemGraphPipeline
    if name == 'TraditionalGraphClusteringService':
        from .TraditionalGraphClusteringService import TraditionalGraphClusteringService
        return TraditionalGraphClusteringService
    if name in ('SemGraphConfig', 'GRACEConfig'):
        return SemGraphConfig
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
