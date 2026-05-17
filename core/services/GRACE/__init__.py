from .GRACEConfig import GRACEConfig

__all__ = ['GRACEPipeline', 'GRACEConfig', 'TraditionalGraphClusteringService']


def __getattr__(name):
    if name == 'GRACEPipeline':
        from .GRACEPipeline import GRACEPipeline
        return GRACEPipeline
    if name == 'TraditionalGraphClusteringService':
        from .TraditionalGraphClusteringService import TraditionalGraphClusteringService
        return TraditionalGraphClusteringService
    if name == 'GRACEConfig':
        return GRACEConfig
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
