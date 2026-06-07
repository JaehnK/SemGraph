# Lazy imports to avoid dependency issues
# Import only when needed to prevent circular dependencies and missing optional packages

__all__ = [
    'DocumentService', 'SentenceProcessingService', 'TextPreprocessingService', 'WordManagementService',
    'SentenceAnalysisService', 'WordAnalysisService', 'WordStatisticsService',
    'GraphService', 'VisualizationService'
]

def __getattr__(name):
    """Lazy import to avoid loading all dependencies at once"""
    if name == 'DocumentService':
        from .Document.DocumentService import DocumentService
        return DocumentService
    elif name == 'SentenceProcessingService':
        from .Document.SentenceProcessingService import SentenceProcessingService
        return SentenceProcessingService
    elif name == 'TextPreprocessingService':
        from .Document.TextPreprocssingService import TextPreprocessingService
        return TextPreprocessingService
    elif name == 'WordManagementService':
        from .Document.WordManagementService import WordManagementService
        return WordManagementService
    elif name == 'SentenceAnalysisService':
        from .Sentence.SentenceAnalysisService import SentenceAnalysisService
        return SentenceAnalysisService
    elif name == 'WordAnalysisService':
        from .Word.wordAnalysisService import WordAnalysisService
        return WordAnalysisService
    elif name == 'WordStatisticsService':
        from .Word.wordStatisticsService import WordStatisticsService
        return WordStatisticsService
    elif name == 'GraphService':
        from .Graph.GraphService import GraphService
        return GraphService
    elif name == 'VisualizationService':
        from .Visualization.VisualizationService import VisualizationService
        return VisualizationService
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
