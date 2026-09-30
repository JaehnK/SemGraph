import sys
from typing import List, Optional, Dict
import threading

from entities import *

class TextPreprocessingService:
    """텍스트 전처리 서비스 (spaCy 관리)"""
    
    def __init__(self, model_name='en_core_web_sm', disable_components=None):
        self._nlp = None
        self._model_name = model_name
        self._disable_components = disable_components or ['parser', 'ner']
        self._lock = threading.Lock()
        self._initialize_spacy()
    
    def _initialize_spacy(self):
        """spaCy 모델 초기화 (spaCy는 이 시점에만 import)"""
        import spacy

        try:
            self._nlp = spacy.load(self._model_name, disable=self._disable_components)
            self._nlp.max_length = 2000000
            print(f"Loaded spaCy model: {self._model_name}", file=sys.stderr)
        except OSError:
            print(f"spaCy model '{self._model_name}' not found. Please install it:", file=sys.stderr)
            print(f"python -m spacy download {self._model_name}", file=sys.stderr)
            raise
        
    @property
    def nlp(self):
        return (self._nlp)
    
    def process_text_batch(self, texts: List[str], batch_size: int = 100, n_process: int = None) -> List:
        """텍스트 배치 처리"""
        if n_process is None:
            n_process = min(4, len(texts) // 50)
        
        valid_texts = [text for text in texts if text.strip()]
        if not valid_texts:
            return []
        
        print(f"Processing {len(valid_texts)} texts with {n_process} processes...", file=sys.stderr)
        
        return list(self._nlp.pipe(
            valid_texts,
            batch_size=batch_size,
            n_process=n_process if n_process > 1 else 1
        ))