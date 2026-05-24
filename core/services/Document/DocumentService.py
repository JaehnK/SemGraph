import sys
from typing import List, Optional, Dict, Any
import threading

import spacy

from entities import *

from .TextPreprocssingService import TextPreprocessingService
from .SentenceProcessingService import SentenceProcessingService
from .WordManagementService import WordManagementService
from ..Word.wordAnalysisService import WordAnalysisService
from ..Word.wordStatisticsService import WordStatisticsService

class DocumentService:
    """문서 처리 Facade 서비스 - 다른 서비스들을 조율"""
    
    def __init__(self, model_name='en_core_web_sm', disable_components=None):
        # 데이터 레이어
        self._documents = Corpus()
        
        # 서비스 레이어들
        self._preprocessing = TextPreprocessingService(model_name, disable_components)
        self._sentence_service = SentenceProcessingService(self._documents, self._preprocessing)
        self._word_service = WordManagementService(self._documents)
        self._word_analysis = WordAnalysisService()
        self._word_stats = WordStatisticsService()
    
    # === 기존 Docs 클래스 호환 인터페이스 ===
    
    @property
    def rawdata(self) -> Optional[List[str]]:
        """원본 데이터 조회"""
        return self._documents.rawdata
    
    @rawdata.setter
    def rawdata(self, docs: List[str]):
        """원본 데이터 설정 및 자동 처리"""
        self._documents.rawdata = docs
    
    @property
    def words_list(self) -> Optional[List[Word]]:
        """단어 리스트 조회"""
        return self._word_service.get_all_words()
    
    @property
    def sentence_list(self) -> Optional[List[Sentence]]:
        """문장 리스트 조회"""
        return self._documents.sentence_list
    
    def add_word(self, word_content: str, pos_tag: str = None) -> Word:
        """단어 추가"""
        return self._word_service.add_word(word_content, pos_tag)
    
    def create_sentence_list(self, documents: Optional[List[str]] = None, max_workers: int = 4, batch_size: int = 100, n_processes: int = None) -> None:
        """문장 리스트를 생성합니다.
        documents 인자가 제공되면 해당 문서를 처리하고, 그렇지 않으면 내부 _documents.rawdata를 사용합니다.
        max_workers 기본값이 4로 설정되어 있어, 명시적으로 1을 지정하지 않는 한 병렬 처리됩니다.

        Args:
            documents (Optional[List[str]]): 처리할 문서 리스트. 제공되지 않으면 기존 rawdata를 사용합니다.
            max_workers (int): 병렬 처리 시 사용할 워커 수. 1보다 크면 병렬 처리합니다. 기본값은 4입니다.
            batch_size (int): 병렬 처리 시 spaCy 배치 크기.
            n_processes (Optional[int]): 병렬 처리 시 사용할 프로세스 수. None이면 CPU 코어 수를 사용합니다.
        """
        if documents is not None:
            self._documents.rawdata = documents # 제공된 문서를 내부 _documents 객체에 설정

        # 처리할 rawdata가 없으면 빈 리스트로 초기화하고 종료
        if not self._documents.rawdata:
            self._documents.sentence_list = []
            return

        if max_workers > 1:
            self.create_sentence_list_parallel(batch_size=batch_size, n_processes=n_processes)
        else:
            self._sentence_service.create_sentence_list_sequential()
    
    def create_sentence_list_parallel(self, batch_size: int = 100, n_processes: int = None) -> None:
        """병렬 문장 리스트 생성"""
        self._sentence_service.create_sentence_list_parallel(batch_size, n_processes)
    
    def get_co_occurrence_edges(self, word_to_node: Dict[str, int]):
        """공출현 엣지 생성 (GraphService에서 사용하기 위한 데이터 제공)"""
        return self._documents.get_co_occurrence_edges(word_to_node)
    
    def get_preprocessed_sentences(self)-> List[List[str]]:
        sentences = self._documents.sentence_list
        if sentences is None:
            return []
        return [sentence.lemmatised for sentence in sentences]
    
    def get_sentences_with_word2id(self)-> List[List[int]]:
        sentences = self._documents.sentence_list
        if sentences is None:
            return []
        return [sentence.word_indices for sentence in sentences]
    
    def get_word2vec_data(self, min_count: int = 5) -> Dict[str, Any]:
        """Word2Vec에 필요한 모든 데이터 반환"""
        words = self._word_service.get_all_words()
        
        # min_count 필터링
        filtered_words = [w for w in words if w.freq >= min_count]
        
        # 새로운 0-indexed ID 맵 생성
        new_word2id: Dict[str, int] = {}
        new_id2word: Dict[int, str] = {}
        new_word_frequency: Dict[int, int] = {}
        
        # 0부터 시작하는 새로운 ID 할당
        for i, word_obj in enumerate(filtered_words):
            new_word2id[word_obj.content] = i
            new_id2word[i] = word_obj.content
            new_word_frequency[i] = word_obj.freq
            
        total_tokens = sum(w.freq for w in filtered_words)
        
        return {
            'word2id': new_word2id,
            'id2word': new_id2word,
            'word_frequency': new_word_frequency,
            'vocab_size': len(filtered_words),
            'total_tokens': total_tokens
        }
    
    # === 새로운 Facade 메서드들 ===
    
    def analyze_word(self, word_content: str) -> Dict[str, Any]:
        """단어 분석 결과 반환"""
        word = self._word_service.get_word(word_content)
        if not word:
            return None
        
        return {
            'word': word,
            'pos_category': self._word_analysis.get_pos_category(word),
            'is_content_word': self._word_analysis.is_content_word(word),
            'is_function_word': self._word_analysis.is_function_word(word),
            'statistics': self._word_stats.get_basic_stats(word),
            'pos_distribution': self._word_stats.get_pos_distribution(word)
        }
    
    def get_top_words(self, top_n: int = 10, exclude_stopwords: bool = False) -> List[Word]:
        """상위 단어들 반환 (파라미터 이름 통일: exclude_stopwords)"""
        words = self._word_service.get_all_words()
        top_words = self._word_stats.get_top_words_by_frequency(words, top_n, exclude_stopwords)
        
        # return [
        #     {
        #         'word': word,
        #         'analysis': self._word_analysis.get_pos_category(word),
        #         'stats': self._word_stats.get_basic_stats(word)
        #     }
        #     for word in top_words
        # ]
        return [ word for word in top_words]
    
    def get_word_statistics_summary(self) -> Dict[str, Any]:
        """전체 단어 통계 요약"""
        words = self._word_service.get_all_words()
        return self._word_stats.get_word_frequency_summary(words)
    
    def get_pos_statistics(self) -> Dict[str, int]:
        """품사별 통계"""
        words = self._word_service.get_all_words()
        return self._word_stats.get_pos_statistics(words)
    
    def get_words_by_pos_category(self, pos_category: str) -> List[Word]:
        """특정 품사 카테고리에 속하는 단어들 반환"""
        words = self._word_service.get_all_words()
        return [
            word for word in words 
            if self._word_analysis.get_pos_category(word) == pos_category
        ]
    
    def __str__(self):
        """기존 호환성"""
        return f"DocumentService(docs={self.get_document_count()}, sentences={self.get_sentence_count()}, words={self.get_word_count()})"
    
    def get_document_count(self) -> int:
        """문서 개수"""
        return len(self._documents.rawdata) if self._documents.rawdata else 0
    
    def get_sentence_count(self) -> int:
        """문장 개수"""
        return len(self._documents.sentence_list) if self._documents.sentence_list else 0
    
    def get_word_count(self) -> int:
        """고유 단어 개수"""
        return self._word_service.get_word_count() if self._word_service else 0
