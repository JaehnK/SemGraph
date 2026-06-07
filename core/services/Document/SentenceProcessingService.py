import sys
from typing import List, Optional, Dict
import threading

import spacy

from entities import *

from .TextPreprocssingService import TextPreprocessingService
from ..Sentence.SentenceAnalysisService import SentenceAnalysisService

import sys
from typing import List, Optional, Dict
import threading

class SentenceProcessingService:
    """문장 처리 서비스"""
    
    def __init__(self, documents: Corpus, preprocessing_service: TextPreprocessingService):
        self._documents = documents
        self._preprocessing = preprocessing_service
    
    def create_sentence_list_parallel(self, batch_size: int = 100, n_process: int = None) -> None:
        """병렬 처리로 문장 리스트 생성"""
        rawdata = self._documents.rawdata
        if rawdata is None or len(rawdata) == 0:
            self._documents.sentence_list = []
            return
        
        print(f"Creating {len(rawdata)} sentences with spaCy...", file=sys.stderr)
        
        try:
            # 유효한 문서들만 필터링
            valid_docs = [(i, doc) for i, doc in enumerate(rawdata) if doc.strip()]
            if not valid_docs:
                self._documents.sentence_list = []
                return
            
            # spaCy 배치 처리
            texts = [doc for _, doc in valid_docs]
            # === Contractions 확장 추가 ===
            import contractions # 함수 내에서 임포트하여 필요할 때만 로드
            expanded_texts = [contractions.fix(text) for text in texts]
            processed_docs = self._preprocessing.process_text_batch(expanded_texts, batch_size, n_process)
            
            # Sentence 객체 생성
            sentences = [None] * len(rawdata)  # 원본 인덱스 유지
            
            for (original_idx, original_text), spacy_doc in zip(valid_docs, processed_docs):
                try:
                    sentence = Sentence(raw=original_text, docs_ref=self._documents)
                    SentenceAnalysisService.process_with_spacy(
                        sentence,
                        spacy_doc,
                        self._documents,
                        original_text=original_text,
                    )
                    sentences[original_idx] = sentence
                    
                    if original_idx % 100 == 0 and original_idx > 0:
                        print(f"Processed {original_idx}/{len(rawdata)} documents", file=sys.stderr)
                        
                except Exception as e:
                    print(f'Document {original_idx} processing failed: {e}', file=sys.stderr)
                    sentence = self._create_empty_sentence(original_text)
                    sentences[original_idx] = sentence
            
            # None인 항목들 처리 (빈 문서들)
            for i, sentence in enumerate(sentences):
                if sentence is None:
                    sentences[i] = self._create_empty_sentence(rawdata[i])
            
            self._documents.sentence_list = sentences
            
        except Exception as e:
            print(f"Parallel processing failed, falling back to sequential: {e}", file=sys.stderr)
            self._create_sentence_list_sequential()
        
        # 모든 문장 처리 완료 후 words_list 업데이트
        print(f"Updating words_list after sentence processing...", file=sys.stderr)
        try:
            all_words = self._documents.word_trie.get_all_words()
            print(f"Found {len(all_words) if all_words else 0} unique words in WordTrie", file=sys.stderr)
        except Exception as e:
            print(f"Error updating words_list: {e}", file=sys.stderr)
    
    def create_sentence_list_sequential(self) -> None:
        """순차 처리로 문장 리스트 생성"""
        self._create_sentence_list_sequential()
    
    def _create_sentence_list_sequential(self) -> None:
        """순차 처리 구현"""
        rawdata = self._documents.rawdata
        if not rawdata:
            self._documents.sentence_list = []
            return
        
        print(f"Creating {len(rawdata)} sentences sequentially...", file=sys.stderr)
        
        sentences = []
        for i, doc_text in enumerate(rawdata):
            if i % 100 == 0 and i > 0:
                print(f"Processed {i}/{len(rawdata)} documents", file=sys.stderr)
            
            try:
                import contractions # 함수 내에서 임포트하여 필요할 때만 로드
                expanded_doc_text = contractions.fix(doc_text)
                sentence = Sentence(raw=doc_text, docs_ref=self._documents)
                spacy_doc = self._preprocessing.nlp(expanded_doc_text)
                SentenceAnalysisService.process_with_spacy(
                    sentence,
                    spacy_doc,
                    self._documents,
                    original_text=doc_text,
                )
                sentences.append(sentence)
            except Exception as e:
                print(f'Document {i} processing failed: {e}', file=sys.stderr)
                sentences.append(self._create_empty_sentence(doc_text))
        
        self._documents.sentence_list = sentences
    
    def _create_empty_sentence(self, text: str) -> Sentence:
        """실패한 경우 빈 Sentence 객체 생성"""
        sentence = Sentence(raw=text, docs_ref=self._documents)
        sentence.set_processed_data([], [], [])
        return sentence
