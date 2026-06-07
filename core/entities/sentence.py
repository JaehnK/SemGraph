from typing import Any, Dict, List, Optional

from .word import Word


class Sentence:
    """문장 처리 결과를 보관하는 경량 데이터 객체."""

    def __init__(
        self,
        raw: str = "",
        doc_id: Optional[str] = None,
        sentence_id: Optional[str] = None,
        docs_ref: Optional[Any] = None,
    ):
        self.__raw = raw
        self.doc_id = doc_id
        self.sentence_id = sentence_id or f"sent_{id(self)}"
        self.docs_ref = docs_ref

        self.lemmatised: List[str] = []
        self.word_objects: List[Word] = []
        self.word_indices: List[int] = []

        self.language: Optional[str] = None
        self.pos_tags: List[str] = []

        self.char_count = len(raw) if raw else 0
        self.word_count = len(raw.split()) if raw else 0
        self.processing_errors: List[str] = []
        self.is_processed = False

    @property
    def raw(self) -> str:
        return self.__raw

    @raw.setter
    def raw(self, sentence: str):
        self.__raw = sentence
        self.char_count = len(sentence) if sentence else 0
        self.word_count = len(sentence.split()) if sentence else 0
        self.is_processed = False

    def set_processed_data(
        self,
        lemmatised: List[str],
        word_objects: List[Word],
        word_indices: List[int],
        pos_tags: Optional[List[str]] = None,
    ) -> None:
        """전처리 서비스가 만든 처리 결과를 설정한다."""
        self.lemmatised = list(lemmatised)
        self.word_objects = list(word_objects)
        self.word_indices = list(word_indices)
        self.pos_tags = list(pos_tags or [])
        self.is_processed = True
        self.word_count = len(self.lemmatised)

    def add_word_data(
        self,
        word: Word,
        lemma: str,
        word_index: int,
        pos_tag: Optional[str] = None,
    ) -> None:
        self.word_objects.append(word)
        self.lemmatised.append(lemma)
        self.word_indices.append(word_index)

        if pos_tag:
            self.pos_tags.append(pos_tag)

        self.word_count = len(self.lemmatised)

    def add_processing_error(self, error: str) -> None:
        self.processing_errors.append(error)

    def clear_processing_errors(self) -> None:
        self.processing_errors.clear()

    def lemmatise(self) -> List[str]:
        """기존 호출부 호환을 위해 처리된 lemma 목록을 반환한다."""
        return self.lemmatised

    def set_from_spacy_doc(self, spacy_doc, original_text: str):
        """기존 호출부 호환 shim. 실제 처리는 SentenceAnalysisService가 담당한다."""
        try:
            from services.Sentence.SentenceAnalysisService import SentenceAnalysisService
        except ImportError:
            from core.services.Sentence.SentenceAnalysisService import SentenceAnalysisService

        SentenceAnalysisService.process_with_spacy(
            self,
            spacy_doc,
            self.docs_ref,
            original_text=original_text,
        )

    def _process_with_fallback(self) -> None:
        """기존 호출부 호환 shim."""
        try:
            from services.Sentence.SentenceAnalysisService import SentenceAnalysisService
        except ImportError:
            from core.services.Sentence.SentenceAnalysisService import SentenceAnalysisService

        SentenceAnalysisService.process_with_fallback(self, self.docs_ref)

    def get_words_by_pos(self, pos_category: str) -> List[Word]:
        if not self.pos_tags or len(self.pos_tags) != len(self.word_objects):
            return []

        return [
            word
            for word, pos in zip(self.word_objects, self.pos_tags)
            if pos.upper().startswith(pos_category.upper())
        ]

    def get_content_words(self) -> List[Word]:
        return [word for word in self.word_objects if not word.get_stopword_status()]

    def get_sentence_stats(self) -> Dict[str, Any]:
        unique_words = len({word.content for word in self.word_objects})
        content_words = len(self.get_content_words())

        return {
            'sentence_id': self.sentence_id,
            'doc_id': self.doc_id,
            'char_count': self.char_count,
            'word_count': self.word_count,
            'unique_words': unique_words,
            'content_words': content_words,
            'pos_count': len(set(self.pos_tags)) if self.pos_tags else 0,
            'has_errors': bool(self.processing_errors),
            'error_count': len(self.processing_errors),
            'is_processed': self.is_processed,
            'language': self.language,
        }

    def is_valid(self) -> bool:
        return bool(self.__raw and self.__raw.strip()) and self.char_count > 0 and self.word_count > 0

    def is_processable(self) -> bool:
        return self.is_valid() and self.word_count >= 2 and self.char_count >= 5

    def get_text_preview(self, max_chars: int = 50) -> str:
        if not self.__raw:
            return ""

        preview = self.__raw.strip()
        if len(preview) <= max_chars:
            return preview

        return preview[:max_chars] + "..."

    def copy(self) -> 'Sentence':
        new_sentence = Sentence(
            raw=self.__raw,
            doc_id=self.doc_id,
            sentence_id=f"{self.sentence_id}_copy",
            docs_ref=self.docs_ref,
        )
        new_sentence.lemmatised = self.lemmatised.copy()
        new_sentence.word_objects = self.word_objects.copy()
        new_sentence.word_indices = self.word_indices.copy()
        new_sentence.language = self.language
        new_sentence.pos_tags = self.pos_tags.copy()
        new_sentence.char_count = self.char_count
        new_sentence.word_count = self.word_count
        new_sentence.processing_errors = self.processing_errors.copy()
        new_sentence.is_processed = self.is_processed
        return new_sentence

    @property
    def _raw(self) -> str:
        return self.__raw

    @_raw.setter
    def _raw(self, value: str):
        self.raw = value

    @property
    def _lemmatised(self) -> List[str]:
        return self.lemmatised

    @_lemmatised.setter
    def _lemmatised(self, value: List[str]):
        self.lemmatised = value

    @property
    def _word_objects(self) -> List[Word]:
        return self.word_objects

    @_word_objects.setter
    def _word_objects(self, value: List[Word]):
        self.word_objects = value

    @property
    def _word_indices(self) -> List[int]:
        return self.word_indices

    @_word_indices.setter
    def _word_indices(self, value: List[int]):
        self.word_indices = value

    def __str__(self) -> str:
        return self.get_text_preview(100)

    def __repr__(self) -> str:
        return f"Sentence(id='{self.sentence_id}', words={self.word_count}, processed={self.is_processed})"

    def __len__(self) -> int:
        return self.word_count

    def __eq__(self, other) -> bool:
        if not isinstance(other, Sentence):
            return False
        return self.sentence_id == other.sentence_id

    def __hash__(self) -> int:
        return hash(self.sentence_id)
