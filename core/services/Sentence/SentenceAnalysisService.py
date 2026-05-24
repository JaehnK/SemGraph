from typing import List, Optional, Any, Dict
from entities import Sentence

class SentenceAnalysisService:
    """문장 분석 서비스 - 정적 메서드들"""
    
    @staticmethod
    def expand_contractions(text: str) -> str:
        """축약형을 확장하는 함수"""
        try:
            import contractions
            expanded_text = contractions.fix(text)
            return expanded_text.lower()
        except Exception as e:
            print(f"Contraction expansion failed for text: {text[:50]}... Error: {e}")
            return text.lower()
    
    @staticmethod
    def is_valid_token(text: str) -> bool:
        """토큰 유효성 검사"""
        import re
        return bool(re.match(r'^[\w가-힣\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002700-\U000027BF\U0001F900-\U0001F9FF]+$', text))
    
    @staticmethod
    def process_with_spacy(
        sentence: Sentence,
        spacy_doc,
        docs_ref,
        original_text: Optional[str] = None,
    ) -> None:
        """spaCy 처리 (외부에서 호출용)"""
        sentence._raw = original_text if original_text is not None else sentence.raw

        try:
            lemmatised_words = []
            word_objects = []
            word_indices = []
            pos_tags = []

            for token in spacy_doc:
                if (
                    token.is_punct
                    or token.is_space
                    or len(token.text.strip()) < 1
                    or not SentenceAnalysisService.is_valid_token(str(token))
                ):
                    continue

                lemma = token.lemma_.lower()
                pos_tag = SentenceAnalysisService.convert_spacy_pos_to_nltk(token.pos_, token.tag_)

                lemmatised_words.append(lemma)
                pos_tags.append(pos_tag)

                if docs_ref is not None:
                    word_obj = docs_ref.add_word(lemma, pos_tag)
                    if not word_obj.stopword_checked:
                        word_obj.set_stopword_status(token.is_stop)
                    word_objects.append(word_obj)
                    word_indices.append(word_obj.idx)

            sentence.set_processed_data(lemmatised_words, word_objects, word_indices, pos_tags)
        except Exception as e:
            sentence.add_processing_error(f"spaCy processing failed: {e}")
            print(f"Error processing sentence: {sentence.get_text_preview()}... Error: {e}")
    
    @staticmethod
    def process_with_fallback(sentence: Sentence, docs_ref) -> None:
        """폴백 처리 (외부에서 호출용)"""
        try:
            import re

            expanded_text = SentenceAnalysisService.expand_contractions(sentence.raw)
            cleaned_text = re.sub(r'[^\w\s]', '', expanded_text.lower())
            tokens = cleaned_text.split()
            lemmatised_words = [
                token for token in tokens
                if len(token) >= 2 or token in ['i', 'a']
            ]

            word_objects = []
            word_indices = []

            if docs_ref is not None:
                for word in lemmatised_words:
                    word_obj = docs_ref.add_word(word, 'NN')
                    word_objects.append(word_obj)
                    word_indices.append(word_obj.idx)

            sentence.set_processed_data(lemmatised_words, word_objects, word_indices)
        except Exception as e:
            sentence.add_processing_error(f"Fallback processing failed: {e}")

    @staticmethod
    def convert_spacy_pos_to_nltk(spacy_pos: str, spacy_tag: str) -> str:
        """spaCy POS 태그를 NLTK 스타일로 변환"""
        pos_mapping = {
            'ADJ': 'JJ', 'ADP': 'IN', 'ADV': 'RB', 'AUX': 'VB',
            'CONJ': 'CC', 'CCONJ': 'CC', 'DET': 'DT', 'INTJ': 'UH',
            'NOUN': 'NN', 'NUM': 'CD', 'PART': 'RP', 'PRON': 'PRP',
            'PROPN': 'NNP', 'PUNCT': '.', 'SCONJ': 'IN', 'SYM': 'SYM',
            'VERB': 'VB', 'X': 'XX', 'SPACE': 'SP',
        }
        return pos_mapping.get(spacy_pos, spacy_tag)
    
    @staticmethod
    def get_sentence_complexity_score(sentence: Sentence) -> Dict[str, float]:
        """문장 복잡도 점수 계산"""
        if not sentence.is_processed:
            return {"error": "Sentence not processed"}
        
        avg_word_length = sum(len(word) for word in sentence.lemmatised) / len(sentence.lemmatised) if sentence.lemmatised else 0
        unique_word_ratio = len(set(sentence.lemmatised)) / len(sentence.lemmatised) if sentence.lemmatised else 0
        content_word_ratio = len(sentence.get_content_words()) / len(sentence.word_objects) if sentence.word_objects else 0
        
        return {
            "avg_word_length": avg_word_length,
            "unique_word_ratio": unique_word_ratio,
            "content_word_ratio": content_word_ratio,
            "sentence_length": sentence.word_count,
            "char_per_word": sentence.char_count / sentence.word_count if sentence.word_count > 0 else 0
        }
