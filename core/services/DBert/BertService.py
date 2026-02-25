import torch
import numpy as np
from typing import List, Dict, Any
from transformers import DistilBertTokenizer, DistilBertModel
import sys
import os
from pathlib import Path

# 상대 임포트와 절대 임포트 모두 지원
try:
    from ..Document.DocumentService import DocumentService
except ImportError:
    # 절대 경로로 임포트 시도
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from services.Document.DocumentService import DocumentService


class BertService:
    """
    BERT 서비스 - 파이프라인 관리
    """

    def __init__(self, docs: DocumentService, model_path: str | None = None):

        self.docs = docs
        root_dir = Path(__file__).resolve().parents[3]
        default_model_path = root_dir / "DistillBERT" / "model"
        self.model_path = str(default_model_path if model_path is None else Path(model_path))
        self.cache_path = self.model_path
        self.model_name = 'distilbert-base-uncased'

        self.tokenizer = DistilBertTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_path
        )

        self.model = DistilBertModel.from_pretrained(
            self.model_name,
            cache_dir=self.cache_path
        )

        self.model.eval()

    def get_token_embeddings(self, text: str) -> List[Dict[str, Any]]:
        """
        입력 텍스트를 토큰 단위로 분해하고 각 토큰의 임베딩을 반환
        """

        # 1. 토크나이징 (uncased로 자동 소문자 변환)
        encoded = self.tokenizer(
            text,
            return_tensors='pt',
            add_special_tokens=True,  # [CLS], [SEP] 토큰 추가
            padding=True,
            truncation=True,
            max_length=512,
            return_attention_mask=True
        )

        # 2. 토큰 ID를 실제 토큰으로 변환
        input_ids = encoded['input_ids'][0]  # 배치 차원 제거
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        # 3. 모델 추론 (임베딩 추출)
        with torch.no_grad():
            outputs = self.model(**encoded)
            # last_hidden_state: [batch_size, sequence_length, hidden_size]
            token_embeddings = outputs.last_hidden_state[0]  # 첫 번째 배치 선택

        # 4. 토큰과 임베딩을 함께 반환
        token_results = []
        for i, (token, embedding) in enumerate(zip(tokens, token_embeddings)):
            token_results.append({
                'position': i,
                'token': token,
                'embedding': embedding.cpu().numpy(),  # tensor를 numpy로 변환
                'embedding_shape': embedding.shape
            })

        return token_results

    def get_combined_embeddings(self, text: str) -> List[Dict[str, Any]]:
        """
        텍스트의 토큰 임베딩을 추출하고 서브워드를 결합하여 반환
        """
        token_results = self.get_token_embeddings(text)
        return self.combine_subword_embeddings(token_results)

    @staticmethod
    def combine_subword_embeddings(token_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        서브워드 토큰들을 원래 단어로 합치고 그 임베딩을 평균합니다.
        예: 'cr', '##inge' -> 'cringe' (평균 임베딩)
        """
        combined_results = []
        current_word_tokens = []
        current_word_embeddings_list = []

        for item in token_results:
            token = item['token']
            embedding = item['embedding']

            if token.startswith('##'):
                # 이전 토큰의 서브워드인 경우
                current_word_tokens.append(token[2:])  # '##' 제거 후 추가
                current_word_embeddings_list.append(embedding)
            else:
                # 새로운 단어의 시작 또는 특수 토큰인 경우
                if current_word_tokens:
                    # 이전에 누적된 서브워드가 있다면 합치고 임베딩 평균 계산
                    combined_token = "".join(current_word_tokens)
                    averaged_embedding = np.mean(current_word_embeddings_list, axis=0)
                    combined_results.append({
                        'token': combined_token,
                        'embedding': averaged_embedding,
                        'embedding_shape': averaged_embedding.shape  # 평균 후 shape
                    })
                    current_word_tokens = []
                    current_word_embeddings_list = []

                # 현재 토큰 처리
                if token in ['[CLS]', '[SEP]']:
                    combined_results.append({
                        'token': token,
                        'embedding': embedding,
                        'embedding_shape': embedding.shape
                    })
                else:
                    # 새로운 단어 (또는 서브워드의 첫 부분) 누적 시작
                    current_word_tokens.append(token)
                    current_word_embeddings_list.append(embedding)

        # 루프 종료 후 남아있는 서브워드 처리 (예: 문장 끝이 서브워드로 끝나는 경우)
        if current_word_tokens:
            combined_token = "".join(current_word_tokens)
            averaged_embedding = np.mean(current_word_embeddings_list, axis=0)
            combined_results.append({
                'token': combined_token,
                'embedding': averaged_embedding,
                'embedding_shape': averaged_embedding.shape
            })

        # 최종 결과에 position 재할당
        for i, item in enumerate(combined_results):
            item['position'] = i

        return combined_results

    def get_sentence_embedding(self, text: str) -> np.ndarray:
        """
        문장 전체의 임베딩을 반환 ([CLS] 토큰 사용)
        """
        token_results = self.get_token_embeddings(text)
        # [CLS] 토큰의 임베딩을 문장 임베딩으로 사용
        for token_info in token_results:
            if token_info['token'] == '[CLS]':
                return token_info['embedding']

        # [CLS] 토큰을 찾지 못한 경우 첫 번째 토큰 반환
        return token_results[0]['embedding'] if token_results else None

    def get_word_embedding(self, word: str) -> np.ndarray:
        """
        단일 단어의 BERT 임베딩을 반환
        GraphMAE에서 사용하기 위한 메소드
        """
        combined_results = self.get_combined_embeddings(word)

        # 특수 토큰이 아닌 첫 번째 토큰의 임베딩 반환
        for result in combined_results:
            if result['token'] not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']:
                return result['embedding']

        # 적절한 토큰을 찾지 못한 경우 [CLS] 토큰 사용
        return combined_results[0]['embedding'] if combined_results else np.zeros(768)

    def update_at_wordtrie(self, token_result:list):
        # 특수 토큰들 필터링
        special_tokens = {'[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]'}

        for result in token_result:
            token = result['token']
            # 특수 토큰은 제외하고 업데이트
            if token not in special_tokens:
                self.docs._word_service.update_word_bert_embedding(
                    word_content=token,
                    embedding=result['embedding']
                    )


    def train(self):
        # 여기서 독스에서 문장을 받아와서 한문장씩 get_sentence_embedding 실시
        # 실시한 결과를 받아와서 독스의 트라이에 업데이트 진행
        # 업데이트 로직은 가중치 연산을 수행함
        sentences = self.docs.Sentence_list
        count = 0
        for sentence in sentences:
            count += 1
            embed = self.get_sentence_embedding(sentence)
            token_results = self.combine_subword_embeddings(embed)
            self.update_at_wordtrie(token_results)

            if count % 10 == 0:
                print(f"BERT Embedding {count} completed")

        return None
