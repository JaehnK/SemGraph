from typing import List
import torch
import numpy as np
import random
from sklearn.decomposition import PCA
from ..Document.DocumentService import DocumentService
from ..DBert import BertService
# Word2VecService는 동적 import로 처리
from entities import Word


class NodeFeatureHandler:
    """
    Handles node features for graph nodes.
    """

    def __init__(self, docs: DocumentService, min_count: int = 1, random_seed: int = 42):
        # 재현성을 위한 랜덤 시드 고정
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

        self.documents = docs
        self.random_seed = random_seed
        # Word2VecService 동적 import 및 초기화
        from ..Word2Vec.Word2VecService import Word2VecService as W2VService
        self.w2v = W2VService.create_default(docs, min_count=min_count)
        self.dbert = BertService(docs)

    def calculate_embeddings(self, words: List[Word], method: str = 'concat', embed_size: int = 64) -> torch.Tensor:
        """
        단어 리스트에 대한 임베딩 계산

        Args:
            words: 임베딩을 계산할 단어들
            method: 임베딩 방법 ('concat', 'w2v', 'bert')
            embed_size: 임베딩 크기

        Returns:
            [num_words, embedding_dim] 형태의 텐서
        """
        if method not in ['concat', 'w2v', 'bert']:
            raise ValueError(f"Unsupported embedding method: {method}")

        if method == 'w2v':
            return self._get_w2v_embeddings(words, embed_size)
        elif method == 'bert':
            return self._get_bert_embeddings(words)
        elif method == 'concat':
            return self._get_concat_embeddings(words, embed_size)

    def _get_w2v_embeddings(self, words: List[Word], embed_size: int) -> torch.Tensor:
        """Word2Vec 임베딩 계산"""
        print(f"    [Word2Vec] {len(words)}개 단어에 대한 임베딩 계산 중 (target_dim={embed_size})...")
        self.w2v.train()
        embeddings = []
        for word in words:
            # Word2Vec 모델에서 임베딩 추출
            embedding = self.w2v.get_word_vector(word.content)
            if embedding is not None:
                # embed_size로 크기 조정 (필요시 패딩 또는 자르기)
                if len(embedding) > embed_size:
                    embedding = embedding[:embed_size]
                elif len(embedding) < embed_size:
                    # 패딩으로 크기 맞춤
                    padding = embed_size - len(embedding)
                    embedding = np.pad(embedding, (0, padding), mode='constant')
                embeddings.append(embedding)
            else:
                # 단어가 없으면 0 벡터
                embeddings.append(np.zeros(embed_size))
        result = torch.tensor(embeddings, dtype=torch.float32)
        print(f"    [Word2Vec] 완료: shape={result.shape}")
        return result

    def _get_bert_embeddings(self, words: List[Word]) -> torch.Tensor:
        """BERT 임베딩 계산"""
        print(f"    [BERT] {len(words)}개 단어에 대한 BERT 임베딩 계산 중...")
        embeddings = []
        for word in words:
            # WordTrie에서 BERT 임베딩 추출
            embedding = self.dbert.get_word_embedding(word.content)
            embeddings.append(embedding)
        result = torch.tensor(embeddings, dtype=torch.float32)
        print(f"    [BERT] 완료: shape={result.shape}")
        return result

    def _get_concat_embeddings(self, words: List[Word], embed_size: int) -> torch.Tensor:
        """Word2Vec + BERT 연결 임베딩 (차원 조정 후 concat)"""
        target_dim = embed_size // 2
        print(f"    [Concat] Word2Vec + BERT 멀티모달 임베딩 (각 {target_dim}차원)")

        # Word2Vec 임베딩 (target_dim 크기로 요청)
        w2v_embeddings = self._get_w2v_embeddings(words, target_dim)
        bert_embeddings = self._get_bert_embeddings(words)

        # BERT 차원 조정 (768 -> target_dim)
        print(f"    [Concat] BERT 차원 조정: {bert_embeddings.shape[1]} -> {target_dim}")
        bert_reduced = self._adjust_embedding_dimension(bert_embeddings, target_dim)

        # Word2Vec도 target_dim으로 조정 (혹시 모를 경우)
        w2v_reduced = self._adjust_embedding_dimension(w2v_embeddings, target_dim)

        # 조정된 임베딩들을 concat
        result = torch.cat([w2v_reduced, bert_reduced], dim=1)
        print(f"    [Concat] 최종 임베딩 shape: {result.shape}")
        return result

    def _adjust_embedding_dimension(self, embeddings: torch.Tensor, target_dim: int) -> torch.Tensor:
        """임베딩 차원을 target_dim으로 조정 (PCA 또는 패딩)"""
        current_dim = embeddings.shape[1]

        if current_dim == target_dim:
            return embeddings
        elif current_dim > target_dim:
            # 차원이 클 경우: PCA로 차원 축소 (정보 보존)
            embeddings_np = embeddings.cpu().numpy()
            pca = PCA(n_components=target_dim, random_state=self.random_seed)
            reduced_embeddings = pca.fit_transform(embeddings_np)
            return torch.tensor(reduced_embeddings, dtype=embeddings.dtype)
        else:
            # 차원이 작을 경우: 0으로 패딩
            num_samples = embeddings.shape[0]
            padding_size = target_dim - current_dim
            padding = torch.zeros(num_samples, padding_size, dtype=embeddings.dtype)
            return torch.cat([embeddings, padding], dim=1)