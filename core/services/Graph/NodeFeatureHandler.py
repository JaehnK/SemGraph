from typing import List
import torch
import numpy as np
import random
from sklearn.decomposition import PCA
from ..Document.DocumentService import DocumentService
from ..DBert import BertService
from entities import Word


class NodeFeatureHandler:
    """
    Handles BERT node features for graph nodes.
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
        self.dbert = BertService(docs)

    def calculate_embeddings(self, words: List[Word], method: str = 'bert', embed_size: int = 64,
                             **kwargs) -> torch.Tensor:
        """
        단어 리스트에 대한 BERT 임베딩 계산

        Args:
            words: 임베딩을 계산할 단어들
            method: 기존 호출부 호환용. 'bert'만 허용.
            embed_size: BERT 임베딩 후처리 target dimension

        Returns:
            [num_words, embedding_dim] 형태의 텐서
        """
        if method != 'bert':
            raise ValueError("SemGraph node features are BERT-only. Use method='bert'.")

        return self._get_bert_embeddings(words, embed_size)

    def _get_bert_embeddings(self, words: List[Word], embed_size: int = None) -> torch.Tensor:
        """BERT 임베딩 계산 (PCA 차원 축소 포함)

        Args:
            words: 단어 리스트
            embed_size: 목표 임베딩 차원 (None이면 원본 768d 유지)
        """
        print(f"    [BERT] {len(words)}개 단어에 대한 BERT 임베딩 계산 중...")
        embeddings = []
        for word in words:
            embedding = self.dbert.get_word_embedding(word.content)
            embeddings.append(embedding)
        result = torch.tensor(embeddings, dtype=torch.float32)
        print(f"    [BERT] 원본 임베딩: shape={result.shape}")

        # embed_size가 지정된 경우 PCA로 차원 축소
        if embed_size is not None and result.shape[1] != embed_size:
            print(f"    [BERT] PCA 차원 축소: {result.shape[1]} -> {embed_size}")
            result = self._adjust_embedding_dimension(result, embed_size)
            print(f"    [BERT] 완료: shape={result.shape}")
        else:
            print(f"    [BERT] 완료 (PCA 없음): shape={result.shape}")

        return result

    def _adjust_embedding_dimension(self, embeddings: torch.Tensor, target_dim: int) -> torch.Tensor:
        """임베딩 차원을 target_dim으로 조정 (PCA 또는 패딩)"""
        current_dim = embeddings.shape[1]
        n_samples = embeddings.shape[0]

        if current_dim == target_dim:
            return embeddings
        elif current_dim > target_dim:
            # 차원이 클 경우: PCA로 차원 축소 (정보 보존)
            # PCA 제약: n_components <= min(n_samples, n_features)
            max_components = min(n_samples, current_dim)

            if target_dim > max_components:
                # PCA 불가능: truncate 후 패딩
                print(f"    [WARNING] PCA 불가 (target={target_dim} > max={max_components})")
                print(f"              Truncate to {max_components}d + padding to {target_dim}d")

                embeddings_np = embeddings.cpu().numpy()
                # 1. Truncate
                truncated = embeddings_np[:, :max_components]
                # 2. Padding
                padding_size = target_dim - max_components
                padded = np.pad(truncated, ((0, 0), (0, padding_size)), mode='constant')
                return torch.tensor(padded, dtype=embeddings.dtype)
            else:
                # PCA 가능
                embeddings_np = embeddings.cpu().numpy()
                pca = PCA(n_components=target_dim, random_state=self.random_seed, whiten=True)
                reduced_embeddings = pca.fit_transform(embeddings_np)
                return torch.tensor(reduced_embeddings, dtype=embeddings.dtype)
        else:
            # 차원이 작을 경우: 0으로 패딩
            num_samples = embeddings.shape[0]
            padding_size = target_dim - current_dim
            padding = torch.zeros(num_samples, padding_size, dtype=embeddings.dtype)
            return torch.cat([embeddings, padding], dim=1)
