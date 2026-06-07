from typing import Any, Optional, Sequence

import numpy as np
import torch

from ..ports import EmbeddingProvider


class BertEmbeddingProvider(EmbeddingProvider):
    """Creates BERT term embeddings behind the EmbeddingProvider port."""

    def __init__(self, docs: Any, random_seed: int = 42, bert_service: Optional[Any] = None):
        self.docs = docs
        self.random_seed = random_seed
        self._bert_service = bert_service

    @property
    def bert_service(self):
        if self._bert_service is None:
            from ..DBert import BertService
            self._bert_service = BertService(self.docs)
        return self._bert_service

    def embed_terms(self, terms: Sequence[Any], target_dim: Optional[int]) -> torch.Tensor:
        print(f"    [BERT] {len(terms)}개 단어에 대한 BERT 임베딩 계산 중...")
        embeddings = []
        for term in terms:
            embedding = self.bert_service.get_word_embedding(self._term_content(term))
            embeddings.append(embedding)

        result = torch.tensor(embeddings, dtype=torch.float32)
        print(f"    [BERT] 원본 임베딩: shape={result.shape}")

        if target_dim is not None and result.shape[1] != target_dim:
            print(f"    [BERT] PCA 차원 축소: {result.shape[1]} -> {target_dim}")
            result = self._adjust_embedding_dimension(result, target_dim)
            print(f"    [BERT] 완료: shape={result.shape}")
        else:
            print(f"    [BERT] 완료 (PCA 없음): shape={result.shape}")

        return result

    def _adjust_embedding_dimension(self, embeddings: torch.Tensor, target_dim: int) -> torch.Tensor:
        current_dim = embeddings.shape[1]
        n_samples = embeddings.shape[0]

        if current_dim == target_dim:
            return embeddings
        if current_dim > target_dim:
            max_components = min(n_samples, current_dim)

            if target_dim > max_components:
                print(f"    [WARNING] PCA 불가 (target={target_dim} > max={max_components})")
                print(f"              Truncate to {max_components}d + padding to {target_dim}d")

                embeddings_np = embeddings.cpu().numpy()
                truncated = embeddings_np[:, :max_components]
                padding_size = target_dim - max_components
                padded = np.pad(truncated, ((0, 0), (0, padding_size)), mode="constant")
                return torch.tensor(padded, dtype=embeddings.dtype)

            embeddings_np = embeddings.cpu().numpy()
            from sklearn.decomposition import PCA
            pca = PCA(n_components=target_dim, random_state=self.random_seed, whiten=True)
            reduced_embeddings = pca.fit_transform(embeddings_np)
            return torch.tensor(reduced_embeddings, dtype=embeddings.dtype)

        num_samples = embeddings.shape[0]
        padding_size = target_dim - current_dim
        padding = torch.zeros(num_samples, padding_size, dtype=embeddings.dtype)
        return torch.cat([embeddings, padding], dim=1)

    @staticmethod
    def _term_content(term: Any) -> str:
        return getattr(term, "content", str(term))
