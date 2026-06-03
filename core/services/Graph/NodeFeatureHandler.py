from typing import List, Optional, TYPE_CHECKING
import torch
import numpy as np
import random
from ..ports import EmbeddingProvider

if TYPE_CHECKING:
    from ..Document.DocumentService import DocumentService
    from entities import Word


class NodeFeatureHandler:
    """
    Handles BERT node features for graph nodes.
    """

    def __init__(
        self,
        docs: "DocumentService",
        min_count: int = 1,
        random_seed: int = 42,
        embedding_provider: Optional[EmbeddingProvider] = None
    ):
        # 재현성을 위한 랜덤 시드 고정
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

        self.documents = docs
        self.random_seed = random_seed
        self.embedding_provider = embedding_provider or self._create_default_embedding_provider(docs)

    def calculate_embeddings(self, words: List["Word"], method: str = 'bert', embed_size: int = 64,
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

        return self.embedding_provider.embed_terms(words, embed_size)

    def _create_default_embedding_provider(self, docs: "DocumentService") -> EmbeddingProvider:
        from ..adapters.bert_embedding_provider import BertEmbeddingProvider
        return BertEmbeddingProvider(docs, random_seed=self.random_seed)
