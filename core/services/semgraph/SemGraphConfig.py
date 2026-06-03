"""
SemGraph Configuration
"""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class SemGraphConfig:
    """SemGraph 파이프라인 설정"""

    # === 데이터 로딩 설정 ===
    csv_path: str
    num_documents: int = 1000
    text_column: str = 'body'

    # === 그래프 구축 설정 ===
    top_n_words: int = 500
    exclude_stopwords: bool = True
    max_sentence_length: int = -1  # -1은 제한 없음
    edge_top_k: int = -1  # 각 노드당 상위 k개 엣지만 유지 (-1은 전부 유지)
    edge_weight_threshold: float = 0.0  # 이 값 이상의 weight를 가진 엣지만 유지

    # === 임베딩 설정 ===
    embedding_method: Literal['bert'] = 'bert'
    embed_size: int = 64  # BERT 임베딩 후처리 target dimension

    # === GraphMAE 설정 ===
    graphmae_epochs: int = 100
    graphmae_lr: float = 0.001
    graphmae_weight_decay: float = 0.0
    graphmae_device: Optional[str] = None  # None이면 자동 (CUDA 우선)
    mask_rate: float = 0.75
    encoder_type: str = "gat"  # Supported: "gat", "tsgat", "gcn", "mlp", "linear"
    decoder_type: str = "gat"  # Supported: "gat", "tsgat", "gcn", "mlp", "linear"

    # === 클러스터링 설정 ===
    clustering_method: Literal['kmeans', 'dbscan', 'hierarchical'] = 'kmeans'
    num_clusters: Optional[int] = None  # None이면 자동 탐색
    min_clusters: int = 3
    max_clusters: int = 20
    post_pca_dim: Optional[int] = None  # GraphMAE 출력 후 PCA 차원 축소 (None이면 축소 안 함)

    # === 평가 설정 ===
    eval_metrics: list = field(default_factory=lambda: [
        'silhouette', 'davies_bouldin', 'calinski_harabasz', 'npmi'
    ])

    # === 출력 설정 ===
    save_results: bool = True
    output_dir: str = './semgraph_output'
    save_graph_viz: bool = True  # t-SNE, Word Cloud 등
    save_network_viz: bool = False  # 네트워크 그래프 (시간 소요)
    save_embeddings: bool = True

    # === 디버그 설정 ===
    verbose: bool = True
    log_interval: int = 10  # epoch마다 로그 출력 간격

    # === 재현성 설정 ===
    random_seed: int = 42  # 재현성을 위한 랜덤 시드

    @classmethod
    def create_default(cls, csv_path: str, num_documents: int = 1000) -> 'SemGraphConfig':
        """기본 설정으로 생성"""
        return cls(
            csv_path=csv_path,
            num_documents=num_documents
        )

    @classmethod
    def create_for_testing(cls, csv_path: str) -> 'SemGraphConfig':
        """테스트용 빠른 설정"""
        return cls(
            csv_path=csv_path,
            num_documents=100,
            top_n_words=100,
            graphmae_epochs=10,
            embed_size=32,
            save_results=False,
            save_graph_viz=False,
            save_embeddings=False
        )

    def validate(self) -> None:
        """설정값 검증"""
        if self.embedding_method != 'bert':
            raise ValueError("SemGraph node embeddings are BERT-only. Use embedding_method='bert'.")

        if self.embed_size <= 0:
            raise ValueError("embed_size must be a positive integer")

        if self.num_clusters is not None:
            if not (self.min_clusters <= self.num_clusters <= self.max_clusters):
                raise ValueError(
                    f"num_clusters({self.num_clusters})는 "
                    f"{self.min_clusters}와 {self.max_clusters} 사이여야 합니다."
                )

        if self.top_n_words < self.max_clusters:
            raise ValueError(
                f"top_n_words({self.top_n_words})는 "
                f"max_clusters({self.max_clusters})보다 커야 합니다."
            )

    def __post_init__(self):
        """초기화 후 검증"""
        self.validate()
