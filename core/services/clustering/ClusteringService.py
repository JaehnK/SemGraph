"""
클러스터링 서비스 (Backward Compatibility Wrapper)

기존 코드와의 호환성을 위한 래퍼 클래스.
내부적으로 KMeansClusteringService를 사용.
"""

import numpy as np
import torch
from typing import Optional, Tuple, List, Dict

from .KMeansClusteringService import KMeansClusteringService


class ClusteringService:
    """
    클러스터링 서비스 (Backward Compatibility Wrapper)

    기존 코드와의 호환성을 위해 KMeansClusteringService를 래핑.
    새로운 코드에서는 KMeansClusteringService 또는 SphericalKMeansClusteringService를 직접 사용 권장.
    """

    def __init__(self, random_state: int = 42):
        """
        Args:
            random_state: 재현성을 위한 랜덤 시드
        """
        self._impl = KMeansClusteringService(random_state=random_state)
        self.random_state = random_state

    @property
    def cluster_labels(self) -> Optional[np.ndarray]:
        return self._impl.cluster_labels

    @cluster_labels.setter
    def cluster_labels(self, value: Optional[np.ndarray]):
        self._impl.cluster_labels = value

    @property
    def inertias(self) -> Optional[List[float]]:
        return self._impl.inertias

    @inertias.setter
    def inertias(self, value: Optional[List[float]]):
        self._impl.inertias = value

    @property
    def silhouette_scores(self) -> Optional[List[float]]:
        return self._impl.silhouette_scores

    @silhouette_scores.setter
    def silhouette_scores(self, value: Optional[List[float]]):
        self._impl.silhouette_scores = value

    @property
    def best_k(self) -> Optional[int]:
        return self._impl.best_k

    @best_k.setter
    def best_k(self, value: Optional[int]):
        self._impl.best_k = value

    def kmeans_clustering(
        self,
        embeddings: torch.Tensor,
        n_clusters: int,
        n_init: int = 10
    ) -> np.ndarray:
        """
        K-means 클러스터링 수행 (backward compatibility method)

        Args:
            embeddings: 노드 임베딩 (Tensor)
            n_clusters: 클러스터 수
            n_init: 초기화 횟수

        Returns:
            클러스터 라벨 배열
        """
        return self._impl.fit_predict(embeddings, n_clusters, n_init)

    def auto_clustering_elbow(
        self,
        embeddings: torch.Tensor,
        min_clusters: int = 3,
        max_clusters: int = 20,
        n_init: int = 10
    ) -> Tuple[np.ndarray, int, List[float], List[float]]:
        """
        Elbow Method로 최적 클러스터 수 탐색 후 클러스터링 (backward compatibility method)

        Args:
            embeddings: 노드 임베딩 (Tensor)
            min_clusters: 최소 클러스터 수
            max_clusters: 최대 클러스터 수
            n_init: 초기화 횟수

        Returns:
            (cluster_labels, best_k, inertias, silhouette_scores)
        """
        return self._impl.auto_clustering(embeddings, min_clusters, max_clusters, n_init)

    def get_cluster_distribution(self) -> Dict[int, int]:
        """
        클러스터별 데이터 포인트 수 반환

        Returns:
            {cluster_id: count} 딕셔너리
        """
        return self._impl.get_cluster_distribution()

    def save_elbow_curve(
        self,
        k_values: List[int],
        output_path: str
    ) -> None:
        """
        Elbow curve 시각화 저장

        Args:
            k_values: 클러스터 수 리스트
            output_path: 저장 경로
        """
        return self._impl.save_elbow_curve(k_values, output_path)
