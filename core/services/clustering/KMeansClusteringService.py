"""
K-Means 클러스터링 서비스

전통적인 K-means 알고리즘 구현
"""

import numpy as np
import torch
from typing import Tuple, List
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .ClusteringInterface import ClusteringInterface


class KMeansClusteringService(ClusteringInterface):
    """K-Means 클러스터링 서비스"""

    def fit_predict(
        self,
        embeddings: torch.Tensor,
        n_clusters: int,
        n_init: int = 10
    ) -> np.ndarray:
        """
        K-means 클러스터링 수행

        Args:
            embeddings: 노드 임베딩 (Tensor)
            n_clusters: 클러스터 수
            n_init: 초기화 횟수

        Returns:
            클러스터 라벨 배열
        """
        embeddings_np = embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings

        kmeans = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            random_state=self.random_state,
            n_init=n_init
        )
        self.cluster_labels = kmeans.fit_predict(embeddings_np)

        return self.cluster_labels

    def auto_clustering(
        self,
        embeddings: torch.Tensor,
        min_clusters: int = 3,
        max_clusters: int = 20,
        n_init: int = 10
    ) -> Tuple[np.ndarray, int, List[float], List[float]]:
        """
        Elbow Method로 최적 클러스터 수 탐색 후 클러스터링

        Args:
            embeddings: 노드 임베딩 (Tensor)
            min_clusters: 최소 클러스터 수
            max_clusters: 최대 클러스터 수
            n_init: 초기화 횟수

        Returns:
            (cluster_labels, best_k, inertias, silhouette_scores)
        """
        embeddings_np = embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings

        self.inertias = []
        self.silhouette_scores = []
        k_range = range(min_clusters, max_clusters + 1)

        # 각 k에 대해 클러스터링 수행
        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=self.random_state, n_init=n_init)
            labels = kmeans.fit_predict(embeddings_np)
            self.inertias.append(kmeans.inertia_)
            self.silhouette_scores.append(silhouette_score(embeddings_np, labels))

        # Elbow point 찾기
        self.best_k = self._find_elbow_point(list(k_range), self.inertias)

        # 최적 k로 최종 클러스터링
        kmeans = KMeans(n_clusters=self.best_k, init='k-means++', random_state=self.random_state, n_init=n_init)
        self.cluster_labels = kmeans.fit_predict(embeddings_np)

        return self.cluster_labels, self.best_k, self.inertias, self.silhouette_scores
