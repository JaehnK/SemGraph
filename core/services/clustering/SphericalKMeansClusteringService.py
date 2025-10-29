"""
Spherical K-Means 클러스터링 서비스

구 표면에서의 거리(코사인 유사도)를 사용하는 K-means 알고리즘
텍스트 임베딩과 같이 정규화된 벡터에 적합
"""

import numpy as np
import torch
from typing import Tuple, List
from sklearn.metrics import silhouette_score

from .ClusteringInterface import ClusteringInterface


class SphericalKMeansClusteringService(ClusteringInterface):
    """Spherical K-Means 클러스터링 서비스

    코사인 유사도 기반의 K-means 변형.
    중심점과 데이터 포인트를 모두 정규화하여 구 표면에서 클러스터링 수행.
    """

    def fit_predict(
        self,
        embeddings: torch.Tensor,
        n_clusters: int,
        n_init: int = 10
    ) -> np.ndarray:
        """
        Spherical K-means 클러스터링 수행

        Args:
            embeddings: 노드 임베딩 (Tensor)
            n_clusters: 클러스터 수
            n_init: 초기화 횟수

        Returns:
            클러스터 라벨 배열
        """
        embeddings_np = embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings

        # 임베딩 정규화
        embeddings_normalized = self._normalize(embeddings_np)

        best_labels = None
        best_inertia = float('inf')

        # 여러 번 초기화하여 최적 결과 선택
        for _ in range(n_init):
            labels, inertia = self._spherical_kmeans(embeddings_normalized, n_clusters)
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels

        self.cluster_labels = best_labels
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
        embeddings_normalized = self._normalize(embeddings_np)

        self.inertias = []
        self.silhouette_scores = []
        k_range = range(min_clusters, max_clusters + 1)

        # 각 k에 대해 클러스터링 수행
        for k in k_range:
            best_labels = None
            best_inertia = float('inf')

            for _ in range(n_init):
                labels, inertia = self._spherical_kmeans(embeddings_normalized, k)
                if inertia < best_inertia:
                    best_inertia = inertia
                    best_labels = labels

            self.inertias.append(best_inertia)

            # 코사인 거리 기반 실루엣 스코어 계산
            silhouette = self._cosine_silhouette_score(embeddings_normalized, best_labels)
            self.silhouette_scores.append(silhouette)

        # Elbow point 찾기
        self.best_k = self._find_elbow_point(list(k_range), self.inertias)

        # 최적 k로 최종 클러스터링
        self.cluster_labels = self.fit_predict(
            torch.from_numpy(embeddings_np) if isinstance(embeddings_np, np.ndarray) else embeddings,
            self.best_k,
            n_init
        )

        return self.cluster_labels, self.best_k, self.inertias, self.silhouette_scores

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        벡터 정규화 (L2 norm)

        Args:
            embeddings: 임베딩 배열

        Returns:
            정규화된 임베딩
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # 0으로 나누기 방지
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms

    def _spherical_kmeans(
        self,
        embeddings: np.ndarray,
        n_clusters: int,
        max_iter: int = 300,
        tol: float = 1e-4
    ) -> Tuple[np.ndarray, float]:
        """
        Spherical K-means 알고리즘 구현

        Args:
            embeddings: 정규화된 임베딩 배열
            n_clusters: 클러스터 수
            max_iter: 최대 반복 횟수
            tol: 수렴 판정 임계값

        Returns:
            (labels, inertia)
        """
        n_samples = embeddings.shape[0]

        # K-means++ 초기화 (코사인 거리 기반)
        centroids = self._kmeans_plusplus_init(embeddings, n_clusters)

        labels = None
        prev_inertia = float('inf')

        for iteration in range(max_iter):
            # 1. 할당 단계: 각 포인트를 가장 가까운 중심점에 할당
            # 코사인 유사도 계산 (정규화된 벡터의 내적)
            similarities = np.dot(embeddings, centroids.T)
            labels = np.argmax(similarities, axis=1)

            # 2. 갱신 단계: 중심점을 클러스터의 평균으로 업데이트
            new_centroids = np.zeros_like(centroids)
            for k in range(n_clusters):
                cluster_mask = (labels == k)
                if np.sum(cluster_mask) > 0:
                    # 클러스터 내 벡터들의 평균
                    cluster_mean = np.mean(embeddings[cluster_mask], axis=0)
                    new_centroids[k] = cluster_mean
                else:
                    # 빈 클러스터의 경우 랜덤하게 재초기화
                    new_centroids[k] = embeddings[np.random.randint(n_samples)]

            # 중심점 정규화
            centroids = self._normalize(new_centroids)

            # 3. 수렴 확인 (inertia 계산)
            # Inertia: 1 - cosine similarity의 합 (코사인 거리의 합)
            inertia = 0.0
            for k in range(n_clusters):
                cluster_mask = (labels == k)
                if np.sum(cluster_mask) > 0:
                    # 코사인 거리 = 1 - 코사인 유사도
                    similarities_k = np.dot(embeddings[cluster_mask], centroids[k])
                    distances = 1 - similarities_k
                    inertia += np.sum(distances)

            # 수렴 체크
            if abs(prev_inertia - inertia) < tol:
                break

            prev_inertia = inertia

        return labels, inertia

    def _kmeans_plusplus_init(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        K-means++ 초기화 (코사인 거리 기반)

        코사인 거리가 먼 점들을 우선적으로 중심점으로 선택

        Args:
            embeddings: 정규화된 임베딩 배열
            n_clusters: 클러스터 수

        Returns:
            초기 중심점 배열
        """
        n_samples = embeddings.shape[0]
        centroids = np.zeros((n_clusters, embeddings.shape[1]))

        # 첫 번째 중심점: 랜덤하게 선택
        np.random.seed(self.random_state)
        first_idx = np.random.randint(n_samples)
        centroids[0] = embeddings[first_idx]

        # 나머지 중심점들을 K-means++ 방식으로 선택
        for k in range(1, n_clusters):
            # 각 점에서 가장 가까운 중심점까지의 코사인 거리 계산
            # 코사인 유사도 계산
            similarities = np.dot(embeddings, centroids[:k].T)  # (n_samples, k)
            max_similarities = np.max(similarities, axis=1)  # 가장 높은 유사도

            # 코사인 거리 = 1 - 코사인 유사도
            distances = 1 - max_similarities

            # 거리가 먼 점일수록 높은 확률로 선택
            # 거리 제곱을 확률로 사용 (원래 K-means++의 방식)
            probabilities = distances ** 2
            probabilities /= probabilities.sum()

            # 확률에 따라 다음 중심점 선택
            next_idx = np.random.choice(n_samples, p=probabilities)
            centroids[k] = embeddings[next_idx]

        # 중심점 정규화
        return self._normalize(centroids)

    def _cosine_silhouette_score(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """
        코사인 거리 기반 실루엣 스코어 계산

        Args:
            embeddings: 정규화된 임베딩 배열
            labels: 클러스터 라벨

        Returns:
            실루엣 스코어
        """
        # sklearn의 silhouette_score는 metric='cosine' 지원
        return silhouette_score(embeddings, labels, metric='cosine')
