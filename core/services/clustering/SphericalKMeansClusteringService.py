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

    def __init__(self, random_state: int = 42):
        super().__init__(random_state)
        # NumPy random generator 생성 (재현성 보장)
        self.rng = np.random.default_rng(random_state)

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
        max_clusters: int = 10,
        n_init: int = 10,
        method: str = 'kneedle'
    ) -> Tuple[np.ndarray, int, List[float], List[float]]:
        """
        최적 클러스터 수 탐색 후 클러스터링

        Args:
            embeddings: 노드 임베딩 (Tensor)
            min_clusters: 최소 클러스터 수
            max_clusters: 최대 클러스터 수
            n_init: 초기화 횟수
            method: 'kneedle' (Kneedle 알고리즘, 기본값),
                   'elbow' (2차 미분), 'gap' (Gap Statistics)

        Returns:
            (cluster_labels, best_k, inertias, silhouette_scores)
        """
        if method == 'kneedle':
            return self._auto_clustering_with_kneedle(
                embeddings,
                min_clusters=min_clusters,
                max_clusters=max_clusters,
                n_init=n_init
            )
        elif method == 'gap':
            return self._auto_clustering_with_gap(
                embeddings,
                min_clusters=min_clusters,
                max_clusters=max_clusters,
                n_init=n_init
            )
        else:  # elbow
            return self._auto_clustering_with_elbow(
                embeddings,
                min_clusters=min_clusters,
                max_clusters=max_clusters,
                n_init=n_init
            )

    def _auto_clustering_with_kneedle(
        self,
        embeddings: torch.Tensor,
        min_clusters: int = 3,
        max_clusters: int = 10,
        n_init: int = 3
    ) -> Tuple[np.ndarray, int, List[float], List[float]]:
        """
        Kneedle 알고리즘으로 최적 클러스터 수 탐색 후 클러스터링

        Kneedle은 Elbow Method의 개선 버전으로 자동으로 knee point를 탐지합니다.
        2차 미분보다 robust하고 노이즈에 덜 민감합니다.

        Args:
            embeddings: 노드 임베딩 (Tensor)
            min_clusters: 최소 클러스터 수
            max_clusters: 최대 클러스터 수
            n_init: 초기화 횟수

        Returns:
            (cluster_labels, best_k, inertias, silhouette_scores)
        """
        try:
            from kneed import KneeLocator
        except ImportError:
            print("Warning: kneed 라이브러리가 설치되어 있지 않습니다.")
            print("pip install kneed 로 설치하거나 method='elbow'를 사용하세요.")
            print("Fallback to Elbow Method...")
            return self._auto_clustering_with_elbow(embeddings, min_clusters, max_clusters, n_init)

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

        # Kneedle 알고리즘으로 knee point 찾기
        kneedle = KneeLocator(
            list(k_range),
            self.inertias,
            curve='convex',     # Inertia는 convex curve
            direction='decreasing',  # Inertia는 감소
            online=False,
            S=1.0  # Sensitivity (기본값 1.0, 높을수록 민감)
        )

        if kneedle.knee is not None:
            self.best_k = kneedle.knee
        else:
            # Knee가 탐지되지 않으면 Silhouette score가 가장 높은 k 선택
            print("Warning: Kneedle이 knee point를 찾지 못했습니다.")
            print("         Silhouette score가 가장 높은 k를 선택합니다.")
            self.best_k = k_range[np.argmax(self.silhouette_scores)]

        # 최적 k로 최종 클러스터링
        self.cluster_labels = self.fit_predict(
            torch.from_numpy(embeddings_np) if isinstance(embeddings_np, np.ndarray) else embeddings,
            self.best_k,
            n_init
        )

        return self.cluster_labels, self.best_k, self.inertias, self.silhouette_scores

    def _auto_clustering_with_gap(
        self,
        embeddings: torch.Tensor,
        min_clusters: int = 3,
        max_clusters: int = 10,
        n_init: int = 3,
        n_iterations: int = 10
    ) -> Tuple[np.ndarray, int, List[float], List[float]]:
        """
        Gap Statistics로 최적 클러스터 수 탐색 후 클러스터링

        Args:
            embeddings: 노드 임베딩 (Tensor)
            min_clusters: 최소 클러스터 수 (강건성을 위한 제약)
            max_clusters: 최대 클러스터 수
            n_init: 초기화 횟수
            n_iterations: Bootstrap 반복 횟수

        Returns:
            (cluster_labels, best_k, [], [])
            Note: Gap Statistics는 inertias, silhouette_scores를 반환하지 않음
        """
        try:
            from gapstatistics.gapstatistics import GapStatistics
        except ImportError:
            print("Warning: gapstatistics 라이브러리가 설치되어 있지 않습니다.")
            print("pip install gapstatistics 로 설치하거나 method='elbow'를 사용하세요.")
            print("Fallback to Elbow Method...")
            return self._auto_clustering_with_elbow(embeddings, min_clusters, max_clusters, n_init)

        embeddings_np = embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings
        embeddings_normalized = self._normalize(embeddings_np)

        # Gap Statistics 적용
        gs = GapStatistics(
            algorithm=self._SphericalKMeansWrapper,
            distance_metric=self._cosine_distance_for_gap,
            return_params=True
        )

        self.best_k, gap_params = gs.fit_predict(
            K=max_clusters,
            X=embeddings_normalized,
            n_iterations=n_iterations
        )

        # 강건성을 위한 제약: min_clusters 미만이면 min_clusters로 설정
        if self.best_k < min_clusters:
            print(f"Warning: Gap Statistics가 k={self.best_k}를 제안했지만, ")
            print(f"         min_clusters={min_clusters} 제약으로 인해 k={min_clusters}로 설정합니다.")
            self.best_k = min_clusters

        # 최적 k로 최종 클러스터링
        self.cluster_labels = self.fit_predict(
            torch.from_numpy(embeddings_np) if isinstance(embeddings_np, np.ndarray) else embeddings,
            self.best_k,
            n_init
        )

        # Gap Statistics는 inertias, silhouette_scores를 제공하지 않으므로 빈 리스트 반환
        self.inertias = []
        self.silhouette_scores = []

        return self.cluster_labels, self.best_k, self.inertias, self.silhouette_scores

    def _auto_clustering_with_elbow(
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
                    new_centroids[k] = embeddings[self.rng.integers(n_samples)]

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
        first_idx = self.rng.integers(n_samples)
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
            next_idx = self.rng.choice(n_samples, p=probabilities)
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

    @staticmethod
    def _cosine_distance_for_gap(X: np.ndarray, Centroid: np.ndarray) -> np.ndarray:
        """
        Gap Statistics를 위한 코사인 거리 계산 함수

        Args:
            X: 데이터 포인트들 (n_samples, n_features)
            Centroid: 중심점 (1, n_features) 또는 (n_features,)

        Returns:
            코사인 거리 배열 (n_samples,)
        """
        # Centroid 차원 조정
        if Centroid.ndim == 2:
            Centroid = Centroid.flatten()

        # 정규화
        X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
        Centroid_normalized = Centroid / np.linalg.norm(Centroid)

        # 코사인 유사도
        similarities = np.dot(X_normalized, Centroid_normalized)

        # 코사인 거리 = 1 - 코사인 유사도
        distances = 1 - similarities

        return distances

    class _SphericalKMeansWrapper:
        """
        Gap Statistics를 위한 Spherical K-Means 래퍼 클래스

        gapstatistics 라이브러리는 sklearn 스타일의 인터페이스를 요구:
        - __init__(n_clusters)
        - fit(X) 메서드
        - predict(X) 메서드
        - cluster_centers_ 속성
        """

        def __init__(self, n_clusters, random_state=42, n_init=3):
            self.n_clusters = n_clusters
            self.random_state = random_state
            self.n_init = n_init
            self.cluster_centers_ = None
            self.labels_ = None
            self._service = None

        def fit(self, X):
            """클러스터링 학습"""
            from core.services.clustering.SphericalKMeansClusteringService import SphericalKMeansClusteringService

            self._service = SphericalKMeansClusteringService(random_state=self.random_state)
            X_tensor = torch.from_numpy(X).float()

            # 클러스터링 수행
            self.labels_ = self._service.fit_predict(
                X_tensor,
                n_clusters=self.n_clusters,
                n_init=self.n_init
            )

            # 클러스터 중심 계산 (정규화된 평균)
            X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
            self.cluster_centers_ = np.zeros((self.n_clusters, X.shape[1]))

            for k in range(self.n_clusters):
                cluster_mask = (self.labels_ == k)
                if np.sum(cluster_mask) > 0:
                    cluster_mean = np.mean(X_normalized[cluster_mask], axis=0)
                    # 중심점 정규화
                    self.cluster_centers_[k] = cluster_mean / np.linalg.norm(cluster_mean)

            return self

        def predict(self, X):
            """새로운 데이터에 대한 클러스터 예측"""
            if self.cluster_centers_ is None:
                raise RuntimeError("fit()을 먼저 호출해야 합니다.")

            # 정규화
            X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)

            # 코사인 유사도 기반 예측
            similarities = np.dot(X_normalized, self.cluster_centers_.T)
            return np.argmax(similarities, axis=1)
