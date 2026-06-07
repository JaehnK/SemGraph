"""
전통적 그래프 클러스터링 서비스

그래프 구조 기반 클러스터링 알고리즘:
- Louvain: Modularity 최적화
- Leiden: Louvain 개선판 (더 정확한 커뮤니티 탐지)
- Girvan-Newman: Edge betweenness 기반

SemGraph와 동일한 WordGraph를 공유하여 공정한 비교를 보장합니다.
"""

import numpy as np
import networkx as nx
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import warnings

from entities import WordGraph


class TraditionalGraphClusteringService:
    """전통적 그래프 클러스터링 서비스
    
    SemGraph와 동일한 WordGraph를 사용하여 전통적인 그래프 커뮤니티 탐지 알고리즘 수행.
    GraphMAE 없이 그래프 구조만으로 클러스터링하여 SemGraph의 차별점을 입증.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Args:
            random_state: 재현성을 위한 랜덤 시드 (Louvain, Leiden에 사용)
        """
        self.random_state = random_state
        self.cluster_labels: Optional[np.ndarray] = None
        self.nx_graph: Optional[nx.Graph] = None
        self.clustering_method: Optional[str] = None
        self.num_clusters: Optional[int] = None
        
        # 알고리즘별 추가 정보
        self.modularity: Optional[float] = None
        self.communities: Optional[List[set]] = None
        
    # ============================================================
    # Louvain 클러스터링
    # ============================================================
    
    def louvain_clustering(
        self, 
        word_graph: WordGraph,
        resolution: float = 1.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Louvain 알고리즘으로 커뮤니티 탐지
        
        Louvain은 Modularity를 최적화하는 탐욕적 알고리즘입니다.
        - 장점: 빠른 속도, 대규모 그래프에 적합
        - 단점: 로컬 옵티멈에 빠질 수 있음
        
        Args:
            word_graph: SemGraph와 동일한 WordGraph 객체
            resolution: Modularity 해상도 (높을수록 작은 커뮤니티 생성)
            
        Returns:
            (cluster_labels, metrics_dict)
            - cluster_labels: [num_nodes] 노드별 클러스터 ID
            - metrics_dict: 평가 지표 딕셔너리
        """
        try:
            import community.community_louvain as community_louvain
        except ImportError:
            raise ImportError(
                "python-louvain package required. Install: pip install python-louvain"
            )
        
        # WordGraph를 NetworkX로 변환 (가중치 포함)
        self.nx_graph = word_graph.export_to_networkx(include_weights=True)
        
        # Louvain 알고리즘 실행
        partition = community_louvain.best_partition(
            self.nx_graph,
            resolution=resolution,
            random_state=self.random_state
        )
        
        # 노드 ID 순서대로 클러스터 라벨 배열 생성
        self.cluster_labels = np.array([partition[i] for i in range(word_graph.num_nodes)])
        
        # 커뮤니티 정보 저장
        self.communities = self._labels_to_communities(self.cluster_labels)
        self.num_clusters = len(self.communities)
        
        # Modularity 계산
        self.modularity = community_louvain.modularity(partition, self.nx_graph)
        
        self.clustering_method = "louvain"
        
        # 메트릭 수집
        metrics = {
            'method': 'louvain',
            'num_clusters': self.num_clusters,
            'modularity': self.modularity,
            'resolution': resolution,
            'num_nodes': word_graph.num_nodes,
            'num_edges': word_graph.num_edges
        }
        
        print(f"✅ Louvain 클러스터링 완료: {self.num_clusters}개 커뮤니티 (Modularity: {self.modularity:.4f})")
        
        return self.cluster_labels, metrics
    
    # ============================================================
    # Leiden 클러스터링
    # ============================================================
    
    def leiden_clustering(
        self,
        word_graph: WordGraph,
        resolution: float = 1.0,
        n_iterations: int = -1
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Leiden 알고리즘으로 커뮤니티 탐지
        
        Leiden은 Louvain의 개선 버전으로, 더 정확한 커뮤니티를 탐지합니다.
        - 장점: Louvain보다 품질 높은 커뮤니티, 연결 보장
        - 단점: 약간 느림
        
        Args:
            word_graph: SemGraph와 동일한 WordGraph 객체
            resolution: Modularity 해상도
            n_iterations: 최대 반복 횟수 (-1은 수렴까지)
            
        Returns:
            (cluster_labels, metrics_dict)
        """
        try:
            import leidenalg
            import igraph as ig
        except ImportError:
            raise ImportError(
                "leidenalg and igraph required. Install:\n"
                "  pip install leidenalg python-igraph"
            )
        
        # WordGraph를 NetworkX로 변환
        self.nx_graph = word_graph.export_to_networkx(include_weights=True)
        
        # NetworkX를 igraph로 변환 (Leiden은 igraph 사용)
        g_igraph = self._networkx_to_igraph(self.nx_graph)
        
        # Leiden 알고리즘 실행
        partition = leidenalg.find_partition(
            g_igraph,
            leidenalg.ModularityVertexPartition,
            n_iterations=n_iterations,
            seed=self.random_state,
            weights='weight' if 'weight' in g_igraph.es.attributes() else None
        )
        
        # 클러스터 라벨 추출
        self.cluster_labels = np.array(partition.membership)
        
        # 커뮤니티 정보
        self.communities = self._labels_to_communities(self.cluster_labels)
        self.num_clusters = len(self.communities)
        
        # Modularity 계산
        self.modularity = partition.modularity
        
        self.clustering_method = "leiden"
        
        metrics = {
            'method': 'leiden',
            'num_clusters': self.num_clusters,
            'modularity': self.modularity,
            'resolution': resolution,
            'num_nodes': word_graph.num_nodes,
            'num_edges': word_graph.num_edges,
            'quality': partition.quality()
        }
        
        print(f"✅ Leiden 클러스터링 완료: {self.num_clusters}개 커뮤니티 (Modularity: {self.modularity:.4f})")
        
        return self.cluster_labels, metrics
    
    # ============================================================
    # Girvan-Newman 클러스터링
    # ============================================================
    
    def girvan_newman_clustering(
        self,
        word_graph: WordGraph,
        num_clusters: int = None,
        verbose: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Girvan-Newman 알고리즘으로 커뮤니티 탐지 (최적화 버전)
        
        Edge betweenness를 기반으로 반복적으로 엣지를 제거하며 커뮤니티를 분할합니다.
        - 장점: 계층적 구조 파악 가능
        - 단점: 느린 속도 (O(m²n) 또는 O(n³)), 대규모 그래프에 부적합
        
        최적화 전략:
        1. num_clusters 지정: 조기 종료로 속도 향상 (권장!)
        2. 그래프 크기 제한: ≤150 노드 권장
        3. 진행상황 표시: verbose=True
        
        병렬 처리의 한계:
        - Edge betweenness 계산은 병렬화 가능하지만 NetworkX는 이미 C로 최적화됨
        - 알고리즘 자체가 순차적이어서 단계 간 병렬화 불가능
        
        Args:
            word_graph: SemGraph와 동일한 WordGraph 객체
            num_clusters: 원하는 클러스터 수 (None이면 modularity 최대화, 느림!)
            verbose: 진행상황 출력 여부
            
        Returns:
            (cluster_labels, metrics_dict)
        """
        from networkx.algorithms.community import girvan_newman
        import time
        
        start_time = time.time()
        
        # WordGraph를 NetworkX로 변환
        self.nx_graph = word_graph.export_to_networkx(include_weights=True)
        
        if verbose:
            print(f"   Girvan-Newman 시작: {word_graph.num_nodes} 노드, {word_graph.num_edges} 엣지")
            if num_clusters:
                print(f"   목표 클러스터: {num_clusters} (조기 종료 모드)")
            else:
                print(f"   ⚠️  자동 탐색 모드 (느림) - num_clusters 지정 권장")
        
        # Girvan-Newman 알고리즘 실행 (제너레이터 반환)
        communities_generator = girvan_newman(self.nx_graph)
        
        # num_clusters가 지정되지 않으면 modularity가 최대인 분할 찾기
        if num_clusters is None:
            best_partition = None
            best_modularity = -1
            
            # 최대 20개 분할까지만 시도 (계산 비용 고려)
            max_iterations = min(20, word_graph.num_nodes - 1)
            
            if verbose:
                print(f"   자동 탐색: 최대 {max_iterations}번 반복")
            
            for i, partition in enumerate(communities_generator):
                if i >= max_iterations:
                    break
                
                iteration_start = time.time()
                    
                # Modularity 계산
                mod = nx.algorithms.community.modularity(self.nx_graph, partition)
                
                if verbose and i % 5 == 0:
                    elapsed = time.time() - start_time
                    print(f"   반복 {i+1}/{max_iterations}: {len(partition)} 클러스터, "
                          f"modularity={mod:.4f}, 소요시간={elapsed:.1f}초")
                
                if mod > best_modularity:
                    best_modularity = mod
                    best_partition = partition
            
            self.communities = [set(c) for c in best_partition]
            self.modularity = best_modularity
        else:
            # 지정된 클러스터 수까지 분할 (빠름!)
            partition = None
            if verbose:
                print(f"   조기 종료 모드: {num_clusters} 클러스터까지 분할")
            
            for i in range(num_clusters - 1):
                iteration_start = time.time()
                partition = next(communities_generator)
                
                if verbose and (i % 2 == 0 or i == num_clusters - 2):
                    elapsed = time.time() - start_time
                    print(f"   반복 {i+1}/{num_clusters-1}: {len(partition)} 클러스터, "
                          f"소요시간={elapsed:.1f}초")
            
            self.communities = [set(c) for c in partition]
            self.modularity = nx.algorithms.community.modularity(self.nx_graph, partition)
        
        # 커뮤니티를 라벨 배열로 변환
        self.cluster_labels = np.zeros(word_graph.num_nodes, dtype=int)
        for cluster_id, community in enumerate(self.communities):
            for node_id in community:
                self.cluster_labels[node_id] = cluster_id
        
        self.num_clusters = len(self.communities)
        self.clustering_method = "girvan_newman"
        
        total_time = time.time() - start_time
        
        metrics = {
            'method': 'girvan_newman',
            'num_clusters': self.num_clusters,
            'modularity': self.modularity,
            'num_nodes': word_graph.num_nodes,
            'num_edges': word_graph.num_edges,
            'computation_time': total_time
        }
        
        if verbose:
            print(f"✅ Girvan-Newman 클러스터링 완료: {self.num_clusters}개 커뮤니티")
            print(f"   Modularity: {self.modularity:.4f}, 소요시간: {total_time:.2f}초")
        else:
            print(f"✅ Girvan-Newman 클러스터링 완료: {self.num_clusters}개 커뮤니티 (Modularity: {self.modularity:.4f})")
        
        return self.cluster_labels, metrics
    
    # ============================================================
    # 유틸리티 메서드
    # ============================================================
    
    def get_cluster_distribution(self) -> Dict[int, int]:
        """
        클러스터별 노드 수 반환
        
        Returns:
            {cluster_id: count} 딕셔너리
        """
        if self.cluster_labels is None:
            raise RuntimeError("클러스터링이 수행되지 않았습니다.")
        
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}
    
    def get_cluster_words(
        self,
        word_graph: WordGraph,
        cluster_id: int,
        top_n: int = 10
    ) -> List[Tuple[str, int]]:
        """
        특정 클러스터의 상위 단어들 반환 (빈도 순)
        
        Args:
            word_graph: WordGraph 객체
            cluster_id: 클러스터 ID
            top_n: 반환할 단어 수
            
        Returns:
            [(word, frequency), ...] 리스트
        """
        if self.cluster_labels is None:
            raise RuntimeError("클러스터링이 수행되지 않았습니다.")
        
        # 해당 클러스터의 노드들 찾기
        cluster_mask = self.cluster_labels == cluster_id
        cluster_node_ids = np.where(cluster_mask)[0]
        
        # 단어와 빈도 수집
        word_freq_pairs = []
        for node_id in cluster_node_ids:
            word = word_graph.get_word_by_node_id(int(node_id))
            word_freq_pairs.append((word.content, word.freq))
        
        # 빈도순 정렬
        word_freq_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return word_freq_pairs[:top_n]
    
    def get_all_cluster_words(
        self,
        word_graph: WordGraph,
        top_n: int = 10
    ) -> Dict[int, List[Tuple[str, int]]]:
        """
        모든 클러스터의 상위 단어들 반환
        
        Args:
            word_graph: WordGraph 객체
            top_n: 클러스터당 반환할 단어 수
            
        Returns:
            {cluster_id: [(word, frequency), ...]} 딕셔너리
        """
        if self.cluster_labels is None:
            raise RuntimeError("클러스터링이 수행되지 않았습니다.")
        
        result = {}
        for cluster_id in range(self.num_clusters):
            result[cluster_id] = self.get_cluster_words(word_graph, cluster_id, top_n)
        
        return result
    
    def compute_graph_statistics(self) -> Dict[str, Any]:
        """
        그래프 통계 정보 계산
        
        Returns:
            통계 딕셔너리
        """
        if self.nx_graph is None:
            raise RuntimeError("그래프가 로드되지 않았습니다.")
        
        stats = {
            'num_nodes': self.nx_graph.number_of_nodes(),
            'num_edges': self.nx_graph.number_of_edges(),
            'density': nx.density(self.nx_graph),
            'average_degree': sum(dict(self.nx_graph.degree()).values()) / self.nx_graph.number_of_nodes(),
        }
        
        # 연결 성분 분석
        if nx.is_connected(self.nx_graph):
            stats['is_connected'] = True
            stats['diameter'] = nx.diameter(self.nx_graph)
            stats['average_shortest_path_length'] = nx.average_shortest_path_length(self.nx_graph)
        else:
            stats['is_connected'] = False
            stats['num_connected_components'] = nx.number_connected_components(self.nx_graph)
            # 가장 큰 연결 성분의 통계
            largest_cc = max(nx.connected_components(self.nx_graph), key=len)
            largest_subgraph = self.nx_graph.subgraph(largest_cc)
            stats['largest_component_size'] = len(largest_cc)
            stats['largest_component_diameter'] = nx.diameter(largest_subgraph)
        
        # 클러스터링 계수
        stats['average_clustering_coefficient'] = nx.average_clustering(self.nx_graph)
        
        return stats
    
    # ============================================================
    # Private Helper Methods
    # ============================================================
    
    def _labels_to_communities(self, labels: np.ndarray) -> List[set]:
        """
        라벨 배열을 커뮤니티 리스트로 변환
        
        Args:
            labels: [num_nodes] 클러스터 라벨 배열
            
        Returns:
            [set(node_ids), ...] 커뮤니티 리스트
        """
        unique_labels = np.unique(labels)
        communities = []
        
        for label in unique_labels:
            community = set(np.where(labels == label)[0].tolist())
            communities.append(community)
        
        return communities
    
    def _networkx_to_igraph(self, nx_graph: nx.Graph) -> 'ig.Graph':
        """
        NetworkX 그래프를 igraph로 변환
        
        Args:
            nx_graph: NetworkX 그래프
            
        Returns:
            igraph 그래프
        """
        import igraph as ig
        
        # 엣지 리스트 추출
        edges = list(nx_graph.edges())
        
        # 가중치 추출 (있으면)
        weights = None
        if nx_graph.edges():
            first_edge = list(nx_graph.edges(data=True))[0]
            if 'weight' in first_edge[2]:
                weights = [nx_graph[u][v].get('weight', 1.0) for u, v in edges]
        
        # igraph 생성
        g = ig.Graph(n=nx_graph.number_of_nodes(), edges=edges, directed=False)
        
        if weights:
            g.es['weight'] = weights
        
        # 노드 속성 복사
        for attr in nx_graph.nodes[0].keys() if nx_graph.nodes() else []:
            g.vs[attr] = [nx_graph.nodes[i].get(attr) for i in range(nx_graph.number_of_nodes())]
        
        return g
    
    def save_clustering_results(
        self,
        word_graph: WordGraph,
        output_path: str,
        include_words: bool = True
    ) -> None:
        """
        클러스터링 결과를 파일로 저장
        
        Args:
            word_graph: WordGraph 객체
            output_path: 저장 경로
            include_words: 각 클러스터의 상위 단어 포함 여부
        """
        if self.cluster_labels is None:
            raise RuntimeError("클러스터링이 수행되지 않았습니다.")
        
        import json
        from datetime import datetime
        
        results = {
            'method': self.clustering_method,
            'num_clusters': self.num_clusters,
            'modularity': self.modularity,
            'cluster_distribution': self.get_cluster_distribution(),
            'timestamp': datetime.now().isoformat(),
            'graph_stats': {
                'num_nodes': word_graph.num_nodes,
                'num_edges': word_graph.num_edges
            }
        }
        
        if include_words:
            results['cluster_words'] = {
                str(k): [(w, f) for w, f in v]
                for k, v in self.get_all_cluster_words(word_graph, top_n=20).items()
            }
        
        # JSON으로 저장
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 클러스터링 결과 저장: {output_path}")
