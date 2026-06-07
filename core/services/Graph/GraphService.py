from typing import List, Optional, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from ..Document.DocumentService import DocumentService
from entities import Word, WordGraph, NodeFeatureType

# DGL은 필요할 때만 import
try:
    import dgl
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False
    print("Warning: DGL not available. GraphMAE features will be disabled.")



class GraphService:
    """그래프 구축 및 분석 서비스"""
    
    def __init__(self, document_service: DocumentService):
        """
        Args:
            document_service: 전처리된 문서 데이터를 제공하는 서비스
        """
        self.doc_service = document_service
        
        # 그래프 데이터
        self.nodes: Optional[torch.Tensor] = None
        self.edges: Optional[torch.Tensor] = None
        self.edges_weight: Optional[torch.Tensor] = None
        
        # 단어-노드 매핑
        self.node_words: Optional[List[Word]] = None
        self.word_to_node: Optional[Dict[str, int]] = None
        
        # PyTorch Geometric Data 객체
        self.graph_data: Optional[Data] = None
    
    def create_word_graph(self, top_n: int = 500, exclude_stopwords: bool = True) -> 'WordGraph':
        """
        DocumentService로부터 WordGraph 객체 생성
        
        Args:
            top_n: 상위 몇 개 단어를 노드로 사용할지
            exclude_stopwords: 불용어 제외 여부
            
        Returns:
            WordGraph 객체
        """
        # WordManagementService에서 상위 단어들 가져오기 (불용어 필터링 포함)
        top_words = self.doc_service._word_service.get_top_words(top_n, exclude_stopwords)
        
        if not top_words:
            raise ValueError("No words found. Please process documents first.")
        
        # WordGraph 객체 생성 (기본적으로 빈도 기반 노드 특성)
        word_graph = WordGraph(top_words)
        
        print(f"Created WordGraph with {len(top_words)} nodes (exclude_stopwords={exclude_stopwords})")
        return word_graph
    
    def set_co_occurrence_edges(self, word_graph: 'WordGraph', max_length: int = -1) -> None:
        """
        기존 Cython build_cooccurrence_edges 함수를 활용하여 WordGraph에 엣지 설정
        
        Args:
            word_graph: 엣지를 설정할 WordGraph 객체
            max_length: 문장당 최대 단어 수 제한 (-1은 제한 없음)
        """
        # DocumentService에서 공출현 엣지 데이터 가져오기 (기존 Cython 함수 활용)
        edges, weights = self.doc_service.get_co_occurrence_edges(word_graph.word_to_node_id)
        
        if not edges:
            print("Warning: No co-occurrence edges found.")
            # 빈 엣지로 설정
            word_graph.set_edges_from_co_occurrence([], [])
            return
        
        # WordGraph에 공출현 엣지 설정
        word_graph.set_edges_from_co_occurrence(edges, weights)
        
        print(f"Set {len(edges)} co-occurrence edges")
    
    def build_complete_graph(self, top_n: int = 500, exclude_stopwords: bool = True, 
                            max_length: int = -1) -> 'WordGraph':
        """
        완전한 WordGraph 생성 (노드 + 공출현 엣지)
        
        Args:
            top_n: 상위 몇 개 단어를 노드로 사용할지
            exclude_stopwords: 불용어 제외 여부
            max_length: 문장당 최대 단어 수 제한
            
        Returns:
            완전히 구성된 WordGraph 객체
        """
        # 1. WordGraph 객체 생성 (빈도 기반 노드 특성)
        word_graph = self.create_word_graph(top_n, exclude_stopwords)
        
        # 2. 공출현 엣지 설정 (기존 Cython 함수 활용)
        self.set_co_occurrence_edges(word_graph, max_length)
        
        return word_graph

    def build_mae_enhanced_graph(self, top_n: int = 500, exclude_stopwords: bool = True,
                               max_length: int = -1, embed_size: int = 64,
                               input_method: str = 'bert', mae_config = None) -> 'WordGraph':
        """
        GraphMAE 사전훈련이 포함된 완전한 그래프 생성

        Args:
            top_n: 상위 몇 개 단어를 노드로 사용할지
            exclude_stopwords: 불용어 제외 여부
            max_length: 문장당 최대 단어 수 제한
            embed_size: 임베딩 크기 (입출력 차원 통일)
            input_method: GraphMAE 입력 특성 방법. SemGraph는 'bert'만 허용한다.
            mae_config: GraphMAE 설정 (None이면 기본값)

        Returns:
            GraphMAE로 향상된 WordGraph 객체
        """
        # 1. 기본 그래프 생성
        word_graph = self.build_complete_graph(top_n, exclude_stopwords, max_length)

        # 2. GraphMAE 서비스 초기화
        from ..GraphMAE import GraphMAEService, GraphMAEConfig
        config = mae_config or GraphMAEConfig.create_default(embed_size)
        mae_service = GraphMAEService(self, config)

        # 3. GraphMAE 사전훈련 및 임베딩 추출
        mae_embeddings = mae_service.pretrain_and_extract(
            word_graph, embed_size, input_method
        )

        # 4. GraphMAE 임베딩을 WordGraph에 설정
        word_graph.set_node_features_custom(mae_embeddings, NodeFeatureType.GRAPHMAE)

        print(f"GraphMAE enhanced graph created with {embed_size}D embeddings")
        return word_graph

    def wordgraph_to_dgl(self, word_graph: 'WordGraph', node_features: Optional[torch.Tensor] = None):
        """
        WordGraph를 DGL 그래프로 변환

        Args:
            word_graph: 변환할 WordGraph 객체
            node_features: 노드 특성 텐서 [num_nodes, feature_dim]
                          None이면 빈도 기반 특성 사용

        Returns:
            DGL 그래프 객체
        """
        if not DGL_AVAILABLE:
            raise ImportError("DGL is not available. Please install DGL to use GraphMAE features.")

        if word_graph.edge_index is None:
            raise ValueError("WordGraph has no edges. Call set_co_occurrence_edges() first.")

        # 엣지 인덱스에서 source, target 노드 추출
        edge_index = word_graph.edge_index
        src_nodes = edge_index[0].numpy()
        dst_nodes = edge_index[1].numpy()

        # DGL 그래프 생성 (무방향 그래프)
        dgl_graph = dgl.graph((src_nodes, dst_nodes), num_nodes=word_graph.num_nodes)
        dgl_graph = dgl.to_bidirected(dgl_graph, copy_ndata=True)

        # Self-loop 추가 (0-in-degree 노드 방지)
        dgl_graph = dgl.add_self_loop(dgl_graph)

        # 노드 특성 설정
        if node_features is not None:
            dgl_graph.ndata['feat'] = node_features
        else:
            # 기본적으로 빈도 기반 특성 사용
            freq_features = torch.tensor([[word.freq] for word in word_graph.words], dtype=torch.float32)
            dgl_graph.ndata['feat'] = freq_features

        # 엣지 가중치 설정 (공출현 빈도 활용)
        if word_graph.edge_attr is not None:
            # 원본 엣지 가중치 추출 (양방향이므로 2배)
            original_weights = word_graph.edge_attr.squeeze().numpy()  # [num_edges]

            # Bidirected 후 가중치도 복제됨 (src->dst, dst->src)
            bidirected_weights = torch.cat([
                torch.tensor(original_weights, dtype=torch.float32),
                torch.tensor(original_weights, dtype=torch.float32)
            ])

            # Self-loop 가중치 (최대값으로 설정)
            num_self_loops = word_graph.num_nodes
            max_weight = bidirected_weights.max() if len(bidirected_weights) > 0 else 1.0
            self_loop_weights = torch.full((num_self_loops,), max_weight, dtype=torch.float32)

            # 전체 가중치 결합
            all_weights = torch.cat([bidirected_weights, self_loop_weights])

            # Min-Max 정규화 [0, 1]
            min_val = all_weights.min()
            max_val = all_weights.max()
            if max_val - min_val > 0:
                all_weights = (all_weights - min_val) / (max_val - min_val)

            # DGL 그래프에 설정
            dgl_graph.edata['weight'] = all_weights.unsqueeze(1)  # [num_edges, 1]

        return dgl_graph

    def build_pytorch_geometric_data(self) -> Data:
        """PyTorch Geometric Data 객체 생성"""
        if self.node_words is None or self.edges is None:
            raise ValueError("Graph not fully constructed. Call create_graph() and create_co_occurrence_edges() first.")
        
        # 노드 특성 생성 (단어 빈도를 특성으로 사용)
        node_features = torch.tensor([[word.freq] for word in self.node_words], dtype=torch.float)
        
        # PyTorch Geometric Data 객체 생성
        self.graph_data = Data(
            x=node_features,
            edge_index=self.edges,
            edge_attr=self.edges_weight.unsqueeze(1)
        )
        
        return self.graph_data
    
    def get_graph_statistics(self) -> Dict[str, any]:
        """그래프 통계 정보 반환"""
        if self.node_words is None:
            return {"error": "Graph not created"}
        
        stats = {
            "num_nodes": len(self.node_words),
            "num_edges": self.edges.shape[1] if self.edges is not None else 0,
            "avg_node_degree": 0,
            "density": 0,
            "avg_weight": 0
        }
        
        if self.edges is not None and self.edges.shape[1] > 0:
            num_nodes = len(self.node_words)
            num_edges = self.edges.shape[1]
            
            stats["avg_node_degree"] = (2 * num_edges) / num_nodes  # 무방향 그래프
            stats["density"] = (2 * num_edges) / (num_nodes * (num_nodes - 1))
            stats["avg_weight"] = float(self.edges_weight.mean()) if self.edges_weight.numel() > 0 else 0
        
        return stats
    
    def get_top_connected_words(self, top_k: int = 10) -> List[Tuple[str, int]]:
        """연결도가 높은 상위 단어들 반환"""
        if self.edges is None or self.node_words is None:
            return []
        
        # 각 노드의 연결도 계산
        node_degrees = torch.bincount(self.edges.flatten(), minlength=len(self.node_words))
        
        # 상위 k개 노드 선택
        top_indices = torch.topk(node_degrees, min(top_k, len(self.node_words))).indices
        
        return [(self.node_words[idx].content, int(node_degrees[idx])) 
                for idx in top_indices]
    
    def visualize_word_graph(self, word_graph: 'WordGraph', top_k: int = 50, 
                        figsize: Tuple[int, int] = (12, 8), min_weight: float = None, 
                        show_labels: bool = True, save_path: str = None, dpi: int = 300) -> None:
        """
        WordGraph 객체 시각화
        
        Args:
            word_graph: 시각화할 WordGraph 객체
            top_k: 상위 k개 노드만 표시
            figsize: 그림 크기
            min_weight: 최소 엣지 가중치 (필터링용)
            show_labels: 노드 라벨 표시 여부
            save_path: 저장 경로
            dpi: 해상도
        """
        if word_graph.edge_index is None or word_graph.words is None:
            raise ValueError("WordGraph not fully constructed.")
        
        # NetworkX 그래프로 변환
        G = word_graph.export_to_networkx(include_weights=True)
        
        # 상위 k개 노드만 선택
        selected_nodes = list(range(min(top_k, len(word_graph.words))))
        node_labels = {i: word_graph.words[i].content for i in selected_nodes}
        
        # 서브그래프 생성
        G_sub = G.subgraph(selected_nodes).copy()
        
        # 최소 가중치 필터링
        if min_weight is not None:
            edges_to_remove = [(u, v) for u, v, d in G_sub.edges(data=True) 
                                if d.get('weight', 0) < min_weight]
            G_sub.remove_edges_from(edges_to_remove)
        
        if not G_sub.edges():
            print("Warning: No edges to display with current filters.")
            return
        
        # 시각화
        plt.figure(figsize=figsize)
        
        # 레이아웃 계산
        pos = nx.spring_layout(G_sub, k=1, iterations=50, seed=42)
        
        # 노드 그리기 (빈도수에 따라 크기 조정)
        node_sizes = [max(50, word_graph.words[i].freq * 10) for i in selected_nodes 
                        if i < len(word_graph.words)]
        nx.draw_networkx_nodes(G_sub, pos, node_size=node_sizes, 
                            node_color='lightblue', alpha=0.7)
        
        # 엣지 그리기 (가중치에 따라 두께 조정)
        if G_sub.edges():
            edge_weights = [G_sub[u][v]['weight'] for u, v in G_sub.edges()]
            max_weight = max(edge_weights) if edge_weights else 1
            edge_widths = np.array(edge_weights) / max_weight * 3
            nx.draw_networkx_edges(G_sub, pos, width=edge_widths, alpha=0.5, edge_color='gray')
        
        # 라벨 그리기
        if show_labels and pos:
            labels_subset = {node: node_labels[node] for node in G_sub.nodes() if node in node_labels}
            nx.draw_networkx_labels(G_sub, pos, labels_subset, font_size=8)
        
        plt.title(f"Co-occurrence Graph (top {top_k} words)")
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                        facecolor='white', edgecolor='none')
            print(f"Graph saved: {save_path}")
        
        plt.show()
        
        # 통계 출력
        print(f"Nodes displayed: {G_sub.number_of_nodes()}")
        print(f"Edges displayed: {G_sub.number_of_edges()}")
        if G_sub.edges():
            edge_weights = [G_sub[u][v]['weight'] for u, v in G_sub.edges()]
            print(f"Average edge weight: {np.mean(edge_weights):.2f}")
    
    # === 기존 인터페이스 호환성 유지 ===
    
    def create_graph(self, top_n: int = 500, exclude_stopwords: bool = True) -> None:
        """기존 인터페이스 호환 - 내부적으로 WordGraph 생성"""
        word_graph = self.create_word_graph(top_n, exclude_stopwords)
        
        # 기존 속성들 설정 (호환성)
        self.node_words = word_graph.words
        self.word_to_node = word_graph.word_to_node_id
        self._current_word_graph = word_graph
    
    def create_co_occurrence_edges(self) -> None:
        """기존 인터페이스 호환 - WordGraph에 엣지 설정"""
        if not hasattr(self, '_current_word_graph') or self._current_word_graph is None:
            raise ValueError("Graph not created. Call create_graph() first.")
        
        self.set_co_occurrence_edges(self._current_word_graph)
        
        # 기존 속성들 설정 (호환성)
        self.edges = self._current_word_graph.edge_index
        self.edges_weight = self._current_word_graph.edge_attr.squeeze() if self._current_word_graph.edge_attr is not None else None
    
    def visualize(self, top_k: int = 50, figsize: Tuple[int, int] = (12, 8), 
                    min_weight: float = None, show_labels: bool = True, 
                    save_path: str = None, dpi: int = 300) -> None:
        """기존 인터페이스 호환 - WordGraph 시각화"""
        if not hasattr(self, '_current_word_graph') or self._current_word_graph is None:
            raise ValueError("Graph not created. Call create_graph() and create_co_occurrence_edges() first.")
        
        self.visualize_word_graph(self._current_word_graph, top_k, figsize, min_weight, 
                                show_labels, save_path, dpi)
    
    def get_current_word_graph(self) -> Optional['WordGraph']:
        """현재 작업 중인 WordGraph 객체 반환"""
        return getattr(self, '_current_word_graph', None)
    
    def export_to_networkx(self, include_weights: bool = True) -> nx.Graph:
        """기존 인터페이스 호환 - NetworkX 그래프로 변환"""
        if not hasattr(self, '_current_word_graph') or self._current_word_graph is None:
            raise ValueError("Graph not created.")
        
        return self._current_word_graph.export_to_networkx(include_weights)
    
    def build_pytorch_geometric_data(self) -> Data:
        """기존 인터페이스 호환 - PyTorch Geometric Data 객체 생성"""
        if not hasattr(self, '_current_word_graph') or self._current_word_graph is None:
            raise ValueError("Graph not created.")
        
        return self._current_word_graph.to_pytorch_geometric()
    
    def get_graph_statistics(self) -> Dict[str, any]:
        """기존 인터페이스 호환 - 그래프 통계 정보 반환"""
        if not hasattr(self, '_current_word_graph') or self._current_word_graph is None:
            return {"error": "Graph not created"}
        
        return self._current_word_graph.get_graph_stats()
    
    def get_top_connected_words(self, top_k: int = 10) -> List[Tuple[str, int]]:
        """기존 인터페이스 호환 - 연결도가 높은 상위 단어들 반환"""
        if not hasattr(self, '_current_word_graph') or self._current_word_graph is None:
            return []
        
        word_graph = self._current_word_graph
        if word_graph.edge_index is None:
            return []
        
        # 각 노드의 연결도 계산
        node_degrees = torch.bincount(word_graph.edge_index.flatten(), minlength=len(word_graph.words))
        
        # 상위 k개 노드 선택
        top_indices = torch.topk(node_degrees, min(top_k, len(word_graph.words))).indices
        
        return [(word_graph.words[idx].content, int(node_degrees[idx])) 
                for idx in top_indices]
        
    def reset(self) -> None:
        """그래프 데이터 초기화"""
        if hasattr(self, '_current_word_graph'):
            self._current_word_graph = None
        
        # 기존 호환성 속성들 초기화
        self.nodes = None
        self.edges = None
        self.edges_weight = None
        self.node_words = None
        self.word_to_node = None
        self.graph_data = None
    
    def __str__(self) -> str:
        if not hasattr(self, '_current_word_graph') or self._current_word_graph is None:
            return "GraphService: No graph created"
        
        word_graph = self._current_word_graph
        return f"GraphService: {word_graph.num_nodes} nodes, {word_graph.num_edges} edges"
