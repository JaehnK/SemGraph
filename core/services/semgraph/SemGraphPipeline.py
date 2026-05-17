"""
SemGraph Pipeline

전체 파이프라인:
1. 데이터 전처리 (DocumentService)
2. 의미연결망 구축 (GraphService)
3. 멀티모달 임베딩 (Word2Vec + BERT)
4. GraphMAE 자기지도학습
5. 클러스터링 (K-means, DBSCAN 등)
6. 평가 및 결과 분석
"""

import os
import sys
import pandas as pd
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import json
from datetime import datetime

# 서비스 import
from ..Document.DocumentService import DocumentService
from ..Graph.GraphService import GraphService
from ..Graph.NodeFeatureHandler import NodeFeatureHandler
from ..GraphMAE import GraphMAEService, GraphMAEConfig
from entities import Word, WordGraph, NodeFeatureType

from .SemGraphConfig import SemGraphConfig
from ..clustering import SphericalKMeansClusteringService
from ..Metric import MetricsService
from ..Visualization import VisualizationService


class SemGraphPipeline:
    """SemGraph 파이프라인 오케스트레이터"""

    def __init__(self, config: SemGraphConfig):
        """
        Args:
            config: SemGraph 파이프라인 설정
        """
        self.config = config

        # 재현성을 위한 랜덤 시드 고정 (전역)
        import random
        import numpy as np
        import dgl
        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        random.seed(self.config.random_seed)
        dgl.seed(self.config.random_seed)  # DGL 랜덤 시드
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_seed)
            # CUDA 연산의 재현성 보장
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # 서비스 인스턴스
        self.doc_service: Optional[DocumentService] = None
        self.graph_service: Optional[GraphService] = None
        self.node_feature_handler: Optional[NodeFeatureHandler] = None
        # Spherical K-means (코사인 거리 기반) - 텍스트 임베딩에 최적화
        self.clustering_service = SphericalKMeansClusteringService(random_state=self.config.random_seed)
        self.metrics_service = MetricsService()

        # 중간 결과 저장
        self.documents: Optional[List[str]] = None
        self.word_graph: Optional[WordGraph] = None
        self.node_features: Optional[torch.Tensor] = None
        self.graphmae_embeddings: Optional[torch.Tensor] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_results: Optional[Dict[str, Any]] = None

        # 메타데이터
        self.pipeline_start_time: Optional[datetime] = None
        self.pipeline_end_time: Optional[datetime] = None

    # ============================================================
    # 메인 실행 메서드
    # ============================================================

    def run(self) -> Dict[str, Any]:
        """
        전체 파이프라인 실행

        Returns:
            최종 결과 딕셔너리
        """
        self.pipeline_start_time = datetime.now()
        self._log("🚀 SemGraph 파이프라인 시작")
        self._log("=" * 80)

        try:
            # 1. 데이터 로딩 및 전처리
            self._log("\n[1/6] 데이터 로딩 및 전처리")
            self.load_and_preprocess()

            # 2. 의미연결망 구축
            self._log("\n[2/6] 의미연결망 구축")
            self.build_semantic_network()

            # 3. 멀티모달 노드 특성 계산
            self._log("\n[3/6] 멀티모달 임베딩 계산")
            self.compute_node_features()

            # 4. GraphMAE 자기지도학습
            self._log("\n[4/6] GraphMAE 자기지도학습")
            self.train_graphmae()

            # 5. 클러스터링
            self._log("\n[5/6] 클러스터링 수행")
            self.perform_clustering()

            # 6. 평가 및 결과 저장
            self._log("\n[6/6] 평가 및 결과 저장")
            results = self.evaluate_and_save()

            self.pipeline_end_time = datetime.now()
            elapsed = (self.pipeline_end_time - self.pipeline_start_time).total_seconds()

            self._log("\n" + "=" * 80)
            self._log(f"✅ SemGraph 파이프라인 완료 (소요시간: {elapsed:.2f}초)")
            self._log("=" * 80)

            return results

        except Exception as e:
            self._log(f"\n❌ 파이프라인 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    # ============================================================
    # 1. 데이터 로딩 및 전처리
    # ============================================================

    def load_and_preprocess(self) -> None:
        """데이터 로딩 및 전처리"""
        # CSV 로드
        self._log(f"📁 데이터셋 로드: {self.config.csv_path}")

        if not os.path.exists(self.config.csv_path):
            raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {self.config.csv_path}")

        df = pd.read_csv(self.config.csv_path)
        self._log(f"  전체 데이터: {len(df)} 행")

        if self.config.text_column not in df.columns:
            raise ValueError(f"CSV에 '{self.config.text_column}' 컬럼이 없습니다.")

        # 문서 추출
        self.documents = df[self.config.text_column].dropna().head(
            self.config.num_documents
        ).tolist()
        self._log(f"  로드된 문서: {len(self.documents)}개")

        # DocumentService 초기화 및 전처리
        self.doc_service = DocumentService()
        self.doc_service.create_sentence_list(documents=self.documents)
        self._log(f"  전처리 완료: {self.doc_service.get_sentence_count()}개 문장")
        self._log(f"  추출된 단어: {len(self.doc_service.words_list)}개")

    # ============================================================
    # 2. 의미연결망 구축
    # ============================================================

    def build_semantic_network(self) -> None:
        """공출현 기반 의미연결망 구축"""
        if self.doc_service is None:
            raise RuntimeError("DocumentService가 초기화되지 않았습니다. load_and_preprocess()를 먼저 호출하세요.")

        # GraphService 초기화
        self.graph_service = GraphService(self.doc_service)

        # 공출현 그래프 생성
        self.word_graph = self.graph_service.build_complete_graph(
            top_n=self.config.top_n_words,
            exclude_stopwords=self.config.exclude_stopwords,
            max_length=self.config.max_sentence_length
        )

        self._log(f"  노드 수: {self.word_graph.num_nodes}")
        self._log(f"  생성된 엣지 수: {self.word_graph.num_edges}")

        # 엣지 필터링 적용
        initial_edges = self.word_graph.num_edges

        # 1. 가중치 임계값 필터링
        if self.config.edge_weight_threshold > 0.0:
            self.word_graph.filter_edges_by_weight(self.config.edge_weight_threshold)
            self._log(f"  가중치 필터링 후: {self.word_graph.num_edges} 엣지 "
                     f"({initial_edges - self.word_graph.num_edges} 제거)")
            initial_edges = self.word_graph.num_edges

        # 2. Top-K 필터링
        if self.config.edge_top_k > 0:
            self.word_graph.filter_edges_top_k_per_node(self.config.edge_top_k)
            self._log(f"  Top-{self.config.edge_top_k} 필터링 후: {self.word_graph.num_edges} 엣지 "
                     f"({initial_edges - self.word_graph.num_edges} 제거)")

        self._log(f"  최종 엣지 수: {self.word_graph.num_edges}")

    # ============================================================
    # 3. 멀티모달 노드 특성 계산
    # ============================================================

    def compute_node_features(self) -> None:
        """멀티모달 임베딩 계산 (Word2Vec + BERT)"""
        if self.word_graph is None or self.graph_service is None:
            raise RuntimeError("WordGraph가 생성되지 않았습니다. build_semantic_network()를 먼저 호출하세요.")

        self.node_feature_handler = NodeFeatureHandler(self.doc_service, random_seed=self.config.random_seed)

        method_desc = {
            'concat': f"Word2Vec({self.config.w2v_dim}) + BERT({self.config.bert_dim})",
            'w2v': f"Word2Vec({self.config.embed_size})",
            'bert': f"BERT({self.config.embed_size})",
            'attention': f"Attention Fusion ({self.config.fusion_type}, {self.config.embed_size}d)"
        }
        self._log(f"  임베딩 방법: {method_desc[self.config.embedding_method]}")

        # Attention fusion의 경우 fusion_type 전달
        if self.config.embedding_method == 'attention':
            self.node_features = self.node_feature_handler.calculate_embeddings(
                self.word_graph.words,
                method=self.config.embedding_method,
                embed_size=self.config.embed_size,
                fusion_type=self.config.fusion_type
            )
        else:
            self.node_features = self.node_feature_handler.calculate_embeddings(
                self.word_graph.words,
                method=self.config.embedding_method,
                embed_size=self.config.embed_size
            )

        self._log(f"  노드 특성 형태: {self.node_features.shape}")

        # WordGraph에 설정
        self.word_graph.set_node_features_custom(self.node_features, NodeFeatureType.CUSTOM)

    # ============================================================
    # 4. GraphMAE 자기지도학습
    # ============================================================

    def train_graphmae(self) -> None:
        """GraphMAE 사전학습 및 임베딩 추출"""
        if self.word_graph is None or self.graph_service is None:
            raise RuntimeError("WordGraph가 생성되지 않았습니다.")

        if self.word_graph.node_features is None:
            raise RuntimeError("노드 특성이 계산되지 않았습니다. compute_node_features()를 먼저 호출하세요.")

        embed_size = self.word_graph.node_features.shape[1]

        # GraphMAE 설정
        mae_config = GraphMAEConfig.create_default(embed_size)
        mae_config.max_epochs = self.config.graphmae_epochs
        mae_config.learning_rate = self.config.graphmae_lr
        mae_config.weight_decay = self.config.graphmae_weight_decay
        mae_config.mask_rate = self.config.mask_rate
        mae_config.encoder_type = self.config.encoder_type
        mae_config.decoder_type = self.config.decoder_type
        mae_config.device = self.config.graphmae_device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._log(f"  학습 설정: {mae_config.max_epochs} epochs, device: {mae_config.device}")
        self._log(f"  Encoder: {mae_config.encoder_type}, Decoder: {mae_config.decoder_type}")
        if torch.cuda.is_available() and mae_config.device == "cuda":
            self._log(f"  GPU: {torch.cuda.get_device_name(0)}")

        # GraphMAE 서비스 초기화
        mae_service = GraphMAEService(self.graph_service, mae_config)

        # DGL 그래프로 변환
        dgl_graph = self.graph_service.wordgraph_to_dgl(
            self.word_graph, self.word_graph.node_features
        )

        # 모델 생성 및 학습
        mae_service.model = mae_service.create_mae_model(embed_size, random_seed=self.config.random_seed)
        device_obj = torch.device(mae_config.device)
        mae_service.model.to(device_obj)
        dgl_graph = dgl_graph.to(device_obj)

        optimizer = torch.optim.Adam(
            mae_service.model.parameters(),
            lr=mae_config.learning_rate,
            weight_decay=mae_config.weight_decay
        )

        # 학습 루프
        mae_service.model.train()
        for epoch in range(mae_config.max_epochs):
            optimizer.zero_grad()
            x = dgl_graph.ndata['feat']
            loss = mae_service.model(dgl_graph, x, epoch=epoch)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % self.config.log_interval == 0:
                self._log(f"  Epoch {epoch + 1}/{mae_config.max_epochs}, Loss: {loss.item():.4f}")

        # 임베딩 추출
        mae_service.model.eval()
        with torch.no_grad():
            self.graphmae_embeddings = mae_service.model.embed(dgl_graph, dgl_graph.ndata['feat'])

        self.graphmae_embeddings = self.graphmae_embeddings.cpu()
        self._log(f"  GraphMAE 임베딩 추출 완료: {self.graphmae_embeddings.shape}")

    # ============================================================
    # 5. 클러스터링
    # ============================================================

    def perform_clustering(self) -> None:
        """클러스터링 수행"""
        if self.graphmae_embeddings is None:
            raise RuntimeError("GraphMAE 임베딩이 없습니다. train_graphmae()를 먼저 호출하세요.")

        if self.config.num_clusters is not None:
            # 지정된 클러스터 수로 실행
            self.cluster_labels = self.clustering_service.fit_predict(
                self.graphmae_embeddings,
                n_clusters=self.config.num_clusters,
                n_init=10
            )
            self._log(f"  Spherical K-means 완료: {self.config.num_clusters}개 클러스터")
        else:
            # Kneedle 알고리즘으로 최적 클러스터 수 탐색
            self._log(f"  Kneedle 알고리즘으로 최적 클러스터 수 탐색 중 ({self.config.min_clusters}-{self.config.max_clusters})...")

            self.cluster_labels, best_k, inertias, silhouette_scores = \
                self.clustering_service.auto_clustering(
                    self.graphmae_embeddings,
                    min_clusters=self.config.min_clusters,
                    max_clusters=self.config.max_clusters
                    # method='kneedle' (기본값)
                    # n_init=3 (기본값)
                )

            k_range = range(self.config.min_clusters, self.config.max_clusters + 1)
            self._log(f"  Kneedle 탐지: k={best_k}")
            if silhouette_scores and len(silhouette_scores) > best_k - self.config.min_clusters:
                self._log(f"  최적 클러스터 수: {best_k} (Silhouette: {silhouette_scores[best_k - self.config.min_clusters]:.4f})")
            else:
                self._log(f"  최적 클러스터 수: {best_k}")

            # Elbow curve 시각화 저장
            if self.config.save_graph_viz:
                from pathlib import Path
                from datetime import datetime
                output_dir = Path(self.config.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = output_dir / f"elbow_curve_{timestamp}.png"
                self.clustering_service.save_elbow_curve(list(k_range), str(save_path))
                self._log(f"  Elbow curve 저장: {save_path}")

        # 클러스터별 단어 분포
        cluster_dist = self.clustering_service.get_cluster_distribution()
        self._log(f"  클러스터별 단어 수: {cluster_dist}")

    # ============================================================
    # 6. 평가 및 결과 저장
    # ============================================================

    def evaluate_and_save(self) -> Dict[str, Any]:
        """평가 지표 계산 및 결과 저장"""
        if self.cluster_labels is None or self.graphmae_embeddings is None:
            raise RuntimeError("클러스터링이 완료되지 않았습니다.")

        cluster_dist = self.clustering_service.get_cluster_distribution()
        results = {
            'config': self.config.__dict__,
            'graph_stats': self.word_graph.get_graph_stats(),
            'num_clusters': len(cluster_dist),
            'cluster_distribution': cluster_dist,
            'metrics': {},
            'clusters': self._build_cluster_info()
        }

        # 평가 지표 계산 (MetricsService 사용)
        metrics = self.metrics_service.calculate_metrics(
            self.graphmae_embeddings,
            self.cluster_labels,
            self.config.eval_metrics,
            # NPMI를 위한 추가 정보
            word_graph=self.word_graph,
            total_docs=len(self.documents) if self.documents else 0
        )
        results['metrics'] = metrics

        # 메트릭 출력
        for metric_name, value in metrics.items():
            self._log(f"  {metric_name}: {value:.4f}")

        # 결과 저장
        if self.config.save_results:
            self._save_results(results)

        # 시각화 생성
        if self.config.save_graph_viz:
            self._generate_visualizations()

        self.cluster_results = results
        return results

    # ============================================================
    # 유틸리티 메서드
    # ============================================================

    def _get_sample_sentences_for_cluster(self, cluster_words: List[str], max_samples: int = 10) -> List[str]:
        """
        클러스터의 단어들을 가장 많이 포함한 문장들을 반환

        Args:
            cluster_words: 클러스터에 속한 단어 리스트
            max_samples: 반환할 최대 문장 수

        Returns:
            클러스터 단어를 많이 포함한 문장 리스트 (매칭 개수 내림차순 정렬)
        """
        if self.doc_service is None or self.doc_service.sentence_list is None:
            return []

        # 각 문장에 대해 클러스터 단어 매칭 개수 계산
        sentence_scores = []
        cluster_words_lower = set(w.lower() for w in cluster_words)

        for sentence in self.doc_service.sentence_list:
            # 문장에 포함된 단어들 중 클러스터 단어와 매칭되는 개수 계산
            sentence_words = set(w.content.lower() for w in sentence.word_objects)
            match_count = len(sentence_words & cluster_words_lower)

            if match_count > 0:  # 최소 1개 이상 매칭된 문장만
                sentence_scores.append((sentence.raw, match_count))

        # 매칭 개수 기준 내림차순 정렬
        sentence_scores.sort(key=lambda x: x[1], reverse=True)

        # 상위 max_samples개 문장만 반환
        return [sent for sent, _ in sentence_scores[:max_samples]]

    def _build_cluster_info(self, include_samples: bool = True, max_samples: int = 10) -> Dict[int, Dict[str, Any]]:
        """
        클러스터별 단어 정보 구성 (빈도 및 예시 문장 포함)

        Args:
            include_samples: 예시 문장 포함 여부
            max_samples: 클러스터당 최대 예시 문장 수

        Returns:
            클러스터 정보 딕셔너리
            {
                cluster_id: {
                    "words": [{"word": str, "frequency": int}, ...],
                    "sample_documents": [str, ...]  # 클러스터 단어를 많이 포함한 문장들
                }
            }
        """
        cluster_info = {}

        for cluster_id in np.unique(self.cluster_labels):
            indices = np.where(self.cluster_labels == cluster_id)[0]

            # 단어 정보 수집 (빈도 포함)
            words_with_freq = [
                {
                    "word": self.word_graph.words[i].content,
                    "frequency": self.word_graph.words[i].freq
                }
                for i in indices
            ]

            # 빈도 기준 내림차순 정렬
            words_with_freq.sort(key=lambda x: x["frequency"], reverse=True)

            cluster_data = {
                "words": words_with_freq
            }

            # 예시 문장 추가
            if include_samples:
                cluster_words = [w["word"] for w in words_with_freq]
                cluster_data["sample_documents"] = self._get_sample_sentences_for_cluster(
                    cluster_words,
                    max_samples=max_samples
                )

            cluster_info[int(cluster_id)] = cluster_data

        return cluster_info

    def _save_results(self, results: Dict[str, Any]) -> None:
        """결과를 디스크에 저장"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON 결과 저장
        json_path = output_dir / f"semgraph_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        self._log(f"  결과 저장: {json_path}")

        # 임베딩 저장
        if self.config.save_embeddings:
            embed_path = output_dir / f"embeddings_{timestamp}.pt"
            torch.save({
                'graphmae_embeddings': self.graphmae_embeddings,
                'node_features': self.node_features,
                'cluster_labels': self.cluster_labels,
                'word_list': [w.content for w in self.word_graph.words]
            }, embed_path)
            self._log(f"  임베딩 저장: {embed_path}")

    def _generate_visualizations(self) -> None:
        """시각화 생성 (t-SNE, Word Cloud, Network Graph)"""
        self._log("\n  시각화 생성 중...")

        output_dir = Path(self.config.output_dir) / "visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        viz_service = VisualizationService(output_dir=str(output_dir))

        embeddings = self.graphmae_embeddings.numpy()
        labels = self.cluster_labels

        # 1. t-SNE 시각화
        try:
            tsne_path = viz_service.visualize_embeddings(
                embeddings=embeddings,
                labels=labels,
                method='tsne',
                filename=f'tsne_{timestamp}.png',
                title='SemGraph Embeddings (t-SNE)'
            )
            self._log(f"    ✓ t-SNE: {tsne_path.name}")
        except Exception as e:
            self._log(f"    ✗ t-SNE 실패: {e}")

        # 2. 워드클라우드
        try:
            # 워드클라우드용 단어 리스트 추출 (샘플 문장 제외)
            cluster_info = self._build_cluster_info(include_samples=False)
            cluster_words = {
                cid: [w["word"] for w in cdata["words"]]
                for cid, cdata in cluster_info.items()
            }
            wordcloud_path = viz_service.visualize_cluster_words(
                cluster_words=cluster_words,
                filename=f'wordcloud_{timestamp}.png',
                max_words=50
            )
            self._log(f"    ✓ Word Cloud: {wordcloud_path.name}")
        except ImportError:
            self._log(f"    ✗ Word Cloud 건너뛰기 (wordcloud 패키지 필요)")
        except Exception as e:
            self._log(f"    ✗ Word Cloud 실패: {e}")

        # 3. 네트워크 그래프 (선택적)
        # 네트워크 시각화는 시간이 오래 걸리므로 config에서 제어
        if self.config.save_network_viz:
            try:
                # 엣지가 너무 많으면 상위 N개만 표시 (가독성을 위해)
                # 노드 수에 비례하여 엣지 수 조정
                if self.word_graph.num_edges > 500:
                    max_edges = min(500, self.word_graph.num_nodes * 2)
                else:
                    max_edges = None

                self._log(f"    Starting network visualization (nodes: {self.word_graph.num_nodes}, edges: {self.word_graph.num_edges}, showing: {max_edges if max_edges else 'all'})...")
                network_path = viz_service.visualize_network(
                    word_graph=self.word_graph,
                    cluster_labels=labels,
                    filename=f'network_{timestamp}.png',
                    title='Semantic Network with Clusters',
                    max_edges=max_edges,
                    random_seed=self.config.random_seed
                )
                self._log(f"    ✓ Network Graph: {network_path.name}")
            except ImportError as e:
                self._log(f"    ✗ Network Graph 건너뛰기 (networkx 패키지 필요): {e}")
            except Exception as e:
                self._log(f"    ✗ Network Graph 실패: {e}")
                if self.config.verbose:
                    import traceback
                    traceback.print_exc()
        else:
            self._log(f"    ○ Network Graph: 건너뛰기 (save_network_viz=False)")

    def _log(self, message: str) -> None:
        """로그 출력"""
        if self.config.verbose:
            print(message)

    # ============================================================
    # 외부 접근 메서드
    # ============================================================

    def get_cluster_words(self, cluster_id: int) -> List[str]:
        """특정 클러스터의 단어들 반환"""
        if self.cluster_labels is None:
            raise RuntimeError("클러스터링이 완료되지 않았습니다.")

        indices = np.where(self.cluster_labels == cluster_id)[0]
        return [self.word_graph.words[i].content for i in indices]

    def get_word_cluster(self, word: str) -> Optional[int]:
        """특정 단어가 속한 클러스터 ID 반환"""
        if self.cluster_labels is None or self.word_graph is None:
            raise RuntimeError("클러스터링이 완료되지 않았습니다.")

        node_id = self.word_graph.get_node_id_by_word(word)
        if node_id is None:
            return None

        return int(self.cluster_labels[node_id])

    def export_cluster_csv(self, output_path: str) -> None:
        """클러스터 결과를 CSV로 저장"""
        if self.cluster_labels is None or self.word_graph is None:
            raise RuntimeError("클러스터링이 완료되지 않았습니다.")

        data = []
        for i, word in enumerate(self.word_graph.words):
            data.append({
                'word': word.content,
                'cluster': int(self.cluster_labels[i]),
                'frequency': word.freq,
                'pos': word.dominant_pos
            })

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, encoding='utf-8')
        self._log(f"  클러스터 CSV 저장: {output_path}")
