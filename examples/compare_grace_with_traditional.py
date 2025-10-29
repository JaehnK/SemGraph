"""
GRACE vs Traditional Graph Clustering 비교 실험

GRACE (GraphMAE 기반)와 전통적 그래프 클러스터링 방법들을 비교합니다.
- Louvain
- Leiden  
- Girvan-Newman

동일한 WordGraph를 공유하여 공정한 비교를 보장합니다.
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'core'))

import numpy as np
import pandas as pd
import torch
from datetime import datetime

from services.GRACE import GRACEConfig, GRACEPipeline, TraditionalGraphClusteringService
from services.Document import DocumentService
from services.Graph import GraphService
from services.Graph import NodeFeatureHandler
from services.GraphMAE import GraphMAEService, GraphMAEConfig
from services.clustering import ClusteringService
from services.Metric import MetricsService
from entities import NodeFeatureType


def run_comparison_experiment(
    csv_path: str,
    num_documents: int = 10000,
    top_n_words: int = 500,
    output_dir: str = './comparison_output'
):
    """
    GRACE와 전통적 클러스터링 방법 비교 실험
    
    Args:
        csv_path: 데이터셋 경로
        num_documents: 사용할 문서 수
        top_n_words: 그래프 노드 수
        output_dir: 결과 저장 경로
    """
    
    print("=" * 80)
    print("🔬 GRACE vs Traditional Graph Clustering 비교 실험")
    print("=" * 80)
    print(f"\n📊 실험 설정:")
    print(f"  - 문서 수: {num_documents}")
    print(f"  - 노드 수: {top_n_words}")
    print(f"  - 데이터: {csv_path}")
    print(f"  - 출력: {output_dir}\n")
    
    # 출력 디렉토리 생성
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # ========================================
    # Step 1: 공통 그래프 구축
    # ========================================
    print("\n[Step 1/5] 데이터 전처리 및 의미연결망 구축")
    print("-" * 80)
    
    # CSV 로드 (pandas 사용)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"  전체 데이터: {len(df)} 행")
    
    text_column = 'body'
    if text_column not in df.columns:
        raise ValueError(f"CSV에 '{text_column}' 컬럼이 없습니다. 사용 가능: {df.columns.tolist()}")
    
    # 문서 추출
    documents = df[text_column].dropna().head(num_documents).tolist()
    print(f"  로드된 문서: {len(documents)}개")
    
    # 문서 전처리
    doc_service = DocumentService()
    doc_service.create_sentence_list(documents=documents)
    print(f"  전처리 완료: {doc_service.get_sentence_count()}개 문장")
    
    # 의미연결망 구축
    graph_service = GraphService(doc_service)
    word_graph = graph_service.build_complete_graph(
        top_n=top_n_words,
        exclude_stopwords=True,
        max_length=-1
    )
    
    print(f"✅ 공통 WordGraph 생성 완료")
    print(f"   - 노드: {word_graph.num_nodes}")
    print(f"   - 엣지: {word_graph.num_edges}")
    print(f"   - 밀도: {word_graph.get_graph_stats()['density']:.4f}")
    
    # ========================================
    # Step 2: 전통적 그래프 클러스터링
    # ========================================
    print("\n[Step 2/5] 전통적 그래프 클러스터링 수행")
    print("-" * 80)
    
    traditional_service = TraditionalGraphClusteringService(random_state=42)
    
    traditional_results = {}
    
    # 2.1. Louvain
    print("\n🔵 [1/3] Louvain 클러스터링...")
    try:
        louvain_labels, louvain_metrics = traditional_service.louvain_clustering(
            word_graph, resolution=1.0
        )
        traditional_results['louvain'] = {
            'labels': louvain_labels,
            'metrics': louvain_metrics
        }
    except Exception as e:
        print(f"⚠️  Louvain 실패: {e}")
        traditional_results['louvain'] = None
    
    # 2.2. Leiden
    print("\n🟢 [2/3] Leiden 클러스터링...")
    try:
        leiden_labels, leiden_metrics = traditional_service.leiden_clustering(
            word_graph, resolution=1.0
        )
        traditional_results['leiden'] = {
            'labels': leiden_labels,
            'metrics': leiden_metrics
        }
    except Exception as e:
        print(f"⚠️  Leiden 실패: {e}")
        traditional_results['leiden'] = None
    
    # 2.3. Girvan-Newman (대규모 그래프에서는 스킵)
    print("\n🟠 [3/3] Girvan-Newman 클러스터링...")
    # Girvan-Newman은 O(m²n) 복잡도로 대규모 그래프에서 비현실적
    # 논문에서 Louvain, Leiden만으로 전통적 방법 대표 가능
    if word_graph.num_nodes <= 50:  # 매우 작은 그래프에만 제한
        try:
            print(f"   자동 클러스터 수 결정 (Modularity 최대화)")
            print(f"   ⏳ 시간이 오래 걸릴 수 있습니다 (최대 20번 반복)...")
            
            gn_labels, gn_metrics = traditional_service.girvan_newman_clustering(
                word_graph, num_clusters=None, verbose=True
            )
            traditional_results['girvan_newman'] = {
                'labels': gn_labels,
                'metrics': gn_metrics
            }
        except Exception as e:
            print(f"⚠️  Girvan-Newman 실패: {e}")
            import traceback
            traceback.print_exc()
            traditional_results['girvan_newman'] = None
    else:
        print(f"⚠️  Girvan-Newman 스킵 (노드: {word_graph.num_nodes} > 50)")
        print(f"   💡 O(m²n) 복잡도로 대규모 그래프에서 실행 불가능")
        print(f"   💡 비교 대상: Louvain, Leiden (전통적 방법 대표)")
        traditional_results['girvan_newman'] = None
    
    # ========================================
    # Step 3: GRACE 클러스터링 (GraphMAE)
    # ========================================
    print("\n[Step 3/5] GRACE 클러스터링 수행 (GraphMAE)")
    print("-" * 80)
    
    # 멀티모달 임베딩 계산
    print("📊 멀티모달 임베딩 계산 중...")
    node_feature_handler = NodeFeatureHandler(doc_service)
    
    # concat 방식으로 Word2Vec + BERT 결합 (총 64차원)
    multimodal_features = node_feature_handler.calculate_embeddings(
        words=word_graph.words,
        method='concat',
        embed_size=64
    )
    
    word_graph.set_node_features_custom(
        multimodal_features,
        feature_type=NodeFeatureType.CUSTOM
    )
    
    # GraphMAE 학습
    print("🧠 GraphMAE 자기지도학습...")
    embed_size = multimodal_features.shape[1]
    
    # GraphMAE 설정
    graphmae_config = GraphMAEConfig.create_default(embed_size)
    graphmae_config.max_epochs = 100
    graphmae_config.learning_rate = 0.001
    graphmae_config.weight_decay = 0.0
    graphmae_config.mask_rate = 0.75
    graphmae_config.encoder_type = 'gat'
    graphmae_config.decoder_type = 'gat'
    graphmae_config.loss_fn = 'sce'
    graphmae_config.alpha_l = 2.0
    graphmae_config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"   Epochs: {graphmae_config.max_epochs}, Device: {graphmae_config.device}")
    
    # GraphMAE 서비스 초기화
    graphmae_service = GraphMAEService(graph_service, graphmae_config)
    
    # DGL 그래프로 변환
    dgl_graph = graph_service.wordgraph_to_dgl(word_graph, multimodal_features)
    
    # 모델 생성 및 학습
    graphmae_service.model = graphmae_service.create_mae_model(embed_size)
    device_obj = torch.device(graphmae_config.device)
    graphmae_service.model.to(device_obj)
    dgl_graph = dgl_graph.to(device_obj)
    
    optimizer = torch.optim.Adam(
        graphmae_service.model.parameters(),
        lr=graphmae_config.learning_rate,
        weight_decay=graphmae_config.weight_decay
    )
    
    # 학습 실행
    print(f"   학습 시작: {graphmae_config.max_epochs} epochs...")
    graphmae_service.model.train()
    
    for epoch in range(graphmae_config.max_epochs):
        loss, loss_dict = graphmae_service.model(dgl_graph, dgl_graph.ndata['feat'])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1}/{graphmae_config.max_epochs}: Loss = {loss.item():.4f}")
    
    # 임베딩 추출
    graphmae_service.model.eval()
    with torch.no_grad():
        graphmae_embeddings = graphmae_service.model.embed(dgl_graph, dgl_graph.ndata['feat'])
    
    graphmae_embeddings = graphmae_embeddings.cpu()
    print(f"✅ GraphMAE 학습 완료: 임베딩 shape = {graphmae_embeddings.shape}")
    
    # GRACE 클러스터링 (Elbow Method)
    print("🎯 GRACE 클러스터링 (K-means + Elbow)...")
    clustering_service = ClusteringService(random_state=42)
    grace_labels, best_k, inertias, silhouette_scores = clustering_service.auto_clustering_elbow(
        graphmae_embeddings,
        min_clusters=3,
        max_clusters=20,
        n_init=10
    )
    
    traditional_results['grace'] = {
        'labels': grace_labels,
        'metrics': {
            'method': 'grace',
            'num_clusters': best_k,
            'num_nodes': word_graph.num_nodes,
            'num_edges': word_graph.num_edges
        }
    }
    
    print(f"✅ GRACE 클러스터링 완료: {best_k}개 클러스터")
    
    # ========================================
    # Step 4: 정량적 평가
    # ========================================
    print("\n[Step 4/5] 정량적 평가 (Clustering Metrics)")
    print("-" * 80)
    
    metrics_service = MetricsService()
    
    # 모든 방법에 대해 평가 지표 계산
    evaluation_results = {}
    
    for method_name, result in traditional_results.items():
        if result is None:
            continue
        
        labels = result['labels']
        
        # GraphMAE 임베딩 사용 (GRACE)
        if method_name == 'grace':
            embeddings_np = graphmae_embeddings.numpy()
        else:
            # 전통적 방법은 멀티모달 임베딩 사용 (공정한 비교)
            embeddings_np = multimodal_features.numpy()
        
        # 메트릭 계산
        print(f"\n📊 {method_name.upper()} 평가:")
        
        silhouette = metrics_service.compute_silhouette_score(embeddings_np, labels)
        davies_bouldin = metrics_service.compute_davies_bouldin_score(embeddings_np, labels)
        calinski = metrics_service.compute_calinski_harabasz_score(embeddings_np, labels)
        
        print(f"   Silhouette: {silhouette:.4f}")
        print(f"   Davies-Bouldin: {davies_bouldin:.4f}")
        print(f"   Calinski-Harabasz: {calinski:.2f}")
        
        # NPMI (주제 일관성)
        try:
            # 클러스터별 상위 단어 추출
            cluster_words = {}
            for cluster_id in range(result['metrics']['num_clusters']):
                mask = labels == cluster_id
                node_ids = np.where(mask)[0]
                words = [word_graph.get_word_by_node_id(int(nid)).content for nid in node_ids[:10]]
                cluster_words[cluster_id] = words
            
            npmi = metrics_service.compute_npmi_coherence(cluster_words, doc_service.sentence_list)
            print(f"   NPMI: {npmi:.4f}")
        except Exception as e:
            print(f"   NPMI: N/A ({e})")
            npmi = None
        
        evaluation_results[method_name] = {
            'num_clusters': result['metrics']['num_clusters'],
            'silhouette': float(silhouette),
            'davies_bouldin': float(davies_bouldin),
            'calinski_harabasz': float(calinski),
            'npmi': float(npmi) if npmi is not None else None,
            'modularity': result['metrics'].get('modularity', None)
        }
    
    # ========================================
    # Step 5: 결과 저장
    # ========================================
    print("\n[Step 5/5] 결과 저장")
    print("-" * 80)
    
    # 비교 테이블 생성
    comparison_df = pd.DataFrame(evaluation_results).T
    comparison_df = comparison_df.round(4)
    
    print("\n📊 비교 결과:")
    print(comparison_df.to_string())
    
    # CSV 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"{output_dir}/comparison_results_{timestamp}.csv"
    comparison_df.to_csv(csv_path)
    print(f"\n💾 결과 저장: {csv_path}")
    
    # JSON 저장 (상세 정보)
    import json
    json_path = f"{output_dir}/comparison_results_{timestamp}.json"
    
    # numpy 타입을 JSON 직렬화 가능하게 변환
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj
    
    json_results = {
        'experiment_config': {
            'num_documents': num_documents,
            'top_n_words': top_n_words,
            'timestamp': timestamp
        },
        'evaluation': evaluation_results,
        'cluster_distributions': {
            method: {
                'num_clusters': result['metrics']['num_clusters'],
                'distribution': {
                    int(k): int(v) for k, v in 
                    zip(*np.unique(result['labels'], return_counts=True))
                }
            }
            for method, result in traditional_results.items() if result is not None
        }
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False, default=convert_to_serializable)
    
    print(f"💾 상세 결과 저장: {json_path}")
    
    # ========================================
    # 결론
    # ========================================
    print("\n" + "=" * 80)
    print("✅ 비교 실험 완료!")
    print("=" * 80)
    print("\n📌 주요 발견:")
    
    # 최고 성능 메서드 찾기
    best_silhouette = max(evaluation_results.items(), key=lambda x: x[1]['silhouette'])
    best_davies = min(evaluation_results.items(), key=lambda x: x[1]['davies_bouldin'])
    
    print(f"  - 최고 Silhouette: {best_silhouette[0].upper()} ({best_silhouette[1]['silhouette']:.4f})")
    print(f"  - 최저 Davies-Bouldin: {best_davies[0].upper()} ({best_davies[1]['davies_bouldin']:.4f})")
    
    if 'grace' in evaluation_results:
        grace_rank = sorted(
            evaluation_results.items(), 
            key=lambda x: x[1]['silhouette'], 
            reverse=True
        )
        grace_position = [i for i, (name, _) in enumerate(grace_rank) if name == 'grace'][0] + 1
        print(f"  - GRACE 순위: {grace_position}/{len(evaluation_results)} (Silhouette 기준)")
    
    print("\n💡 논문 작성 시 활용:")
    print("  1. GRACE가 전통적 방법보다 우수한지 확인")
    print("  2. GraphMAE의 효과 (구조 학습) 입증")
    print("  3. Modularity vs 클러스터 품질 메트릭 비교")
    print("  4. 정성적 분석: 각 방법의 클러스터 단어 비교")
    
    return evaluation_results, traditional_results


if __name__ == '__main__':
    # 실험 설정
    csv_path = '/home/jaehun/lab/SENTIMENT/kaggle_RC_2019-05.csv'
    
    # 실험 실행 (매우 작은 설정으로 빠른 테스트)
    evaluation_results, clustering_results = run_comparison_experiment(
        csv_path=csv_path,
        num_documents=10000,  # 매우 작은 데이터셋으로 빠른 테스트
        top_n_words=500,     # 매우 작은 그래프로 빠른 테스트
        output_dir='./comparison_output'
    )
