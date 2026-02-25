"""
GRACE Pipeline + Visualization 통합 실행 스크립트

GRACEPipeline을 실행하여 클러스터링 결과를 얻고,
VisualizationService를 사용하여 다양한 시각화를 생성합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent / "core"
sys.path.insert(0, str(project_root))

import numpy as np
from datetime import datetime

from services.GRACE import GRACEPipeline
from services.GRACE.GRACEConfig import GRACEConfig
from services.Visualization import VisualizationService


def main():
    print("=" * 80)
    print("GRACE Pipeline + Visualization 실행")
    print("=" * 80)

    # ========================================
    # 1. GRACE 파이프라인 설정 및 실행
    # ========================================

    # CSV 경로 설정
    csv_path = "kaggle_RC_2019-05.csv"

    # GRACE 설정 생성
    config = GRACEConfig(
        csv_path=csv_path,
        num_documents=100000,
        text_column='body',  # CSV의 텍스트 컬럼명 (subreddit,body,controversiality,score)

        # 그래프 설정
        top_n_words=500,
        exclude_stopwords=True,

        # 임베딩 설정
        embedding_method='concat',  # 'concat', 'w2v', 'bert'
        embed_size=64,
        w2v_dim=32,
        bert_dim=32,

        # GraphMAE 설정
        graphmae_epochs=100,
        graphmae_lr=0.001,
        mask_rate=0.75,

        # 클러스터링 설정
        num_clusters=None,  # None이면 Elbow Method로 자동 탐색
        min_clusters=3,
        max_clusters=15,

        # 평가 지표
        eval_metrics=['silhouette', 'davies_bouldin', 'calinski_harabasz', 'npmi'],

        # 출력 설정
        save_results=True,
        output_dir='./grace_output',
        save_graph_viz=True,
        save_embeddings=True,
        verbose=True,
        log_interval=10
    )

    print("\n✓ GRACE 설정 완료")
    print(f"  데이터: {config.csv_path}")
    print(f"  문서 수: {config.num_documents}")
    print(f"  임베딩: {config.embedding_method} (dim={config.embed_size})")
    print(f"  GraphMAE epochs: {config.graphmae_epochs}")

    # GRACE 파이프라인 실행
    pipeline = GRACEPipeline(config)
    results = pipeline.run()

    print("\n✓ GRACE 파이프라인 완료")
    print(f"  클러스터 수: {results['num_clusters']}")
    print(f"  클러스터 분포: {results['cluster_distribution']}")
    print("\n평가 지표:")
    for metric_name, value in results['metrics'].items():
        print(f"  - {metric_name}: {value:.4f}")

    # ========================================
    # 2. 시각화 서비스 초기화
    # ========================================

    viz_output_dir = Path(config.output_dir) / "visualizations"
    viz = VisualizationService(output_dir=str(viz_output_dir))

    print(f"\n✓ VisualizationService 초기화: {viz_output_dir}")

    # ========================================
    # 3. 임베딩 시각화 (t-SNE, UMAP)
    # ========================================

    print("\n[1/5] 임베딩 시각화 생성 중...")

    embeddings = pipeline.graphmae_embeddings.numpy()
    labels = pipeline.cluster_labels

    # t-SNE
    tsne_path = viz.visualize_embeddings(
        embeddings=embeddings,
        labels=labels,
        method='tsne',
        filename='grace_embeddings_tsne.png',
        title='GRACE Embeddings (t-SNE)'
    )
    print(f"  ✓ t-SNE: {tsne_path}")

    # UMAP (설치되어 있는 경우)
    try:
        umap_path = viz.visualize_embeddings(
            embeddings=embeddings,
            labels=labels,
            method='umap',
            filename='grace_embeddings_umap.png',
            title='GRACE Embeddings (UMAP)'
        )
        print(f"  ✓ UMAP: {umap_path}")
    except ImportError:
        print("  ⚠ UMAP 미설치 (pip install umap-learn)")

    # ========================================
    # 4. 네트워크 시각화 (클러스터별 색상)
    # ========================================

    print("\n[2/6] 네트워크 그래프 시각화 생성 중...")

    try:
        network_path = viz.visualize_network(
            word_graph=pipeline.word_graph,
            cluster_labels=labels,
            filename='grace_network_visualization.png',
            title='Semantic Network with GRACE Clusters',
            node_size_scale=300,
            edge_width_scale=0.5,
            k=2.0,
            max_edges=1000  # 엣지가 너무 많으면 상위 1000개만 표시
        )
        print(f"  ✓ 네트워크 그래프: {network_path}")
    except ImportError:
        print("  ⚠ networkx 미설치 (pip install networkx)")
    except Exception as e:
        print(f"  ⚠ 네트워크 시각화 실패: {e}")

    # ========================================
    # 5. 클러스터별 워드클라우드
    # ========================================

    print("\n[3/6] 클러스터별 워드클라우드 생성 중...")

    # 클러스터별 단어 추출
    cluster_words = {}
    for cluster_id in range(results['num_clusters']):
        words = pipeline.get_cluster_words(cluster_id)
        cluster_words[cluster_id] = words

    try:
        wordcloud_path = viz.visualize_cluster_words(
            cluster_words=cluster_words,
            filename='grace_cluster_wordclouds.png',
            max_words=50
        )
        print(f"  ✓ 워드클라우드: {wordcloud_path}")
    except ImportError:
        print("  ⚠ wordcloud 미설치 (pip install wordcloud)")

    # ========================================
    # 6. 메트릭 비교 (GRACE vs 베이스라인)
    # ========================================

    print("\n[4/6] 메트릭 비교 차트 생성 중...")

    # GRACE 결과를 비교 형식으로 변환
    comparison_results = {
        'GRACE': results['metrics']
    }

    # 베이스라인이 있다면 추가 (예시)
    # comparison_results['TF-IDF'] = {...}
    # comparison_results['Word2Vec'] = {...}

    comparison_path = viz.compare_methods(
        results_dict=comparison_results,
        filename='grace_metrics_comparison.png'
    )
    print(f"  ✓ 메트릭 비교: {comparison_path}")

    # ========================================
    # 7. Elbow Curve (num_clusters=None인 경우)
    # ========================================

    if config.num_clusters is None and hasattr(pipeline.clustering_service, 'k_range'):
        print("\n[5/6] Elbow Curve 생성 중...")

        k_range = list(range(config.min_clusters, config.max_clusters + 1))
        inertias = pipeline.clustering_service.inertias

        elbow_path = viz.plot_elbow(
            k_range=k_range,
            inertias=inertias,
            filename='grace_elbow_curve.png',
            title='Optimal Number of Clusters (Elbow Method)'
        )
        print(f"  ✓ Elbow Curve: {elbow_path}")
    else:
        print("\n[5/6] Elbow Curve 건너뛰기 (num_clusters 지정됨)")

    # ========================================
    # 8. 전체 리포트 생성 (선택사항)
    # ========================================

    print("\n[6/6] 전체 리포트 생성 중...")

    try:
        report_paths = viz.generate_full_report(
            embeddings=embeddings,
            labels=labels,
            results_dict=comparison_results,
            ablation_results=None,  # Ablation study가 있다면 추가
            param_names=None,
            cluster_words=cluster_words,
            prefix='grace_'
        )

        print(f"  ✓ 전체 리포트 생성 완료 ({len(report_paths)}개 파일)")
        for viz_type, path in report_paths.items():
            print(f"    - {viz_type}: {Path(path).name}")
    except Exception as e:
        print(f"  ⚠ 전체 리포트 생성 실패: {e}")

    # ========================================
    # 8. 클러스터 결과 CSV 저장
    # ========================================

    print("\n클러스터 결과 CSV 저장 중...")
    csv_output_path = Path(config.output_dir) / f"grace_clusters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pipeline.export_cluster_csv(str(csv_output_path))
    print(f"  ✓ CSV 저장: {csv_output_path}")

    # ========================================
    # 완료
    # ========================================

    print("\n" + "=" * 80)
    print("✅ 모든 작업 완료!")
    print("=" * 80)
    print(f"\n결과 디렉토리:")
    print(f"  - GRACE 출력: {Path(config.output_dir).absolute()}")
    print(f"  - 시각화: {viz_output_dir.absolute()}")
    print()


if __name__ == "__main__":
    main()
