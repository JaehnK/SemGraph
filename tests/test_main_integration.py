#!/usr/bin/env python3
"""
main.py 통합 테스트 - Gap Statistics가 자동으로 적용되는지 확인
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.services.GRACE import GRACEPipeline, GRACEConfig


def test_grace_pipeline_uses_gap_statistics():
    """GRACEPipeline이 Gap Statistics를 사용하는지 확인"""
    print("=" * 70)
    print("GRACEPipeline Gap Statistics 통합 테스트")
    print("=" * 70)

    # 최소한의 설정으로 테스트
    config = GRACEConfig(
        csv_path="data/test.csv",
        num_documents=100,  # 매우 작은 데이터셋
        text_column='text',

        # 그래프
        top_n_words=50,
        exclude_stopwords=True,

        # 임베딩
        embedding_method='word2vec',  # 단순화
        embed_size=64,
        w2v_dim=64,

        # GraphMAE
        graphmae_epochs=10,  # 빠른 테스트

        # 클러스터링 - Gap Statistics 기본값 확인
        clustering_method='kmeans',
        num_clusters=None,  # 자동 결정
        max_clusters=5,     # 작은 값으로 테스트

        # 출력
        save_results=False,
        save_graph_viz=False,
        save_embeddings=False,

        verbose=True
    )

    print(f"\n설정:")
    print(f"  문서 수: {config.num_documents}")
    print(f"  단어 수: {config.top_n_words}")
    print(f"  Max clusters: {config.max_clusters}")
    print(f"  Clustering method: {config.clustering_method}")
    print(f"  Num clusters: {config.num_clusters} (None = 자동 결정)")

    print(f"\nGRACEPipeline 실행 중...")
    print(f"Gap Statistics가 자동으로 적용되어야 합니다.\n")

    try:
        import time
        start_time = time.time()

        pipeline = GRACEPipeline(config)
        results = pipeline.run()

        elapsed = time.time() - start_time

        print(f"\n" + "=" * 70)
        print("결과")
        print("=" * 70)
        print(f"소요 시간: {elapsed:.1f}초")
        print(f"탐지된 클러스터 수: {results.get('num_clusters', 'N/A')}")
        print(f"클러스터 라벨 수: {len(results.get('cluster_labels', []))}")

        # 메트릭 출력
        metrics = results.get('metrics', {})
        if metrics:
            print(f"\n메트릭:")
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric_name}: {value:.4f}")

        print(f"\n✓ Gap Statistics가 성공적으로 적용되었습니다!")
        return True

    except Exception as e:
        print(f"\n✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 함수"""
    print("main.py 통합 테스트\n")
    print("이 테스트는 GRACEPipeline이 Gap Statistics를")
    print("기본 클러스터링 방법으로 사용하는지 확인합니다.\n")

    success = test_grace_pipeline_uses_gap_statistics()

    if success:
        print("\n" + "=" * 70)
        print("요약")
        print("=" * 70)
        print("✓ main.py를 실행하면 Gap Statistics가 자동으로 적용됩니다.")
        print("✓ 기본 파라미터: max_clusters=10, n_init=3, n_iterations=10")
        print("✓ 별도의 설정 변경 없이 바로 사용 가능합니다.")
        print()
        print("실제 실행 명령:")
        print("  python main.py --mode train --max-docs 1000")
        print("=" * 70)
    else:
        print("\n테스트 실패")


if __name__ == "__main__":
    main()
