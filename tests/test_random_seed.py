"""
랜덤 시드 적용 테스트 스크립트

재현성 보장이 제대로 되는지 확인:
- 같은 설정으로 2회 실행 시 결과가 동일한지 검증
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'core'))

from core.services.GRACE.GRACEConfig import GRACEConfig
from core.services.GRACE.GRACEPipeline import GRACEPipeline

def test_reproducibility():
    """재현성 테스트: 2회 실행 결과 비교"""

    csv_path = '/home/jaehun/lab/SENTIMENT/kaggle_RC_2019-05.csv'

    # 매우 작은 설정 (빠른 테스트)
    config = GRACEConfig(
        csv_path=csv_path,
        num_documents=50,        # 매우 적은 문서
        top_n_words=30,          # 매우 적은 단어
        graphmae_epochs=5,       # 매우 짧은 학습
        embed_size=16,           # 작은 차원
        w2v_dim=8,
        bert_dim=8,
        num_clusters=3,          # 고정 클러스터 수
        save_results=False,
        save_embeddings=False,
        save_graph_viz=False,
        verbose=False
    )

    print("=" * 80)
    print("🔬 재현성 테스트 시작")
    print("=" * 80)
    print(f"설정: {config.num_documents} docs, {config.top_n_words} words, {config.graphmae_epochs} epochs")
    print()

    # 첫 번째 실행
    print("1️⃣  첫 번째 실행...")
    pipeline1 = GRACEPipeline(config)
    results1 = pipeline1.run()

    metrics1 = results1['metrics']
    print(f"   Silhouette: {metrics1['silhouette']:.6f}")
    print(f"   Davies-Bouldin: {metrics1['davies_bouldin']:.6f}")
    print()

    # 두 번째 실행
    print("2️⃣  두 번째 실행...")
    pipeline2 = GRACEPipeline(config)
    results2 = pipeline2.run()

    metrics2 = results2['metrics']
    print(f"   Silhouette: {metrics2['silhouette']:.6f}")
    print(f"   Davies-Bouldin: {metrics2['davies_bouldin']:.6f}")
    print()

    # 비교
    print("=" * 80)
    print("🔍 결과 비교")
    print("=" * 80)

    all_same = True
    for metric_name in metrics1.keys():
        val1 = metrics1[metric_name]
        val2 = metrics2[metric_name]
        diff = abs(val1 - val2)

        if diff < 1e-6:
            status = "✅ IDENTICAL"
        else:
            status = f"❌ DIFFERENT (diff={diff:.8f})"
            all_same = False

        print(f"{metric_name:20s}: {val1:.6f} vs {val2:.6f} - {status}")

    print()
    print("=" * 80)
    if all_same:
        print("✅✅✅ 재현성 테스트 통과! 모든 메트릭이 동일합니다.")
        print("✅ 실험 시작 가능!")
    else:
        print("❌❌❌ 재현성 테스트 실패! 결과가 다릅니다.")
        print("⚠️  랜덤 시드 설정을 다시 확인하세요.")
    print("=" * 80)

    return all_same


if __name__ == '__main__':
    try:
        success = test_reproducibility()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
