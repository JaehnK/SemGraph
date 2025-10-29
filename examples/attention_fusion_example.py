#!/usr/bin/env python3
"""
Attention Fusion 사용 예제

Concat vs Attention Fusion 비교
"""

import sys
from pathlib import Path

# PYTHONPATH 설정
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from core.services.GRACE import GRACEConfig, GRACEPipeline


def example_concat():
    """기존 Concat 방식 (512d)"""
    print("="*80)
    print("Example 1: Concat (기존 방식)")
    print("="*80)

    config = GRACEConfig(
        csv_path='data/ag_news.csv',
        num_documents=1000,

        # 임베딩: Concat (512d = 256d + 256d)
        embedding_method='concat',
        embed_size=512,
        w2v_dim=256,
        bert_dim=256,

        # GraphMAE
        graphmae_epochs=100,

        # 출력
        output_dir='results/example_concat',
        verbose=True
    )

    pipeline = GRACEPipeline(config)
    results = pipeline.run()

    print(f"\n✅ Concat 완료!")
    print(f"   Silhouette: {results['metrics']['silhouette']:.4f}")
    print(f"   NPMI: {results['metrics']['npmi']:.4f}")


def example_attention_gated():
    """Attention Fusion - Gated (256d)"""
    print("\n" + "="*80)
    print("Example 2: Attention Fusion - Gated (256d)")
    print("="*80)

    config = GRACEConfig(
        csv_path='data/ag_news.csv',
        num_documents=1000,

        # 임베딩: Attention Fusion (256d 출력)
        embedding_method='attention',
        embed_size=256,          # 최종 출력 차원
        fusion_type='gated',     # 'cross', 'bidirectional', 'weighted', 'gated'

        # GraphMAE
        graphmae_epochs=100,

        # 출력
        output_dir='results/example_attention_gated',
        verbose=True
    )

    pipeline = GRACEPipeline(config)
    results = pipeline.run()

    print(f"\n✅ Attention (Gated) 완료!")
    print(f"   Silhouette: {results['metrics']['silhouette']:.4f}")
    print(f"   NPMI: {results['metrics']['npmi']:.4f}")


def example_attention_weighted():
    """Attention Fusion - Weighted (256d)"""
    print("\n" + "="*80)
    print("Example 3: Attention Fusion - Weighted (256d)")
    print("="*80)

    config = GRACEConfig(
        csv_path='data/ag_news.csv',
        num_documents=1000,

        # 임베딩: Attention Fusion (256d 출력)
        embedding_method='attention',
        embed_size=256,
        fusion_type='weighted',  # 학습된 가중 평균

        # GraphMAE
        graphmae_epochs=100,

        # 출력
        output_dir='results/example_attention_weighted',
        verbose=True
    )

    pipeline = GRACEPipeline(config)
    results = pipeline.run()

    print(f"\n✅ Attention (Weighted) 완료!")
    print(f"   Silhouette: {results['metrics']['silhouette']:.4f}")
    print(f"   NPMI: {results['metrics']['npmi']:.4f}")


def compare_all_fusion_types():
    """모든 Fusion 방식 비교"""
    print("\n" + "="*80)
    print("Example 4: 모든 Fusion 방식 비교")
    print("="*80)

    fusion_types = ['cross', 'bidirectional', 'weighted', 'gated']
    results_all = {}

    for fusion_type in fusion_types:
        print(f"\n{'='*60}")
        print(f"Testing: {fusion_type}")
        print(f"{'='*60}")

        config = GRACEConfig(
            csv_path='data/ag_news.csv',
            num_documents=500,  # 빠른 테스트

            embedding_method='attention',
            embed_size=256,
            fusion_type=fusion_type,

            graphmae_epochs=50,  # 빠른 테스트

            output_dir=f'results/example_fusion_{fusion_type}',
            verbose=False  # 로그 최소화
        )

        pipeline = GRACEPipeline(config)
        results = pipeline.run()

        results_all[fusion_type] = results['metrics']

    # 결과 비교
    print("\n" + "="*80)
    print("Results Comparison")
    print("="*80)
    print(f"{'Fusion Type':<20} {'Silhouette':<15} {'NPMI':<15}")
    print("-"*80)

    for fusion_type, metrics in results_all.items():
        print(f"{fusion_type:<20} {metrics['silhouette']:>12.4f}    {metrics['npmi']:>12.4f}")

    # 최고 성능
    best_fusion = max(results_all.items(), key=lambda x: x[1]['silhouette'])
    print(f"\n🏆 Best Fusion: {best_fusion[0]} (Silhouette = {best_fusion[1]['silhouette']:.4f})")


def main():
    """메인 함수"""

    print("""
Attention Fusion 예제
=====================

이 스크립트는 다음을 실행합니다:
1. Concat (기존 방식, 512d)
2. Attention Fusion - Gated (256d)
3. Attention Fusion - Weighted (256d)
4. 모든 Fusion 방식 비교

선택:
1. 빠른 테스트 (Example 4만)
2. 전체 실행 (1-4 모두)
3. 종료

""")

    choice = input("선택하세요 (1/2/3): ").strip()

    if choice == '1':
        compare_all_fusion_types()
    elif choice == '2':
        example_concat()
        example_attention_gated()
        example_attention_weighted()
        compare_all_fusion_types()
    else:
        print("종료합니다.")
        return

    print("\n" + "="*80)
    print("✅ 모든 예제 완료!")
    print("="*80)


if __name__ == '__main__':
    main()
