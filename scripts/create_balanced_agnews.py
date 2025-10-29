#!/usr/bin/env python3
"""
AG News 원본에서 균형 잡힌 데이터셋 생성

각 클래스당 2,500개씩 총 10,000개 샘플링
"""

import pandas as pd
from datasets import load_dataset
import random

def create_balanced_agnews(samples_per_class=2500, random_seed=42):
    """
    AG News 원본에서 균형 잡힌 데이터셋 생성

    Args:
        samples_per_class: 각 클래스당 샘플 개수
        random_seed: 재현성을 위한 랜덤 시드
    """

    random.seed(random_seed)

    print("="*80)
    print("AG News 균형 잡힌 데이터셋 생성")
    print("="*80)

    # AG News 원본 다운로드 (train split)
    print("\n1. AG News 원본 다운로드 중...")
    dataset = load_dataset("ag_news", split="train")
    print(f"   원본 데이터: {len(dataset)} 샘플")

    # DataFrame 변환
    df = pd.DataFrame({
        'text': dataset['text'],
        'label': dataset['label']
    })

    print(f"\n2. 원본 클래스 분포:")
    print(df['label'].value_counts().sort_index())

    # 각 클래스별로 샘플링
    print(f"\n3. 각 클래스당 {samples_per_class}개씩 샘플링 중...")
    balanced_dfs = []

    for label in sorted(df['label'].unique()):
        class_df = df[df['label'] == label]

        # 충분한 샘플이 있는지 확인
        if len(class_df) < samples_per_class:
            raise ValueError(f"클래스 {label}에 {samples_per_class}개보다 적은 샘플이 있습니다.")

        # 랜덤 샘플링
        sampled = class_df.sample(n=samples_per_class, random_state=random_seed)
        balanced_dfs.append(sampled)
        print(f"   클래스 {label}: {len(sampled)} 샘플")

    # 합치기
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)

    # 셔플
    balanced_df = balanced_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    print(f"\n4. 최종 데이터셋:")
    print(f"   총 샘플: {len(balanced_df)}")
    print(f"   클래스 분포:")
    print(balanced_df['label'].value_counts().sort_index())

    # 저장
    output_path = "data/ag_news.csv"
    balanced_df.to_csv(output_path, index=False)
    print(f"\n✅ 저장 완료: {output_path}")

    # 통계 출력
    print(f"\n📊 데이터 통계:")
    print(f"   평균 텍스트 길이: {balanced_df['text'].str.len().mean():.1f} 문자")
    print(f"   최소 텍스트 길이: {balanced_df['text'].str.len().min()} 문자")
    print(f"   최대 텍스트 길이: {balanced_df['text'].str.len().max()} 문자")

    # 샘플 출력
    print(f"\n📝 샘플 데이터:")
    for i in range(2):
        row = balanced_df.iloc[i]
        print(f"   [{row['label']}] {row['text'][:100]}...")

    print("\n" + "="*80)
    print("✅ 완료!")
    print("="*80)

    return balanced_df


if __name__ == "__main__":
    # 각 클래스당 2,500개씩 총 10,000개
    create_balanced_agnews(samples_per_class=2500, random_seed=42)
