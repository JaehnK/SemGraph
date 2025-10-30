#!/bin/bash

# Main.py 재현성 테스트 스크립트

echo "======================================================================"
echo "Main.py 재현성 테스트"
echo "======================================================================"

# 기존 테스트 결과 삭제
rm -rf results/test_repro1 results/test_repro2

# 첫 번째 실행
echo ""
echo "첫 번째 실행..."
/home/jaehun/anaconda3/envs/SENTIMENT/bin/python main.py \
    --mode train \
    --random-seed 42 \
    --max-docs 1000 \
    --epochs 100 \
    --output results/test_repro1 \
    --device cpu

# 결과 추출
k1=$(grep -oP 'Optimal k: \K\d+' results/test_repro1/grace_results_*.json | head -1)
sil1=$(grep -oP '"silhouette": \K[0-9.]+' results/test_repro1/grace_results_*.json | head -1)

echo ""
echo "첫 번째 결과: k=$k1, Silhouette=$sil1"

# 두 번째 실행
echo ""
echo "두 번째 실행..."
/home/jaehun/anaconda3/envs/SENTIMENT/bin/python main.py \
    --mode train \
    --random-seed 42 \
    --max-docs 1000 \
    --epochs 100 \
    --output results/test_repro2 \
    --device cpu

# 결과 추출
k2=$(grep -oP 'Optimal k: \K\d+' results/test_repro2/grace_results_*.json | head -1)
sil2=$(grep -oP '"silhouette": \K[0-9.]+' results/test_repro2/grace_results_*.json | head -1)

echo ""
echo "두 번째 결과: k=$k2, Silhouette=$sil2"

# 비교
echo ""
echo "======================================================================"
echo "결과 비교"
echo "======================================================================"

if [ "$k1" == "$k2" ]; then
    echo "✅ k 값 일치: $k1"
else
    echo "❌ k 값 불일치: $k1 vs $k2"
fi

if [ "$sil1" == "$sil2" ]; then
    echo "✅ Silhouette 완벽 일치: $sil1"
else
    # 부동소수점 비교 (소수점 4자리)
    diff=$(echo "scale=4; ($sil1 - $sil2)" | bc | tr -d '-')
    if (( $(echo "$diff < 0.0001" | bc -l) )); then
        echo "✅ Silhouette 거의 일치: $sil1 vs $sil2 (diff: $diff)"
    else
        echo "❌ Silhouette 불일치: $sil1 vs $sil2 (diff: $diff)"
    fi
fi

echo ""
echo "======================================================================"
