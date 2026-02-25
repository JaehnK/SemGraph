#!/bin/bash
# RQ1 Complete Ablation Study 실행 스크립트

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "================================================================"
echo "RQ1 Complete Ablation Study"
echo "================================================================"
echo ""
echo "실험 조건:"
echo "  - 6 models × 5 seeds = 30 runs"
echo "  - Models: W2V-KMeans, BERT-KMeans, W2V-GraphMAE, BERT-GraphMAE, Concat-KMeans, GRACE"
echo "  - Seeds: [42, 123, 456, 789, 101]"
echo ""
echo "예상 소요 시간:"
echo "  - W2V-KMeans (5 runs):      ~25분"
echo "  - BERT-KMeans (5 runs):     ~25분"
echo "  - W2V-GraphMAE (5 runs):    ~100분"
echo "  - BERT-GraphMAE (5 runs):   ~100분"
echo "  - Concat-KMeans (5 runs):   ~50분"
echo "  - GRACE (5 runs):           ~150분"
echo "  - 총 예상 시간:             ~450분 (약 7.5시간)"
echo ""
echo "================================================================"
echo ""

read -p "계속 진행하시겠습니까? (y/n): " confirm

if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "실험 취소"
    exit 0
fi

echo ""
echo "실험 시작..."
echo ""

# PYTHONPATH 설정
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

# Conda 환경 활성화
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate SENTIMENT
fi

# 실험 실행
python "$ROOT_DIR/experiments/rq1_single_vs_multi_embedding.py"

echo ""
echo "================================================================"
echo "✅ RQ1 Complete Ablation Study 완료!"
echo "================================================================"
