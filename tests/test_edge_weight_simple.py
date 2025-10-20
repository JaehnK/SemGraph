"""
Edge Weight 통합 간단한 테스트

수동으로 WordGraph를 만들어서 edge weight 통합을 검증
"""

import torch
import numpy as np
from entities import Word, WordGraph


def test_dgl_edge_weight():
    """DGL 그래프에 edge weight가 올바르게 추가되는지 간단 테스트"""
    print("=" * 80)
    print("Test: DGL Edge Weight 통합 검증")
    print("=" * 80)

    # 1. 간단한 WordGraph 생성
    words = [
        Word("apple", idx=0),
        Word("banana", idx=1),
        Word("cherry", idx=2),
        Word("date", idx=3)
    ]
    for i, w in enumerate(words):
        w.freq = (i + 1) * 10

    word_graph = WordGraph(words)
    print(f"✓ WordGraph 생성: {word_graph.num_nodes} 노드")

    # 2. 엣지 설정 (공출현 빈도)
    edge_list = [(0, 1), (1, 2), (2, 3), (0, 2)]
    co_occurrence_weights = [5.0, 3.0, 7.0, 2.0]
    word_graph.set_edges_from_co_occurrence(edge_list, co_occurrence_weights)
    print(f"✓ 엣지 설정: {word_graph.num_edges} 엣지")
    print(f"  원본 가중치: {co_occurrence_weights}")

    # 3. GraphService import 및 DGL 변환
    from core.services.Graph.GraphService import GraphService
    from core.services.Document.DocumentService import DocumentService

    doc_service = DocumentService()
    graph_service = GraphService(doc_service)

    # 노드 특성 (랜덤)
    node_features = torch.randn(word_graph.num_nodes, 16)

    # DGL 그래프로 변환
    dgl_graph = graph_service.wordgraph_to_dgl(word_graph, node_features)
    print(f"✓ DGL 그래프 변환: {dgl_graph.num_nodes()} 노드, {dgl_graph.num_edges()} 엣지")

    # 4. Edge weight 검증
    if 'weight' not in dgl_graph.edata:
        print("❌ FAILED: edge weight가 DGL 그래프에 없습니다!")
        return False

    edge_weights = dgl_graph.edata['weight']
    print(f"✓ Edge weight 존재: shape={edge_weights.shape}")

    # 5. 정규화 확인
    min_w = edge_weights.min().item()
    max_w = edge_weights.max().item()
    print(f"✓ Edge weight 범위: [{min_w:.4f}, {max_w:.4f}]")

    if min_w < 0.0 or max_w > 1.0 + 1e-5:
        print(f"❌ FAILED: Min-Max 정규화 실패 (범위가 [0,1]을 벗어남)")
        return False

    # 6. NaN/Inf 체크
    if torch.isnan(edge_weights).any():
        print("❌ FAILED: NaN 값 발견!")
        return False

    if torch.isinf(edge_weights).any():
        print("❌ FAILED: Inf 값 발견!")
        return False

    print("✓ NaN/Inf 없음")
    print("\n✅ SUCCESS: DGL 그래프 edge weight 정상 작동!")
    return True


def test_gcn_forward():
    """GCN이 edge weight를 사용하는지 테스트"""
    print("\n" + "=" * 80)
    print("Test: GCN Edge Weight 사용 검증")
    print("=" * 80)

    try:
        import dgl
        from core.GraphMAE2.models.gcn import GCN
    except ImportError as e:
        print(f"❌ FAILED: Import 실패 - {e}")
        return False

    # 간단한 그래프 생성
    src = torch.tensor([0, 1, 2])
    dst = torch.tensor([1, 2, 0])
    g = dgl.graph((src, dst), num_nodes=3)
    g = dgl.add_self_loop(g)

    feat = torch.randn(3, 8)

    # GCN 모델
    gcn = GCN(
        in_dim=8,
        num_hidden=4,
        out_dim=4,
        num_layers=1,
        dropout=0.0,
        activation='relu',
        residual=False,
        norm=None
    )

    # Edge weight 있을 때
    g.edata['weight'] = torch.tensor([0.3, 0.5, 0.8, 1.0, 1.0, 1.0], dtype=torch.float32).unsqueeze(1)
    output_with = gcn(g, feat)
    print(f"✓ Edge weight 있음: output shape={output_with.shape}")

    # Edge weight 없을 때
    g.edata.pop('weight')
    output_without = gcn(g, feat)
    print(f"✓ Edge weight 없음: output shape={output_without.shape}")

    # 차이 계산
    diff = torch.abs(output_with - output_without).mean().item()
    print(f"✓ 출력 차이: {diff:.6f}")

    if diff < 1e-6:
        print("⚠️  WARNING: 출력 차이가 너무 작습니다. Edge weight가 제대로 사용되지 않을 수 있습니다.")
        return False

    print("✓ Edge weight가 출력에 영향을 줌")
    print("\n✅ SUCCESS: GCN edge weight 사용 확인!")
    return True


def main():
    print("\n" + "=" * 80)
    print("Edge Weight 통합 간단 테스트")
    print("=" * 80 + "\n")

    success = True

    # Test 1
    if not test_dgl_edge_weight():
        success = False

    # Test 2
    if not test_gcn_forward():
        success = False

    # 결과
    print("\n" + "=" * 80)
    if success:
        print("✅ 모든 테스트 통과!")
        print("=" * 80)
        print("\n다음 단계:")
        print("  1. 전체 파이프라인 실행")
        print("  2. 성능 비교 (GAT vs GCN+EdgeWeight)")
        print("  3. Ablation study 실행")
    else:
        print("❌ 일부 테스트 실패")
        print("=" * 80)


if __name__ == "__main__":
    main()
