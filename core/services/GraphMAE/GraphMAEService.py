from typing import Any, List, Optional, TYPE_CHECKING
import torch
import sys
import os
import numpy as np
import random

from .GraphMAEConfig import GraphMAEConfig

if TYPE_CHECKING:
    from ..Graph.GraphService import GraphService
    from entities import Word, WordGraph


class GraphMAEService:
    """GraphMAE 기반 노드 임베딩 서비스"""

    def __init__(self, graph_service: "GraphService", config: Optional[GraphMAEConfig] = None):
        """
        Args:
            graph_service: 그래프 생성을 위한 GraphService
            config: GraphMAE 설정 (None이면 기본값 사용)
        """
        self.graph_service = graph_service
        self.config = config or GraphMAEConfig.create_default()
        self.model: Optional[Any] = None

        # NodeFeatureHandler 초기화 (수정된 생성자에 맞춤)
        from ..Graph.NodeFeatureHandler import NodeFeatureHandler
        self.node_handler = NodeFeatureHandler(graph_service.doc_service)

    def create_mae_model(self, input_dim: int, random_seed: int = 42) -> Any:
        """
        주어진 입력 차원에 맞는 GraphMAE 모델 생성

        Args:
            input_dim: 입력 특성 차원 (embed_size와 동일)
            random_seed: 재현성을 위한 랜덤 시드

        Returns:
            GraphMAE PreModel 인스턴스
        """
        # 재현성을 위한 랜덤 시드 고정
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # 재현성을 위한 독립적인 generator 생성
        generator = torch.Generator()
        generator.manual_seed(random_seed)

        PreModel = self._load_premodel_class()
        model = PreModel(
            in_dim=input_dim,
            num_hidden=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            num_dec_layers=self.config.num_dec_layers,
            num_remasking=self.config.num_remasking,
            nhead=self.config.nhead,
            nhead_out=self.config.nhead_out,
            activation=self.config.activation,
            feat_drop=self.config.feat_drop,
            attn_drop=self.config.attn_drop,
            negative_slope=self.config.negative_slope,
            residual=self.config.residual,
            norm=self.config.norm,
            mask_rate=self.config.mask_rate,
            remask_rate=self.config.remask_rate,
            remask_method=self.config.remask_method,
            mask_method=self.config.mask_method,
            encoder_type=self.config.encoder_type,
            decoder_type=self.config.decoder_type,
            loss_fn=self.config.loss_fn,
            drop_edge_rate=self.config.drop_edge_rate,
            alpha_l=self.config.alpha_l,
            lam=self.config.lam,
            delayed_ema_epoch=self.config.delayed_ema_epoch,
            momentum=self.config.momentum,
            replace_rate=self.config.replace_rate,
            zero_init=self.config.zero_init,
            generator=generator
        )

        return model

    def prepare_input_features(self, words: List["Word"], embed_size: int,
                             method: str = 'bert') -> torch.Tensor:
        """
        GraphMAE 입력용 노드 특성 준비

        Args:
            words: 단어 리스트
            embed_size: 임베딩 크기 (입출력 차원 통일)
            method: 특성 계산 방법 ('bert', 'w2v', 'concat')

        Returns:
            [num_nodes, embed_size] 형태의 특성 텐서
        """
        return self.node_handler.calculate_embeddings(words, method, embed_size)

    def pretrain_and_extract(self, word_graph: "WordGraph", embed_size: int = 64,
                           input_method: str = 'bert') -> torch.Tensor:
        """
        GraphMAE 사전훈련 및 임베딩 추출

        Args:
            word_graph: 훈련할 WordGraph
            embed_size: 임베딩 크기
            input_method: 입력 특성 방법

        Returns:
            [num_nodes, embed_size] 형태의 학습된 임베딩
        """
        # 1. 입력 특성 준비
        input_features = self.prepare_input_features(word_graph.words, embed_size, input_method)

        # 2. DGL 그래프 변환
        dgl_graph = self.graph_service.wordgraph_to_dgl(word_graph, input_features)

        # 3. GraphMAE 모델 생성
        self.model = self.create_mae_model(embed_size)
        device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        dgl_graph = dgl_graph.to(device)

        # 4. 옵티마이저 설정
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # 5. 훈련 루프
        self.model.train()
        print(f"Starting GraphMAE pretraining for {self.config.max_epochs} epochs...")

        for epoch in range(self.config.max_epochs):
            optimizer.zero_grad()

            # Forward pass
            x = dgl_graph.ndata['feat']
            loss = self.model(dgl_graph, x, epoch=epoch)

            # Backward pass
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.config.max_epochs}, Loss: {loss.item():.4f}")

        # 6. 임베딩 추출 (L2 정규화 자동 적용)
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.embed(dgl_graph, dgl_graph.ndata['feat'])

        print(f"GraphMAE pretraining completed. Generated embeddings: {embeddings.shape}")
        print(f"Embeddings are L2-normalized for consistency with SCE loss training objective.")
        return embeddings.cpu()

    def extract_embeddings(self, word_graph: "WordGraph", embed_size: int = 64) -> torch.Tensor:
        """
        사전훈련된 모델에서 임베딩 추출 (훈련 없이)

        Args:
            word_graph: 임베딩을 추출할 WordGraph
            embed_size: 임베딩 크기

        Returns:
            [num_nodes, embed_size] 형태의 L2 정규화된 임베딩
        """
        if self.model is None:
            raise ValueError("Model not trained. Call pretrain_and_extract() first.")

        # 입력 특성 준비
        input_features = self.prepare_input_features(word_graph.words, embed_size)

        # DGL 그래프 변환
        dgl_graph = self.graph_service.wordgraph_to_dgl(word_graph, input_features)
        device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')
        dgl_graph = dgl_graph.to(device)

        # 임베딩 추출 (L2 정규화 자동 적용)
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.embed(dgl_graph, dgl_graph.ndata['feat'])

        return embeddings.cpu()

    def save_model(self, path: str) -> None:
        """모델 저장"""
        if self.model is None:
            raise ValueError("No model to save.")
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str, input_dim: int) -> None:
        """모델 로드"""
        self.model = self.create_mae_model(input_dim)
        self.model.load_state_dict(torch.load(path, map_location='cpu'))
        print(f"Model loaded from {path}")

    @staticmethod
    def _load_premodel_class():
        graphmae2_path = os.path.join(os.path.dirname(__file__), '../../GraphMAE2')
        if graphmae2_path not in sys.path:
            sys.path.append(graphmae2_path)

        from models.edcoder import PreModel
        return PreModel
