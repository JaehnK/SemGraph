"""
Attention-based Multi-modal Fusion
Concat의 대안으로 학습 가능한 융합 메커니즘 제공
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CrossAttentionFusion(nn.Module):
    """Cross-attention based fusion (BERT -> W2V)"""

    def __init__(self, dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, w2v_emb: torch.Tensor, bert_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            w2v_emb: (N, 256)
            bert_emb: (N, 256)
        Returns:
            fused: (N, 256)
            attn_weights: (N, N)
        """
        # BERT를 Query, W2V를 Key/Value
        attn_output, attn_weights = self.cross_attn(
            query=bert_emb,
            key=w2v_emb,
            value=w2v_emb
        )

        # Residual connection + normalization
        fused = self.norm(bert_emb + self.dropout(attn_output))

        return fused, attn_weights


class BiDirectionalFusion(nn.Module):
    """Bi-directional cross-attention fusion"""

    def __init__(self, dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        # W2V -> BERT
        self.attn_w2b = nn.MultiheadAttention(dim, num_heads, dropout, batch_first=True)
        self.norm_w2v = nn.LayerNorm(dim)

        # BERT -> W2V
        self.attn_b2w = nn.MultiheadAttention(dim, num_heads, dropout, batch_first=True)
        self.norm_bert = nn.LayerNorm(dim)

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        self.norm_final = nn.LayerNorm(dim)

    def forward(self, w2v_emb: torch.Tensor, bert_emb: torch.Tensor) -> torch.Tensor:
        # W2V 관점에서 BERT 정보 가져오기
        w2v_enhanced, _ = self.attn_w2b(
            query=w2v_emb,
            key=bert_emb,
            value=bert_emb
        )
        w2v_enhanced = self.norm_w2v(w2v_emb + w2v_enhanced)

        # BERT 관점에서 W2V 정보 가져오기
        bert_enhanced, _ = self.attn_b2w(
            query=bert_emb,
            key=w2v_emb,
            value=w2v_emb
        )
        bert_enhanced = self.norm_bert(bert_emb + bert_enhanced)

        # Fusion
        concat = torch.cat([w2v_enhanced, bert_enhanced], dim=-1)
        fused = self.fusion(concat)
        fused = self.norm_final(fused)

        return fused


class LearnedWeightedFusion(nn.Module):
    """Learned weighted average fusion (가장 간단)"""

    def __init__(self, dim: int = 256, hidden_dim: int = 128):
        super().__init__()

        self.weight_net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, w2v_emb: torch.Tensor, bert_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            w2v_emb: (N, 256)
            bert_emb: (N, 256)
        Returns:
            fused: (N, 256)
            weights: (N, 2) - [w2v_weight, bert_weight]
        """
        # Predict weights
        concat = torch.cat([w2v_emb, bert_emb], dim=-1)
        weights = self.weight_net(concat)  # (N, 2)

        w_w2v = weights[:, 0:1]
        w_bert = weights[:, 1:2]

        # Weighted sum
        fused = w_w2v * w2v_emb + w_bert * bert_emb

        return fused, weights


class GatedFusion(nn.Module):
    """Gated fusion mechanism (BERT-style)"""

    def __init__(self, dim: int = 256, dropout: float = 0.1):
        super().__init__()

        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

        # Transform layers
        self.transform_w2v = nn.Linear(dim, dim)
        self.transform_bert = nn.Linear(dim, dim)

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, w2v_emb: torch.Tensor, bert_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            w2v_emb: (N, 256)
            bert_emb: (N, 256)
        Returns:
            fused: (N, 256)
            gate: (N, 256) - gate values [0, 1]
        """
        # Transform
        w2v_t = self.transform_w2v(w2v_emb)
        bert_t = self.transform_bert(bert_emb)

        # Compute gate
        concat = torch.cat([w2v_t, bert_t], dim=-1)
        gate = self.gate(concat)  # (N, 256), values in [0, 1]

        # Gated fusion
        fused = gate * w2v_t + (1 - gate) * bert_t
        fused = self.norm(self.dropout(fused))

        return fused, gate


class AttentionFusionFactory:
    """Factory for creating fusion modules"""

    @staticmethod
    def create(fusion_type: str, dim: int = 256, **kwargs):
        """
        Create fusion module

        Args:
            fusion_type: 'cross', 'bidirectional', 'weighted', 'gated'
            dim: embedding dimension
        """
        if fusion_type == 'cross':
            return CrossAttentionFusion(dim, **kwargs)
        elif fusion_type == 'bidirectional':
            return BiDirectionalFusion(dim, **kwargs)
        elif fusion_type == 'weighted':
            return LearnedWeightedFusion(dim, **kwargs)
        elif fusion_type == 'gated':
            return GatedFusion(dim, **kwargs)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")


# ============================================================
# Sklearn-compatible wrapper for clustering
# ============================================================

class AttentionFusionWrapper:
    """Sklearn-compatible wrapper for attention fusion + clustering"""

    def __init__(
        self,
        fusion_type: str = 'gated',
        dim: int = 256,
        epochs: int = 100,
        lr: float = 0.001,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.fusion_type = fusion_type
        self.dim = dim
        self.epochs = epochs
        self.lr = lr
        self.device = device

        # Create fusion module
        self.fusion_module = AttentionFusionFactory.create(fusion_type, dim)
        self.fusion_module = self.fusion_module.to(device)

    def fit_transform(
        self,
        w2v_emb: torch.Tensor,
        bert_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Train fusion module and return fused embeddings

        Args:
            w2v_emb: (N, 256)
            bert_emb: (N, 256)
        Returns:
            fused: (N, 256)
        """
        # Move to device
        w2v_emb = w2v_emb.to(self.device)
        bert_emb = bert_emb.to(self.device)

        # Optimizer
        optimizer = torch.optim.Adam(self.fusion_module.parameters(), lr=self.lr)

        # Training loop (unsupervised - reconstruction loss)
        self.fusion_module.train()

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # Forward
            if self.fusion_type in ['cross', 'weighted', 'gated']:
                fused, weights_or_gate = self.fusion_module(w2v_emb, bert_emb)
            else:
                fused = self.fusion_module(w2v_emb, bert_emb)

            # Reconstruction loss (fused should be similar to inputs)
            loss_w2v = F.mse_loss(fused, w2v_emb)
            loss_bert = F.mse_loss(fused, bert_emb)
            loss = loss_w2v + loss_bert

            # Regularization: encourage diversity
            if self.fusion_type == 'weighted':
                # Encourage balanced weights
                weight_mean = weights_or_gate.mean(dim=0)  # (2,)
                balance_loss = torch.abs(weight_mean[0] - weight_mean[1])
                loss = loss + 0.1 * balance_loss

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")

        # Inference
        self.fusion_module.eval()
        with torch.no_grad():
            if self.fusion_type in ['cross', 'weighted', 'gated']:
                fused, _ = self.fusion_module(w2v_emb, bert_emb)
            else:
                fused = self.fusion_module(w2v_emb, bert_emb)

        return fused.cpu()


# ============================================================
# Test code
# ============================================================

if __name__ == '__main__':
    # Test
    N = 100  # documents
    dim = 256

    w2v_emb = torch.randn(N, dim)
    bert_emb = torch.randn(N, dim)

    print("Testing fusion modules...")
    print("="*60)

    for fusion_type in ['cross', 'bidirectional', 'weighted', 'gated']:
        print(f"\nTesting {fusion_type} fusion:")

        module = AttentionFusionFactory.create(fusion_type, dim)

        if fusion_type in ['cross', 'weighted', 'gated']:
            fused, extra = module(w2v_emb, bert_emb)
            print(f"  Fused shape: {fused.shape}")
            print(f"  Extra shape: {extra.shape}")
        else:
            fused = module(w2v_emb, bert_emb)
            print(f"  Fused shape: {fused.shape}")

        assert fused.shape == (N, dim), f"Expected {(N, dim)}, got {fused.shape}"
        print(f"  ✓ {fusion_type} fusion passed!")

    print("\n" + "="*60)
    print("✅ All tests passed!")
