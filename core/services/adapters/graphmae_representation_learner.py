from typing import Any, Callable, Optional

import torch

from ..ports import RepresentationLearner


class GraphMAERepresentationLearner(RepresentationLearner):
    """Trains GraphMAE2 on a WordGraph behind the RepresentationLearner port.

    DGL and GraphMAE2 are imported only when ``fit_transform`` runs.
    """

    def __init__(
        self,
        graph_service: Any,
        config: Any,
        random_seed: int = 42,
        log_interval: int = 10,
        log: Callable[[str], None] = print,
    ):
        self.graph_service = graph_service
        self.config = config
        self.random_seed = random_seed
        self.log_interval = log_interval
        self._log = log
        self.model: Optional[Any] = None

    @staticmethod
    def default_device(log: Callable[[str], None] = print) -> str:
        """CUDA only when both torch and the installed DGL build support it."""
        if not torch.cuda.is_available():
            return "cpu"

        import dgl

        try:
            dgl.graph(([0], [0])).to("cuda")
        except dgl.DGLError:
            log("  ⚠️ 설치된 DGL이 CUDA를 지원하지 않아 GraphMAE를 CPU로 학습합니다.")
            return "cpu"
        return "cuda"

    def fit_transform(self, graph: Any, node_features: torch.Tensor) -> torch.Tensor:
        from ..GraphMAE import GraphMAEService

        config = self.config
        embed_size = node_features.shape[1]

        mae_service = GraphMAEService(self.graph_service, config)
        dgl_graph = self.graph_service.wordgraph_to_dgl(graph, node_features)

        model = mae_service.create_mae_model(embed_size, random_seed=self.random_seed)
        device = torch.device(config.device)
        model.to(device)
        dgl_graph = dgl_graph.to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        model.train()
        for epoch in range(config.max_epochs):
            optimizer.zero_grad()
            x = dgl_graph.ndata['feat']
            loss = model(dgl_graph, x, epoch=epoch)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % self.log_interval == 0:
                self._log(f"  Epoch {epoch + 1}/{config.max_epochs}, Loss: {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            embeddings = model.embed(dgl_graph, dgl_graph.ndata['feat'])

        self.model = model
        return embeddings.cpu()
