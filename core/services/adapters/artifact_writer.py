import json
from pathlib import Path
from typing import Any, Mapping

from ..ports import ArtifactWriter


class JsonArtifactWriter(ArtifactWriter):
    """Writes JSON artifacts under a configured output directory."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)

    def write_json(self, filename: str, payload: Mapping[str, Any]) -> str:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return str(path)

