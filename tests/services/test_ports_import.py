import subprocess
import sys


def test_ports_import_without_heavy_runtime_modules():
    code = """
import sys
from core.services import ports

heavy_modules = ("spacy", "transformers", "dgl", "matplotlib")
loaded = sorted(
    name for name in sys.modules
    if any(name == module or name.startswith(module + ".") for module in heavy_modules)
)

assert ports.EmbeddingProvider.__name__ == "EmbeddingProvider"
assert loaded == [], loaded
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stderr == ""
