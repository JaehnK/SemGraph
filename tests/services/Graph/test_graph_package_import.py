import subprocess
import sys


def test_graph_package_import_does_not_load_graph_adapters():
    code = """
import sys
from core.services import Graph

heavy_modules = ("dgl", "matplotlib")
loaded = sorted(
    name for name in sys.modules
    if any(name == module or name.startswith(module + ".") for module in heavy_modules)
)

assert "GraphService" in Graph.__all__
assert loaded == [], loaded
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stderr == ""
