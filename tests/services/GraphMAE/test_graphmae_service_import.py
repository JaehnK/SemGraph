import subprocess
import sys


def test_graphmae_service_import_does_not_load_runtime_backend():
    code = """
import sys
from core.services.GraphMAE.GraphMAEService import GraphMAEService

loaded = sorted(
    name for name in sys.modules
    if name == "dgl" or name.startswith("dgl.") or name == "models.edcoder"
)

assert GraphMAEService.__name__ == "GraphMAEService"
assert loaded == [], loaded
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stderr == ""
