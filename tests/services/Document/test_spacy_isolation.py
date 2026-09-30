import subprocess
import sys


def test_document_package_import_does_not_load_spacy():
    code = """
import sys
from core.services import Document

loaded = sorted(
    name for name in sys.modules
    if name == "spacy" or name.startswith("spacy.")
)

assert "DocumentService" in Document.__all__
assert loaded == [], loaded
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stderr == ""


def test_document_service_with_injected_preprocessor_does_not_load_spacy():
    code = """
import sys
from core.services.Document import DocumentService


class FakePreprocessor:
    nlp = None

    def process_text_batch(self, texts, batch_size=100, n_process=None):
        return []


service = DocumentService(preprocessor=FakePreprocessor())

assert isinstance(service._preprocessing, FakePreprocessor)
assert "spacy" not in sys.modules
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stderr == ""
