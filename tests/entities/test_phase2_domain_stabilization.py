import importlib.util
import numpy as np
import pytest
import torch
from pathlib import Path

from entities import Corpus, Documents, EdgeFeatureType, NodeFeatureType, Sentence, Word, WordGraph
from services.Sentence.SentenceAnalysisService import SentenceAnalysisService


def load_word_management_service():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "core/services/Document/WordManagementService.py"
    spec = importlib.util.spec_from_file_location("phase2_word_management_service", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WordManagementService


WordManagementService = load_word_management_service()


class FakeToken:
    def __init__(
        self,
        text,
        lemma,
        pos="NOUN",
        tag="NN",
        is_stop=False,
        is_punct=False,
        is_space=False,
    ):
        self.text = text
        self.lemma_ = lemma
        self.pos_ = pos
        self.tag_ = tag
        self.is_stop = is_stop
        self.is_punct = is_punct
        self.is_space = is_space

    def __str__(self):
        return self.text


def test_documents_remains_compatible_corpus_alias():
    corpus = Corpus()
    corpus.rawdata = [f"document {i}" for i in range(10)]

    assert len(corpus) == 10
    assert corpus.get_document(0) == "document 0"
    assert str(corpus).startswith("Corpus:")

    documents = Documents()
    documents.rawdata = [f"document {i}" for i in range(10)]

    assert isinstance(documents, Corpus)
    assert str(documents).startswith("Documents:")


def test_word_embeddings_are_stored_outside_word_entity():
    service = WordManagementService(Corpus())
    embedding = np.array([0.1, 0.2], dtype=np.float32)

    service.update_word_bert_embedding("topic", embedding)

    word = service.get_all_words()[0]
    assert word.content == "topic"
    assert not hasattr(word, "bert_embedding")
    assert service.get_word_bert_embedding("topic") is embedding


def test_sentence_processing_lives_in_sentence_service():
    corpus = Corpus()
    sentence = Sentence(raw="Running quickly.", docs_ref=corpus)
    spacy_doc = [
        FakeToken("Running", "run", pos="VERB", tag="VBG"),
        FakeToken("quickly", "quickly", pos="ADV", tag="RB"),
        FakeToken(".", ".", is_punct=True),
    ]

    SentenceAnalysisService.process_with_spacy(sentence, spacy_doc, corpus)

    assert sentence.is_processed is True
    assert sentence.lemmatised == ["run", "quickly"]
    assert sentence.pos_tags == ["VB", "RB"]
    assert [word.content for word in sentence.word_objects] == ["run", "quickly"]


def test_word_graph_validates_domain_shape_and_copies_features():
    words = [Word("alpha", idx=0), Word("beta", idx=1)]
    graph = WordGraph(words)
    words.append(Word("gamma", idx=2))

    assert graph.num_nodes == 2

    features = torch.ones(2, 3)
    graph.set_node_features_custom(features, NodeFeatureType.CUSTOM)
    features[0, 0] = 99

    assert graph.node_features[0, 0].item() == 1

    graph.set_edges_combined([], [], [])
    assert graph.num_edges == 0
    assert graph.edge_feature_type == EdgeFeatureType.COMBINED

    with pytest.raises(ValueError, match="unknown node"):
        graph.set_edges_from_co_occurrence([(0, 2)], [1.0])

    with pytest.raises(ValueError, match="unique content"):
        WordGraph([Word("alpha"), Word("alpha")])
