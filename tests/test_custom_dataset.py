"""Tests for custom dataset loading."""

import tempfile
from pathlib import Path

import pytest

from khoji.dataset import RetrievalDataset, load_custom


@pytest.fixture
def custom_dataset_dir():
    """Create a temporary directory with a valid custom dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)

        # queries.jsonl
        (base / "queries.jsonl").write_text(
            '{"_id": "q1", "text": "What is Python?"}\n'
            '{"_id": "q2", "text": "How does garbage collection work?"}\n'
        )

        # corpus.jsonl
        (base / "corpus.jsonl").write_text(
            '{"_id": "d1", "text": "Python is a programming language.", "title": "Python"}\n'
            '{"_id": "d2", "text": "GC reclaims memory automatically."}\n'
            '{"_id": "d3", "text": "Unrelated document about cooking."}\n'
        )

        # qrels.tsv
        (base / "qrels.tsv").write_text("q1\td1\t1\nq2\td2\t1\n")

        yield tmpdir


class TestLoadCustom:
    def test_loads_queries(self, custom_dataset_dir):
        ds = load_custom(custom_dataset_dir)
        assert len(ds.queries) == 2
        assert ds.queries["q1"] == "What is Python?"

    def test_loads_corpus(self, custom_dataset_dir):
        ds = load_custom(custom_dataset_dir)
        assert len(ds.corpus) == 3
        # Title should be prepended
        assert ds.corpus["d1"] == "Python Python is a programming language."

    def test_loads_qrels(self, custom_dataset_dir):
        ds = load_custom(custom_dataset_dir)
        assert "q1" in ds.qrels
        assert ds.qrels["q1"]["d1"] == 1

    def test_corpus_without_title(self, custom_dataset_dir):
        ds = load_custom(custom_dataset_dir)
        # d2 has no title, should just be text
        assert ds.corpus["d2"] == "GC reclaims memory automatically."

    def test_missing_queries_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "corpus.jsonl").write_text('{"_id": "d1", "text": "doc"}\n')
            Path(tmpdir, "qrels.tsv").write_text("q1\td1\t1\n")
            with pytest.raises(FileNotFoundError, match="queries"):
                load_custom(tmpdir)

    def test_missing_corpus_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "queries.jsonl").write_text('{"_id": "q1", "text": "query"}\n')
            Path(tmpdir, "qrels.tsv").write_text("q1\td1\t1\n")
            with pytest.raises(FileNotFoundError, match="corpus"):
                load_custom(tmpdir)

    def test_missing_qrels_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "queries.jsonl").write_text('{"_id": "q1", "text": "query"}\n')
            Path(tmpdir, "corpus.jsonl").write_text('{"_id": "d1", "text": "doc"}\n')
            with pytest.raises(FileNotFoundError, match="qrels"):
                load_custom(tmpdir)


class TestRetrievalDataset:
    def test_direct_construction(self):
        ds = RetrievalDataset(
            queries={"q1": "query"},
            corpus={"d1": "doc"},
            qrels={"q1": {"d1": 1}},
        )
        assert ds.queries["q1"] == "query"
        assert ds.corpus["d1"] == "doc"
        assert ds.qrels["q1"]["d1"] == 1
