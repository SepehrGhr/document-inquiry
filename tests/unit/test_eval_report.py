from rag.evaluation.report import _format, _table, retrieval_markdown


def test_table_renders_every_group_and_column():
    table = _table(
        {
            "simple": {"questions": 2, "recall@1": 0.5, "recall@3": 1.0, "mrr": 0.75},
            "multi_hop": {"questions": 1, "recall@1": 0.0, "recall@3": 0.0, "mrr": 0.0},
        }
    )

    lines = table.splitlines()
    assert lines[0].startswith("| group | questions | recall@1 |")
    assert "| simple | 2 | 0.500 | 1.000 |" in lines[2]
    assert "| multi_hop | 1 | 0.000 | 0.000 |" in lines[3]


def test_missing_metric_renders_as_a_dash():
    assert _format(None) == "—"
    assert _format(0.12345) == "0.123"
    assert _format(7) == "7"


def test_markdown_includes_metadata_and_sections():
    report = {
        "metadata": {
            "generated_at": "2026-11-07T00:00:00+00:00",
            "git_commit": "abc1234",
            "profile": {
                "name": "default",
                "provider": "openrouter",
                "model_id": "nvidia/nemotron-3-embed-1b:free",
                "dimension": 2048,
                "metric": "cosine",
                "chunker_version": "char-1",
                "chunk_size": 1200,
                "chunk_overlap": 200,
            },
            "build_id": "build-1",
            "top_k": 10,
            "retrieval_mode": "dense",
        },
        "overall": {"questions": 59, "recall@5": 0.8, "mrr": 0.6},
        "by_difficulty": {"simple": {"questions": 18, "recall@5": 0.9}},
        "by_language": {"fa": {"questions": 30, "recall@5": 0.7}},
        "errors": [{"question_id": "q009", "error": "timeout"}],
    }

    markdown = retrieval_markdown(report)

    assert "# Retrieval evaluation" in markdown
    assert "`abc1234`" in markdown
    assert "2048 dims" in markdown
    assert "## By difficulty" in markdown
    assert "## By language" in markdown
    assert "`q009`: timeout" in markdown
