import pytest

from ingestion.parse_raw import parse_raw_file


def test_parse_raw_file_splits_header_and_body(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text(
        "SOURCE: Test Source\n"
        "TITLE: Test Title\n"
        "URL: https://example.com\n"
        "RETRIEVED: 2026-01-01\n"
        "---\n"
        "This is the body text.\n"
        "Second line.\n",
        encoding="utf-8",
    )

    doc = parse_raw_file(path)

    assert doc.source_file == "doc.txt"
    assert doc.title == "Test Title"
    assert doc.url == "https://example.com"
    assert doc.retrieved_date == "2026-01-01"
    assert doc.body == "This is the body text.\nSecond line."


def test_parse_raw_file_raises_when_separator_missing(tmp_path):
    path = tmp_path / "broken.txt"
    path.write_text("TITLE: No separator here\nJust body text, no --- line.\n", encoding="utf-8")

    with pytest.raises(ValueError):
        parse_raw_file(path)
