import io

import pytest

from documents.exceptions import InvalidDocumentInput
from documents.ingestion.docx import parse_docx


def test_extracts_paragraphs_and_tables_in_body_order(make_docx):
    upload = make_docx(
        paragraphs=("عنوان فارسی", "Mixed English and فارسی text"),
        table_rows=[["Name", "نام"], ["Value", "مقدار"]],
    )

    parsed = parse_docx(upload)

    assert parsed.text == (
        "عنوان فارسی\n\nMixed English and فارسی text\n\nName | نام\nValue | مقدار"
    )


def test_preserves_half_space_and_zero_width_characters(make_docx):
    original = "می‌خواهم​ متن را حفظ کنم"

    parsed = parse_docx(make_docx(paragraphs=(original,)))

    assert parsed.text == original


def test_rejects_non_zip_file():
    upload = io.BytesIO(b"not a docx")
    upload.name = "fake.docx"

    with pytest.raises(InvalidDocumentInput):
        parse_docx(upload)


def test_rejects_docx_without_text(make_docx):
    with pytest.raises(InvalidDocumentInput):
        parse_docx(make_docx(paragraphs=("   ",)))
