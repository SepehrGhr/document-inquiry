import io

import pytest
from django.contrib.auth import get_user_model
from docx import Document as build_docx
from rest_framework.test import APIClient


@pytest.fixture
def staff_user(db):
    return get_user_model().objects.create_user(
        username="staff", password="test-password", is_staff=True
    )


@pytest.fixture
def api_client(staff_user):
    client = APIClient()
    client.force_authenticate(user=staff_user)
    return client


@pytest.fixture
def make_docx():
    """Build an in-memory DOCX with paragraphs and a table in body order."""

    def _make(paragraphs=("First paragraph",), table_rows=None):
        document = build_docx()
        for paragraph in paragraphs:
            document.add_paragraph(paragraph)
        if table_rows:
            table = document.add_table(rows=len(table_rows), cols=len(table_rows[0]))
            for row_index, row in enumerate(table_rows):
                for col_index, value in enumerate(row):
                    table.cell(row_index, col_index).text = value
        buffer = io.BytesIO()
        document.save(buffer)
        buffer.seek(0)
        buffer.name = "sample.docx"
        return buffer

    return _make
