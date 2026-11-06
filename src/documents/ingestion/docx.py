"""Bounded DOCX text extraction.

The parser is independent of Django models and storage so it can be unit-tested
with in-memory files. It reads the OOXML package directly rather than trusting
the filename or content type supplied by the client.
"""

import io
import zipfile
from xml.etree import ElementTree

from docx import Document as open_docx
from docx.table import Table
from docx.text.paragraph import Paragraph

from documents.exceptions import InvalidDocumentInput
from documents.ingestion.types import ParsedDocument

PARSER_VERSION = "docx-1"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 2000
MAX_TEXT_CHARS = 5_000_000

_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_REQUIRED_MEMBERS = {"[Content_Types].xml", "word/document.xml"}


def parse_docx(upload) -> ParsedDocument:
    """Extract body paragraphs and tables in document order.

    Raises InvalidDocumentInput for anything that is not a safe, readable DOCX.
    """
    data = _read_upload(upload)
    _check_package(data)

    try:
        document = open_docx(io.BytesIO(data))
    except Exception as exc:  # python-docx raises several unrelated types
        raise InvalidDocumentInput("The file is not a readable DOCX document.") from exc

    blocks = []
    for block in _iter_body_blocks(document):
        if isinstance(block, Paragraph):
            if block.text.strip():
                blocks.append(block.text)
        else:
            rows = [
                " | ".join(cell.text.strip() for cell in row.cells) for row in block.rows
            ]
            table_text = "\n".join(row for row in rows if row.strip(" |"))
            if table_text:
                blocks.append(table_text)

    text = "\n\n".join(blocks)
    if not text.strip():
        raise InvalidDocumentInput("The DOCX document contains no extractable text.")
    if len(text) > MAX_TEXT_CHARS:
        raise InvalidDocumentInput("The extracted text exceeds the allowed size.")
    return ParsedDocument(text=text, parser_version=PARSER_VERSION)


def _read_upload(upload) -> bytes:
    if getattr(upload, "size", None) is not None and upload.size > MAX_UPLOAD_BYTES:
        raise InvalidDocumentInput("The uploaded file is too large.")
    data = upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise InvalidDocumentInput("The uploaded file is too large.")
    if not data:
        raise InvalidDocumentInput("The uploaded file is empty.")
    return data


def _check_package(data: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = archive.infolist()
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise InvalidDocumentInput("The DOCX package contains too many files.")
            if sum(member.file_size for member in members) > MAX_UNCOMPRESSED_BYTES:
                raise InvalidDocumentInput("The DOCX package expands beyond the allowed size.")
            names = {member.filename for member in members}
            if not _REQUIRED_MEMBERS <= names:
                raise InvalidDocumentInput("The file is not a valid DOCX document.")
            if any(name.startswith("word/vbaProject") for name in names):
                raise InvalidDocumentInput("Macro-enabled documents are not supported.")
            if "word/encryptedPackage" in names:
                raise InvalidDocumentInput("Encrypted documents are not supported.")
            _check_document_xml(archive.read("word/document.xml"))
    except zipfile.BadZipFile as exc:
        raise InvalidDocumentInput("The file is not a valid DOCX document.") from exc


def _check_document_xml(xml: bytes) -> None:
    # Reject external entities and DTDs before python-docx parses the package.
    if b"<!DOCTYPE" in xml or b"<!ENTITY" in xml:
        raise InvalidDocumentInput("The DOCX document contains unsupported XML declarations.")
    try:
        ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        raise InvalidDocumentInput("The DOCX document contains malformed XML.") from exc


def _iter_body_blocks(document):
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == f"{_WORD_NS}p":
            yield Paragraph(child, document)
        elif child.tag == f"{_WORD_NS}tbl":
            yield Table(child, document)
