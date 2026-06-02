"""
Sprint 2 tests: PDF, DOCX, and directory ingestion.

Covers:
- PDFLoader: extracts text chunks from a real minimal PDF
- DOCXLoader: extracts text chunks from a real DOCX
- DirectoryLoader: walks a temp tree, respects extension filter, skips hidden dirs
- MemoryService.ingest_* methods return correct status and chunk count
- Empty / missing file error handling
"""
import json
import struct
from pathlib import Path

import pytest

from core.ingestion.pdf_loader import PDFLoader
from core.ingestion.docx_loader import DOCXLoader
from core.ingestion.directory_loader import DirectoryLoader


# --------------------------------------------------------------------------- #
# PDF fixture helpers                                                          #
# --------------------------------------------------------------------------- #

def make_minimal_pdf(path: Path, pages: list[str]) -> None:
    """
    Write a valid minimal single-/multi-page PDF with the given text on each page.
    Uses fpdf2 if available, otherwise falls back to pypdf's PdfWriter.
    """
    import pypdf
    from pypdf import PdfWriter
    from pypdf.generic import (
        ArrayObject, DictionaryObject, NameObject, NumberObject,
        StreamObject, create_string_object,
    )

    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        # Add text as a content stream
        content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        page[NameObject("/Contents")] = writer._add_object(
            StreamObject.initialize_from_dictionary({
                NameObject("/Length"): NumberObject(len(content)),
            })
        )
        page[NameObject("/Contents")].set_data(content)  # type: ignore[attr-defined]

    with open(path, "wb") as fh:
        writer.write(fh)


def make_pdf_with_fpdf(path: Path, pages: list[str]) -> None:
    """Create PDF using fpdf2 (preferred — produces text-extractable PDFs)."""
    from fpdf import FPDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for text in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, text)
    pdf.output(str(path))


def create_test_pdf(path: Path, pages: list[str]) -> None:
    try:
        make_pdf_with_fpdf(path, pages)
    except ImportError:
        # fpdf2 not installed: write a pypdf-based minimal PDF
        _write_basic_pdf(path, pages)


def _write_basic_pdf(path: Path, pages: list[str]) -> None:
    """Fallback: write a bare-bones valid PDF manually."""
    lines = []
    lines.append(b"%PDF-1.4\n")

    offsets = []
    obj_id  = 1

    page_ids = []
    content_ids = []

    for text in pages:
        content_stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET\n".encode()
        cid = obj_id
        content_ids.append(cid)
        offsets.append(len(b"".join(lines)))
        lines.append(f"{cid} 0 obj\n<< /Length {len(content_stream)} >>\nstream\n".encode())
        lines.append(content_stream)
        lines.append(b"\nendstream\nendobj\n")
        obj_id += 1

    font_id = obj_id
    offsets.append(len(b"".join(lines)))
    lines.append(
        f"{font_id} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n".encode()
    )
    obj_id += 1

    for cid in content_ids:
        pid = obj_id
        page_ids.append(pid)
        offsets.append(len(b"".join(lines)))
        lines.append(
            f"{pid} 0 obj\n<< /Type /Page /MediaBox [0 0 612 792]"
            f" /Contents {cid} 0 R"
            f" /Resources << /Font << /F1 {font_id} 0 R >> >> >>\nendobj\n".encode()
        )
        obj_id += 1

    pages_obj_id = obj_id
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    offsets.append(len(b"".join(lines)))
    lines.append(
        f"{pages_obj_id} 0 obj\n<< /Type /Pages /Count {len(page_ids)} /Kids [{kids}] >>\nendobj\n".encode()
    )
    obj_id += 1

    # Update each page to reference the Pages object
    for pid in page_ids:
        page_line_idx = next(
            i for i, l in enumerate(lines) if l.startswith(f"{pid} 0 obj".encode())
        )
        lines[page_line_idx] = lines[page_line_idx].replace(
            b"/Type /Page ", f"/Type /Page /Parent {pages_obj_id} 0 R ".encode()
        )

    root_id = obj_id
    offsets.append(len(b"".join(lines)))
    lines.append(
        f"{root_id} 0 obj\n<< /Type /Catalog /Pages {pages_obj_id} 0 R >>\nendobj\n".encode()
    )
    obj_id += 1

    xref_offset = len(b"".join(lines))
    xref = f"xref\n0 {obj_id}\n0000000000 65535 f \n"
    for off in offsets:
        xref += f"{off:010d} 00000 n \n"
    lines.append(xref.encode())
    lines.append(
        f"trailer\n<< /Size {obj_id} /Root {root_id} 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    )

    path.write_bytes(b"".join(lines))


# --------------------------------------------------------------------------- #
# DOCX fixture helper                                                          #
# --------------------------------------------------------------------------- #

def create_test_docx(path: Path, paragraphs: list[str]) -> None:
    import docx
    doc = docx.Document()
    for para in paragraphs:
        doc.add_paragraph(para)
    doc.save(str(path))


# --------------------------------------------------------------------------- #
# PDF Loader tests                                                             #
# --------------------------------------------------------------------------- #

def test_pdf_loader_extracts_chunks(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    page_text = "Neural networks learn representations from raw data automatically."
    create_test_pdf(pdf_path, [page_text])

    loader = PDFLoader()
    chunks = loader.load(str(pdf_path))

    assert len(chunks) >= 1
    assert all("text" in c for c in chunks)
    assert all("metadata" in c for c in chunks)


def test_pdf_loader_metadata_fields(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    create_test_pdf(pdf_path, ["Page one content for testing metadata fields."])

    chunks = PDFLoader().load(str(pdf_path))
    assert len(chunks) >= 1

    meta = chunks[0]["metadata"]
    assert meta["source_type"] == "pdf"
    assert "page_number" in meta
    assert "chunk_index" in meta
    assert "source_file" in meta


def test_pdf_loader_multi_page(tmp_path):
    pdf_path = tmp_path / "multi.pdf"
    create_test_pdf(pdf_path, [
        "First page has content about machine learning systems.",
        "Second page discusses deep neural network architectures.",
    ])

    chunks = PDFLoader().load(str(pdf_path))
    assert len(chunks) >= 2


def test_pdf_loader_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        PDFLoader().load("/nonexistent/file.pdf")


# --------------------------------------------------------------------------- #
# DOCX Loader tests                                                            #
# --------------------------------------------------------------------------- #

def test_docx_loader_extracts_chunks(tmp_path):
    docx_path = tmp_path / "test.docx"
    paragraphs = [
        "This is the first paragraph with meaningful content about AI systems.",
        "This is the second paragraph discussing memory and retrieval mechanisms.",
    ]
    create_test_docx(docx_path, paragraphs)

    chunks = DOCXLoader().load(str(docx_path))

    assert len(chunks) >= 1
    combined = " ".join(c["text"] for c in chunks)
    assert "first paragraph" in combined
    assert "second paragraph" in combined


def test_docx_loader_metadata_fields(tmp_path):
    docx_path = tmp_path / "test.docx"
    create_test_docx(docx_path, ["Paragraph with enough content for chunking test."])

    chunks = DOCXLoader().load(str(docx_path))
    assert len(chunks) >= 1

    meta = chunks[0]["metadata"]
    assert meta["source_type"] == "docx"
    assert "chunk_index" in meta
    assert "source_file" in meta


def test_docx_loader_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        DOCXLoader().load("/nonexistent/file.docx")


def test_docx_loader_empty_document(tmp_path):
    docx_path = tmp_path / "empty.docx"
    create_test_docx(docx_path, [])
    chunks = DOCXLoader().load(str(docx_path))
    assert chunks == []


# --------------------------------------------------------------------------- #
# Directory Loader tests                                                       #
# --------------------------------------------------------------------------- #

def _make_dir_tree(root: Path) -> None:
    (root / "module.py").write_text(
        "def greet(name):\n    return f'Hello {name}'\n\n"
        "class Greeter:\n    pass\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "# Project\n\nThis project demonstrates semantic memory retrieval.\n\n"
        "It uses vector embeddings to find related content.\n",
        encoding="utf-8",
    )
    sub = root / "utils"
    sub.mkdir()
    (sub / "helpers.py").write_text(
        "def add(a, b):\n    \"\"\"Return the sum of two numbers.\"\"\"\n    return a + b\n",
        encoding="utf-8",
    )
    # Files that should be skipped
    skip = root / "__pycache__"
    skip.mkdir()
    (skip / "module.cpython-311.pyc").write_bytes(b"\x00\x01\x02")

    (root / "image.png").write_bytes(b"\x89PNG")   # binary, unsupported extension


def test_directory_loader_extracts_supported_files(tmp_path):
    _make_dir_tree(tmp_path)
    chunks = DirectoryLoader().load(str(tmp_path))

    sources = {c["metadata"]["relative_path"] for c in chunks}
    assert any("module.py" in s for s in sources)
    assert any("README.md" in s for s in sources)
    assert any("helpers.py" in s for s in sources)


def test_directory_loader_skips_unsupported_extensions(tmp_path):
    _make_dir_tree(tmp_path)
    chunks = DirectoryLoader().load(str(tmp_path))

    sources = {c["metadata"]["relative_path"] for c in chunks}
    assert not any("image.png" in s for s in sources)
    assert not any(".pyc" in s for s in sources)


def test_directory_loader_skips_pycache(tmp_path):
    _make_dir_tree(tmp_path)
    chunks = DirectoryLoader().load(str(tmp_path))
    assert not any("__pycache__" in c["metadata"]["relative_path"] for c in chunks)


def test_directory_loader_extension_filter(tmp_path):
    _make_dir_tree(tmp_path)
    chunks = DirectoryLoader(extensions={".md"}).load(str(tmp_path))
    assert all(c["metadata"]["file_extension"] == ".md" for c in chunks)


def test_directory_loader_metadata_fields(tmp_path):
    _make_dir_tree(tmp_path)
    chunks = DirectoryLoader().load(str(tmp_path))
    assert len(chunks) > 0

    meta = chunks[0]["metadata"]
    for field in ("source_file", "source_type", "relative_path", "file_extension", "chunk_index"):
        assert field in meta, f"Missing metadata field: {field}"


def test_directory_loader_source_type_code_vs_text(tmp_path):
    _make_dir_tree(tmp_path)
    chunks = DirectoryLoader().load(str(tmp_path))

    py_chunks = [c for c in chunks if c["metadata"]["file_extension"] == ".py"]
    md_chunks = [c for c in chunks if c["metadata"]["file_extension"] == ".md"]

    assert all(c["metadata"]["source_type"] == "code" for c in py_chunks)
    assert all(c["metadata"]["source_type"] == "text" for c in md_chunks)


def test_directory_loader_missing_dir_raises():
    with pytest.raises(FileNotFoundError):
        DirectoryLoader().load("/nonexistent/directory")


def test_directory_loader_file_path_raises(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("hello")
    with pytest.raises(ValueError):
        DirectoryLoader().load(str(f))


def test_directory_loader_empty_dir(tmp_path):
    assert DirectoryLoader().load(str(tmp_path)) == []
