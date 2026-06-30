"""
SmartHire — File Reader
Extracts plain text from uploaded resume files (PDF, DOCX, or TXT).
"""

import pdfplumber
import docx


def read_pdf(file) -> str:
    """Extract text from a PDF file object (e.g. Streamlit's UploadedFile)."""
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def read_docx(file) -> str:
    """Extract text from a DOCX file object."""
    doc = docx.Document(file)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs).strip()


def read_txt(file) -> str:
    """Extract text from a plain text file object."""
    return file.read().decode("utf-8").strip()


def extract_text_from_file(file) -> str:
    """
    Detect file type by extension and extract text accordingly.
    `file` is a Streamlit UploadedFile (has .name and is file-like).
    """
    filename = file.name.lower()

    if filename.endswith(".pdf"):
        return read_pdf(file)
    elif filename.endswith(".docx"):
        return read_docx(file)
    elif filename.endswith(".txt"):
        return read_txt(file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
