import fitz  
import docx
import re
from typing import List
from fastapi import UploadFile

async def extract_references_from_file(file: UploadFile) -> List[str]:
    content = await file.read()

    if file.filename.lower().endswith('.pdf'):
        return extract_from_pdf(content)
    elif file.filename.lower().endswith('.docx'):
        return extract_from_docx(content)
    else:
        return []

def extract_from_pdf(file_bytes: bytes) -> List[str]:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text = "\n".join([page.get_text() for page in doc])
    return extract_references_from_text(full_text)

def extract_from_docx(file_bytes: bytes) -> List[str]:
    from io import BytesIO
    doc = docx.Document(BytesIO(file_bytes))
    full_text = "\n".join([para.text for para in doc.paragraphs])
    return extract_references_from_text(full_text)

def extract_references_from_text(text: str) -> List[str]:
    # Basic heuristic: look for 'References' or 'Bibliography'
    match = re.search(r'\b(References|Bibliography)\b', text, re.IGNORECASE)
    if match:
        references_section = text[match.end():]
    else:
        references_section = text

    # Split into individual references
    refs = re.split(r'\n\d*\.\s|\n(?=[A-Z][a-z]+,)', references_section)
    cleaned_refs = [r.strip() for r in refs if len(r.strip()) > 10]
    return cleaned_refs
