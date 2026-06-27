from pathlib import Path

from app.services.ingestion import parse_document


def extract_text(file_path: Path) -> str:
    return parse_document(file_path).text
