# Backend

FastAPI backend for the tutor MVP.

## Upload flow

- `POST /documents/upload` accepts a file plus a `domain` field.
- Supported types for the first pass: `.txt`, `.md`, `.pdf`, `.docx`.
- Files are stored under `backend/data/uploads/` and manifests are written to `backend/data/manifests/`.
- Example:

```bash
curl -F "file=@your-dsa-book.pdf" -F "domain=Data Structures and Algorithms" http://127.0.0.1:8000/documents/upload
```

## Run

```bash
uvicorn app.main:app --reload
```
