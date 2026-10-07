from pydantic import BaseModel, Field
from typing import Optional

class PageRecord(BaseModel):
    document_id: str
    document_name: str
    file_type: str = "pdf"
    page_number: int
    text: str
    source_path: str
    metadata: Optional[dict] = Field(default_factory=dict)

class ChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    text: str
    page_start: int
    page_end: int
    section: Optional[str] = None
    source_path: str
    chunk_index: int
