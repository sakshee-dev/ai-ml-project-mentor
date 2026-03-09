from pydantic import BaseModel, Field
from typing import List, Optional


class DatasetItem(BaseModel):
    name: str
    source: str
    url: Optional[str] = None
    description: Optional[str] = None
    task_type: Optional[str] = None
    modality: Optional[str] = None
    difficulty: Optional[str] = None
    size: Optional[str] = None
    instance_count: Optional[int] = None
    feature_count: Optional[int] = None
    tags: List[str] = Field(default_factory=list)
    score: float = 0.0