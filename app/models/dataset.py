from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

@dataclass
class SearchIntent:
    query: str
    skill_level: str = "beginner"
    modality: str = "any"
    source: str = "openml"
    max_results: int = 10

@dataclass
class DatasetScore:
    relevance: float = 0.0
    beginner_friendly: float = 0.0
    model_fit: float = 0.0
    metadata_quality: float = 0.0
    usability: float = 0.0
    final: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

@dataclass
class DatasetRecommendation:
    rank: int
    dataset_id: int
    name: str
    source: str
    url: str
    description: str = ""
    modality: str = "tabular"
    task_type: Optional[str] = None
    difficulty: Optional[str] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    default_target_attribute: Optional[str] = None
    version: Optional[int] = None
    status: Optional[str] = None
    format: Optional[str] = None
    licence: Optional[str] = None
    creator: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    search_match_reason: str = ""
    recommendation_reason: str = ""
    score: DatasetScore = field(default_factory=DatasetScore)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["score"] = self.score.to_dict()
        return data

@dataclass
class DatasetColumnProfile:
    name: str
    dtype: str
    missing_count: int
    missing_percent: float
    unique_count: int
    sample_values: List[Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class DatasetAnalytics:
    dataset_id: int
    name: str
    source: str
    url: str
    shape: Dict[str, int]
    columns: List[DatasetColumnProfile]
    candidate_target_columns: List[str]
    numeric_summary: Dict[str, Dict[str, Any]]
    categorical_summary: Dict[str, Dict[str, Any]]
    preview_rows: List[Dict[str, Any]]
    duplicate_rows: int
    plain_english_summary: str
    recommended_problem_type: Optional[str]
    recommended_metrics: List[str]
    xgboost_fit_explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "source": self.source,
            "url": self.url,
            "shape": self.shape,
            "columns": [col.to_dict() for col in self.columns],
            "candidate_target_columns": self.candidate_target_columns,
            "numeric_summary": self.numeric_summary,
            "categorical_summary": self.categorical_summary,
            "preview_rows": self.preview_rows,
            "duplicate_rows": self.duplicate_rows,
            "plain_english_summary": self.plain_english_summary,
            "recommended_problem_type": self.recommended_problem_type,
            "recommended_metrics": self.recommended_metrics,
            "xgboost_fit_explanation": self.xgboost_fit_explanation,
        }