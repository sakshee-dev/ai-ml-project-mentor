import logging
import openml

from app.models.dataset import DatasetItem
from app.services.ranking_service import infer_size

logger = logging.getLogger(__name__)


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def search_openml_datasets(query: str, limit: int = 20) -> list[DatasetItem]:
    try:
        df = openml.datasets.list_datasets(output_format="dataframe")
    except Exception as exc:
        logger.exception("OpenML request failed: %s", exc)
        raise RuntimeError("OpenML connection failed") from exc

    q = query.lower().strip()
    mask = df["name"].astype(str).str.lower().str.contains(q, na=False)
    filtered = df[mask].head(limit)

    results = []
    for _, row in filtered.iterrows():
        instances = _safe_int(row.get("NumberOfInstances"))
        features = _safe_int(row.get("NumberOfFeatures"))
        did = row.get("did")

        results.append(
            DatasetItem(
                name=str(row.get("name", "")),
                source="OpenML",
                url=f"https://www.openml.org/search?type=data&id={did}",
                description=f"OpenML dataset id {did}",
                task_type=None,
                modality="tabular",
                difficulty="beginner" if instances and instances < 10000 else "intermediate",
                size=infer_size(instances),
                instance_count=instances,
                feature_count=features,
                tags=["openml", "ml", "tabular"],
            )
        )

    return results