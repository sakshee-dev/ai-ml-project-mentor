import logging
import openml
from app.models.dataset import DatasetItem
from app.services.ranking_service import infer_size

logger = logging.getLogger(__name__)

QUERY_EXPANSIONS = {
    "nlp": ["nlp", "text", "tweet", "tweets", "review", "reviews", "sentiment", "spam", "language", "qa", "question"],
    "cv": ["cv", "image", "vision", "photo", "photos", "face", "xray", "mnist"],
    "tabular": ["tabular", "classification", "regression", "structured"],
}

def _get_search_terms(query: str) -> list[str]:
    q = query.lower().strip()
    return QUERY_EXPANSIONS.get(q, [q])


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
        return []

    search_terms = _get_search_terms(query)
    # Search only by dataset name for now, but with expanded terms
    name_series = df["name"].astype(str).str.lower()
    mask = name_series.str.contains(search_terms[0], na=False)
    for term in search_terms[1:]:
        mask = mask | name_series.str.contains(term, na=False)

    filtered = df[mask].head(limit)
    logger.info("Matched OpenML dataset names: %s", filtered["name"].tolist())

    logger.info(
        "OpenML query='%s' expanded_terms=%s matched_rows=%s",
        query,
        search_terms,
        len(filtered),
    )

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
    logger.info("OpenML returned %s results for query='%s'", len(results), query)
    return results