import logging

from app.models.dataset import DatasetItem
from app.services.openml_service import search_openml_datasets
from app.services.huggingface_service import search_huggingface_datasets
from app.services.uci_service import search_uci_datasets
from app.services.kaggle_service import search_kaggle_datasets
from app.services.ranking_service import score_dataset

logger = logging.getLogger(__name__)


def search_all_datasets(
    query: str,
    level: str | None = None,
    modality: str | None = None,
    source: str | None = None,
    limit: int = 20,
) -> tuple[list[DatasetItem], list[str]]:
    results: list[DatasetItem] = []
    warnings: list[str] = []

    source_normalized = source.lower() if source else None

    providers = []
    if source_normalized in (None, "openml"):
        providers.append(("OpenML", search_openml_datasets))
    if source_normalized in (None, "huggingface"):
        providers.append(("HuggingFace", search_huggingface_datasets))
    if source_normalized in (None, "uci"):
        providers.append(("UCI", search_uci_datasets))
    if source_normalized in (None, "kaggle"):
        providers.append(("Kaggle", search_kaggle_datasets))

    for provider_name, provider_fn in providers:
        try:
            provider_results = provider_fn(query=query, limit=limit)
            results.extend(provider_results)
        except Exception as exc:
            logger.exception("%s search failed: %s", provider_name, exc)
            warnings.append(f"{provider_name} is unavailable right now.")

    for item in results:
        item.score = score_dataset(
            dataset=item,
            query=query,
            level=level,
            modality=modality,
        )

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:limit], warnings