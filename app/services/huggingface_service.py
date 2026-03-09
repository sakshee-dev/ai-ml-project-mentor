from huggingface_hub import list_datasets
from app.models.dataset import DatasetItem


def search_huggingface_datasets(query: str, limit: int = 20) -> list[DatasetItem]:
    results = []

    try:
        datasets = list_datasets(search=query, limit=limit, full=True)
    except Exception:
        return results

    for ds in datasets:
        tags = list(ds.tags) if getattr(ds, "tags", None) else []
        lowered_tags = [t.lower() for t in tags]

        if any("text" in t or "nlp" in t for t in lowered_tags):
            modality = "text"
        elif any("image" in t or "vision" in t for t in lowered_tags):
            modality = "image"
        elif any("audio" in t for t in lowered_tags):
            modality = "audio"
        else:
            modality = "tabular"

        results.append(
            DatasetItem(
                name=ds.id,
                source="HuggingFace",
                url=f"https://huggingface.co/datasets/{ds.id}",
                description=getattr(ds, "description", None),
                task_type=None,
                modality=modality,
                difficulty=None,
                size=None,
                tags=tags[:15],
            )
        )

    return results