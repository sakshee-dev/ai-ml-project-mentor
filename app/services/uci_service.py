from app.models.dataset import DatasetItem

UCI_DATASETS = [
    DatasetItem(
        name="Iris",
        source="UCI",
        url="https://archive.ics.uci.edu/dataset/53/iris",
        description="Classic flower classification dataset.",
        task_type="classification",
        modality="tabular",
        difficulty="beginner",
        size="small",
        instance_count=150,
        feature_count=4,
        tags=["uci", "classification", "tabular", "beginner"],
    ),
    DatasetItem(
        name="Heart Disease",
        source="UCI",
        url="https://archive.ics.uci.edu/dataset/45/heart+disease",
        description="Clinical dataset for heart disease prediction.",
        task_type="classification",
        modality="tabular",
        difficulty="intermediate",
        size="small",
        instance_count=303,
        feature_count=13,
        tags=["uci", "classification", "health", "tabular"],
    ),
    DatasetItem(
        name="Wine Quality",
        source="UCI",
        url="https://archive.ics.uci.edu/dataset/186/wine+quality",
        description="Predict wine quality from physicochemical tests.",
        task_type="classification/regression",
        modality="tabular",
        difficulty="beginner",
        size="small",
        instance_count=4898,
        feature_count=12,
        tags=["uci", "classification", "regression", "tabular"],
    ),
    DatasetItem(
        name="SMS Spam Collection",
        source="UCI",
        url="https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
        description="Text dataset for spam detection.",
        task_type="classification",
        modality="text",
        difficulty="beginner",
        size="small",
        tags=["uci", "nlp", "text", "classification"],
    ),
]


def search_uci_datasets(query: str, limit: int = 20) -> list[DatasetItem]:
    q = query.lower().strip()
    results = []

    for item in UCI_DATASETS:
        haystack = f"{item.name} {item.description or ''} {' '.join(item.tags)}".lower()
        if q in haystack:
            results.append(item)

    return results[:limit]