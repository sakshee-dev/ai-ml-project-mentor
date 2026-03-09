from app.models.dataset import DatasetItem


def infer_size(instance_count: int | None) -> str | None:
    if instance_count is None:
        return None
    if instance_count < 10_000:
        return "small"
    if instance_count < 500_000:
        return "medium"
    return "large"


def score_dataset(
    dataset: DatasetItem,
    query: str,
    level: str | None = None,
    modality: str | None = None,
) -> float:
    score = 0.0

    q = query.lower().strip()
    name = (dataset.name or "").lower()
    desc = (dataset.description or "").lower()
    tags = [t.lower() for t in dataset.tags]

    if q in name:
        score += 4.0
    if q in desc:
        score += 2.0
    if any(q in tag for tag in tags):
        score += 1.5

    if modality and dataset.modality and modality.lower() in dataset.modality.lower():
        score += 2.0

    if level:
        level = level.lower()
        if level == "beginner":
            if dataset.size == "small":
                score += 2.0
            if dataset.difficulty == "beginner":
                score += 2.0
            if dataset.modality == "tabular":
                score += 1.0
        elif level == "intermediate":
            if dataset.size in {"small", "medium"}:
                score += 1.5
        elif level == "advanced":
            if dataset.size in {"medium", "large"}:
                score += 1.5

    return score