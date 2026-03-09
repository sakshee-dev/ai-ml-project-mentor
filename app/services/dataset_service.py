import json
from pathlib import Path


DATASET_FILE = Path("data/dataset_catalog.json")


def load_datasets():
    with open(DATASET_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def recommend_datasets(topic: str, difficulty: str, subtopic: str | None = None):
    datasets = load_datasets()

    results = []
    for dataset in datasets:
        if dataset["topic"].lower() != topic.lower():
            continue
        if dataset["difficulty"].lower() != difficulty.lower():
            continue
        if subtopic and dataset["subtopic"].lower() != subtopic.lower():
            continue
        results.append(dataset)

    return results