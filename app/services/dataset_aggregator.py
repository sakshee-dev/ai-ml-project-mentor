from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from app.services.openml_service import OpenMLService

logger = logging.getLogger('log_console_file')

class DatasetAggregatorService:
    def __init__(self) -> None:
        try:
            self.logger = logger
            self.openml_service = OpenMLService()
            self.logger.info("[Aggregator] DatasetAggregatorService initialized successfully")
        except Exception as exc:
            self.logger.exception("[Aggregator] Failed to initialize DatasetAggregatorService: %s", exc)
            raise

    def search_all_datasets(
        self,
        query: str,
        level: Optional[str] = None,
        modality: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 20,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        results: List[Dict[str, Any]] = []
        warnings: List[str] = []

        try:
            source_normalized = source.lower().strip() if source else None
            level_normalized = level.lower().strip() if level else "beginner"
            modality_normalized = modality.lower().strip() if modality else "any"

            self.logger.info(
                "[Aggregator] search_all_datasets start | query=%s | level=%s | modality=%s | source=%s | limit=%s",
                query,
                level_normalized,
                modality_normalized,
                source_normalized,
                limit,
            )

            providers: List[Tuple[str, Any]] = []

            if source_normalized in (None, "openml"):
                providers.append(("OpenML", self.openml_service))

            if not providers:
                warnings.append("No valid dataset source was selected.")
                self.logger.warning("[Aggregator] No valid providers selected")
                return [], warnings

            self.logger.info("[Aggregator] active providers=%s", [name for name, _ in providers])

            for provider_name, provider_service in providers:
                try:
                    self.logger.info("[Aggregator] Calling provider=%s", provider_name)

                    provider_response = provider_service.search_datasets(
                        query=query,
                        skill_level=level_normalized,
                        modality=modality_normalized,
                        max_results=limit,
                    )

                    provider_results = provider_response.get("results", [])
                    self.logger.info(
                        "[Aggregator] provider=%s returned result_count=%s",
                        provider_name,
                        len(provider_results),
                    )

                    results.extend(provider_results)

                except Exception as exc:
                    self.logger.exception("[Aggregator] Provider failure | provider=%s | error=%s", provider_name, exc)
                    warnings.append(f"{provider_name} is unavailable right now.")

            results.sort(
                key=lambda item: item.get("score", {}).get("final", 0.0),
                reverse=True,
            )

            self.logger.info("[Aggregator] merged result_count=%s", len(results))
            self.logger.debug("[Aggregator] warnings=%s", warnings)

            if results:
                self.logger.info(
                    "[Aggregator] top_results=%s",
                    [
                        {
                            "rank": idx + 1,
                            "dataset_id": item.get("dataset_id"),
                            "name": item.get("name"),
                            "score": item.get("score", {}).get("final", 0.0),
                        }
                        for idx, item in enumerate(results[:5])
                    ],
                )

            return results[:limit], warnings

        except Exception as exc:
            self.logger.exception("[Aggregator] search_all_datasets fatal error: %s", exc)
            warnings.append("Unexpected error while aggregating datasets.")
            return [], warnings

    def get_dataset_details(
        self,
        dataset_id: int,
        source: str = "openml",
        preview_rows: int = 10,
    ) -> Dict[str, Any]:
        try:
            source_normalized = source.lower().strip()
            self.logger.info(
                "[Aggregator] get_dataset_details start | dataset_id=%s | source=%s | preview_rows=%s",
                dataset_id,
                source_normalized,
                preview_rows,
            )

            if source_normalized == "openml":
                result = self.openml_service.get_dataset_details(
                    dataset_id=dataset_id,
                    preview_rows=preview_rows,
                )
                self.logger.info("[Aggregator] get_dataset_details success | dataset_id=%s", dataset_id)
                return result

            raise ValueError(f"Unsupported source: {source}")

        except Exception as exc:
            self.logger.exception("[Aggregator] get_dataset_details failed | dataset_id=%s | error=%s", dataset_id, exc)
            raise

    def ask_about_dataset(
        self,
        dataset_id: int,
        question: str,
        source: str = "openml",
    ) -> Dict[str, Any]:
        try:
            source_normalized = source.lower().strip()
            self.logger.info(
                "[Aggregator] ask_about_dataset start | dataset_id=%s | source=%s | question=%s",
                dataset_id,
                source_normalized,
                question,
            )

            if source_normalized == "openml":
                result = self.openml_service.ask_about_dataset(
                    dataset_id=dataset_id,
                    question=question,
                )
                self.logger.info("[Aggregator] ask_about_dataset success | dataset_id=%s", dataset_id)
                return result

            raise ValueError(f"Unsupported source: {source}")

        except Exception as exc:
            self.logger.exception("[Aggregator] ask_about_dataset failed | dataset_id=%s | error=%s", dataset_id, exc)
            raise