from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from app.models.dataset import (
    DatasetAnalytics,
    DatasetColumnProfile,
    DatasetRecommendation,
    DatasetScore,
    SearchIntent,
)

try:
    import openml
except ImportError as exc:
    raise ImportError(
        "openml is not installed. Run: pip install openml"
    ) from exc

logger = logging.getLogger('log_console_file')

class OpenMLService:
    def __init__(self, cache_directory: Optional[str] = None) -> None:
        self.logger = logger
        if cache_directory:
            openml.config.cache_directory = cache_directory

    # =========================================================
    # PUBLIC API
    # =========================================================

    def search_datasets(
        self,
        query: str,
        skill_level: str = "beginner",
        modality: str = "any",
        max_results: int = 10,
    ) -> Dict[str, Any]:
        intent = SearchIntent(
            query=query.strip(),
            skill_level=skill_level.strip().lower(),
            modality=modality.strip().lower(),
            source="openml",
            max_results=max_results,
        )

        self.logger.info(
            "[OpenML] search_datasets start | query=%s | skill_level=%s | modality=%s | max_results=%s",
            intent.query,
            intent.skill_level,
            intent.modality,
            intent.max_results,
        )

        raw_df = self._retrieve_candidates(intent.query, intent.max_results)
        self.logger.info("[OpenML] retrieved candidate rows=%s", len(raw_df))

        candidates = self._normalize_candidates(raw_df=raw_df, query=intent.query)
        self.logger.info("[OpenML] normalized candidates=%s", len(candidates))

        ranked = self._rank_candidates(candidates=candidates, intent=intent)
        self.logger.info("[OpenML] ranked candidates=%s", len(ranked))

        if ranked:
            self.logger.info(
                "[OpenML] top recommendation | id=%s | name=%s | final_score=%.4f",
                ranked[0].dataset_id,
                ranked[0].name,
                ranked[0].score.final,
            )
        else:
            self.logger.warning("[OpenML] no ranked candidates found")

        return {
            "query": intent.query,
            "source": "OpenML",
            "total_found": len(ranked),
            "recommended": ranked[0].to_dict() if ranked else None,
            "results": [item.to_dict() for item in ranked[: intent.max_results]],
        }

    def get_dataset_details(
        self,
        dataset_id: int,
        preview_rows: int = 10,
    ) -> Dict[str, Any]:
        self.logger.info("[OpenML] get_dataset_details start | dataset_id=%s", dataset_id)

        dataset = openml.datasets.get_dataset(dataset_id, download_data=True)

        X, y, _, _ = dataset.get_data(
            dataset_format="dataframe",
            target=dataset.default_target_attribute,
        )

        df = X.copy()

        if y is not None:
            target_name = dataset.default_target_attribute or "target"
            if target_name not in df.columns:
                df[target_name] = y

        self.logger.info(
            "[OpenML] dataset loaded | id=%s | name=%s | shape=(%s, %s)",
            dataset_id,
            dataset.name,
            df.shape[0],
            df.shape[1],
        )

        analytics = self._build_dataset_analytics(
            dataset=dataset,
            df=df,
            preview_rows=preview_rows,
        )

        return analytics.to_dict()

    def ask_about_dataset(
        self,
        dataset_id: int,
        question: str,
    ) -> Dict[str, Any]:
        self.logger.info(
            "[OpenML] ask_about_dataset start | dataset_id=%s | question=%s",
            dataset_id,
            question,
        )

        dataset = openml.datasets.get_dataset(dataset_id, download_data=True)

        X, y, _, _ = dataset.get_data(
            dataset_format="dataframe",
            target=dataset.default_target_attribute,
        )

        df = X.copy()
        if y is not None:
            target_name = dataset.default_target_attribute or "target"
            if target_name not in df.columns:
                df[target_name] = y

        q = question.strip().lower()

        if "target" in q:
            return {
                "question": question,
                "answer": {
                    "default_target_attribute": dataset.default_target_attribute,
                    "candidate_target_columns": self._detect_candidate_target_columns(
                        df=df,
                        default_target=dataset.default_target_attribute,
                    ),
                },
            }

        if "problem type" in q or "classification" in q or "regression" in q:
            problem_type = self._infer_problem_type_from_dataframe(
                df=df,
                default_target=dataset.default_target_attribute,
            )
            return {
                "question": question,
                "answer": {
                    "recommended_problem_type": problem_type,
                    "recommended_metrics": self._recommend_metrics(problem_type),
                },
            }

        if "xgboost" in q or "is this good for xgboost" in q:
            problem_type = self._infer_problem_type_from_dataframe(
                df=df,
                default_target=dataset.default_target_attribute,
            )
            return {
                "question": question,
                "answer": {
                    "xgboost_fit_explanation": self._explain_xgboost_fit(df, problem_type),
                },
            }

        if "missing" in q or "null" in q:
            missing = df.isna().sum().to_dict()
            missing_pct = {
                col: round((count / len(df)) * 100, 2) if len(df) else 0.0
                for col, count in missing.items()
            }
            return {
                "question": question,
                "answer": {
                    "missing_counts": missing,
                    "missing_percent": missing_pct,
                },
            }

        return {
            "question": question,
            "answer": {
                "dataset_name": dataset.name,
                "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
                "default_target_attribute": dataset.default_target_attribute,
                "candidate_target_columns": self._detect_candidate_target_columns(
                    df=df,
                    default_target=dataset.default_target_attribute,
                ),
            },
        }

    # =========================================================
    # RETRIEVAL
    # =========================================================

    def _retrieve_candidates(self, query: str, max_results: int) -> pd.DataFrame:
        """
        Strategy:
        1. exact/broad metadata matches
        2. if query is an algorithm term and exact matches are too few,
           augment with generally suitable supervised tabular datasets
        """
        self.logger.info("[OpenML] _retrieve_candidates start | query=%s", query)

        try:
            raw_df = openml.datasets.list_datasets(output_format="dataframe")
        except Exception as exc:
            self.logger.exception("[OpenML] Failed to retrieve OpenML datasets")
            raise RuntimeError(f"OpenML dataset retrieval failed: {exc}") from exc

        self.logger.info("[OpenML] list_datasets total rows=%s", len(raw_df))

        if raw_df.empty:
            return raw_df

        query_lower = query.lower().strip()

        searchable_columns = [
            col
            for col in raw_df.columns
            if col.lower()
            in {
                "name",
                "did",
                "version",
                "status",
                "format",
                "creator",
                "default_target_attribute",
            }
        ]

        mask = pd.Series(False, index=raw_df.index)

        for col in searchable_columns:
            values = raw_df[col].astype(str).str.lower()
            mask = mask | values.str.contains(re.escape(query_lower), na=False)

        direct_matches = raw_df[mask].copy()
        self.logger.info("[OpenML] direct metadata matches=%s", len(direct_matches))

        if self._is_algorithm_query(query_lower) and len(direct_matches) < max(5, max_results):
            self.logger.info(
                "[OpenML] algorithm-style query detected; augmenting candidate pool for query=%s",
                query_lower,
            )

            augmented = raw_df.copy()

            # Prefer supervised-style tabular datasets with reasonable metadata
            if "default_target_attribute" in augmented.columns:
                augmented = augmented[augmented["default_target_attribute"].notna()]

            if "NumberOfInstances" in augmented.columns:
                augmented = augmented[
                    augmented["NumberOfInstances"].fillna(0).astype(float).between(100, 500000)
                ]

            if "NumberOfFeatures" in augmented.columns:
                augmented = augmented[
                    augmented["NumberOfFeatures"].fillna(0).astype(float).between(3, 100)
                ]

            self.logger.info("[OpenML] augmented supervised/tabular candidates=%s", len(augmented))

            combined = pd.concat([direct_matches, augmented], ignore_index=True)
            combined = combined.drop_duplicates(subset=["did"]).reset_index(drop=True)

            self.logger.info("[OpenML] combined candidate pool=%s", len(combined))

            return combined

        direct_matches = direct_matches.sort_values(by=["did"], ascending=True).reset_index(drop=True)
        return direct_matches

    # =========================================================
    # NORMALIZATION
    # =========================================================

    def _normalize_candidates(self, raw_df: pd.DataFrame, query: str) -> List[DatasetRecommendation]:
        items: List[DatasetRecommendation] = []

        if raw_df.empty:
            return items

        self.logger.info("[OpenML] normalizing rows=%s", len(raw_df))

        for _, row in raw_df.iterrows():
            dataset_id = self._safe_int(row.get("did"))
            if dataset_id is None:
                continue

            name = self._safe_str(row.get("name")) or f"dataset_{dataset_id}"
            rows = self._safe_int(row.get("NumberOfInstances"))
            cols = self._safe_int(row.get("NumberOfFeatures"))
            default_target = self._safe_str(row.get("default_target_attribute"))

            recommendation = DatasetRecommendation(
                rank=0,
                dataset_id=dataset_id,
                name=name,
                source="OpenML",
                url=f"https://www.openml.org/d/{dataset_id}",
                description="",
                modality="tabular",
                task_type=self._infer_task_type_from_metadata(default_target),
                difficulty=self._estimate_difficulty(rows=rows, columns=cols),
                rows=rows,
                columns=cols,
                default_target_attribute=default_target,
                version=self._safe_int(row.get("version")),
                status=self._safe_str(row.get("status")),
                format=self._safe_str(row.get("format")),
                licence=self._safe_str(row.get("licence")),
                creator=self._safe_str(row.get("creator")),
                tags=self._build_tags(row),
                search_match_reason=self._build_search_match_reason(row=row, query=query),
            )

            items.append(recommendation)

        return items

    # =========================================================
    # RANKING
    # =========================================================

    def _rank_candidates(
        self,
        candidates: List[DatasetRecommendation],
        intent: SearchIntent,
    ) -> List[DatasetRecommendation]:
        ranked_items: List[DatasetRecommendation] = []

        self.logger.info(
            "[OpenML] ranking candidates=%s for query=%s skill=%s modality=%s",
            len(candidates),
            intent.query,
            intent.skill_level,
            intent.modality,
        )

        for item in candidates:
            item.score = DatasetScore(
                relevance=self._score_relevance(item, intent),
                beginner_friendly=self._score_beginner_friendliness(item, intent),
                model_fit=self._score_model_fit(item, intent),
                metadata_quality=self._score_metadata_quality(item),
                usability=self._score_usability(item),
            )

            item.score.final = round(
                (0.15 * item.score.relevance)
                + (0.30 * item.score.beginner_friendly)
                + (0.30 * item.score.model_fit)
                + (0.15 * item.score.metadata_quality)
                + (0.10 * item.score.usability),
                4,
            )

            ranked_items.append(item)

        ranked_items.sort(
            key=lambda x: (
                x.score.final,
                x.score.model_fit,
                x.score.beginner_friendly,
                x.score.relevance,
            ),
            reverse=True,
        )

        for idx, item in enumerate(ranked_items, start=1):
            item.rank = idx
            item.recommendation_reason = self._build_recommendation_reason(item, intent, ranked_items)

        self.logger.info(
            "[OpenML] ranking complete | top5=%s",
            [
                {
                    "rank": item.rank,
                    "id": item.dataset_id,
                    "name": item.name,
                    "final": item.score.final,
                }
                for item in ranked_items[:5]
            ],
        )

        return ranked_items

    def _score_relevance(self, item: DatasetRecommendation, intent: SearchIntent) -> float:
        query = intent.query.lower().strip()
        name = item.name.lower()
        tags = " ".join(item.tags).lower()
        target = (item.default_target_attribute or "").lower()

        score = 0.0

        # Lower direct-name dominance for algorithm queries
        if self._is_algorithm_query(query):
            if query in name:
                score += 0.20
            if query in tags:
                score += 0.10
            if item.modality == "tabular":
                score += 0.25
            if item.task_type in {"classification", "regression", "supervised"}:
                score += 0.25
            if target:
                score += 0.10
        else:
            if query in name:
                score += 0.55
            if query in tags:
                score += 0.20
            if query in target:
                score += 0.10

        return min(score, 1.0)

    def _score_beginner_friendliness(self, item: DatasetRecommendation, intent: SearchIntent) -> float:
        rows = item.rows or 0
        cols = item.columns or 0

        if intent.skill_level == "advanced":
            score = 0.55
            if rows >= 50000:
                score += 0.15
            if cols >= 10:
                score += 0.10
            if item.default_target_attribute:
                score += 0.10
            return max(0.0, min(score, 1.0))

        if intent.skill_level == "intermediate":
            score = 0.45
            if 1000 <= rows <= 500000:
                score += 0.20
            if 5 <= cols <= 100:
                score += 0.15
            if item.default_target_attribute:
                score += 0.10
            return max(0.0, min(score, 1.0))

        score = 0.35

        if 100 <= rows <= 100000:
            score += 0.35
        elif 100000 < rows <= 500000:
            score += 0.20
        elif rows > 1000000:
            score -= 0.15

        if 3 <= cols <= 50:
            score += 0.20
        elif cols > 200:
            score -= 0.10

        if item.default_target_attribute:
            score += 0.10

        return max(0.0, min(score, 1.0))

    def _score_model_fit(self, item: DatasetRecommendation, intent: SearchIntent) -> float:
        query = intent.query.lower().strip()
        score = 0.35

        if self._is_algorithm_query(query):
            if item.modality == "tabular":
                score += 0.30
            if item.task_type in {"classification", "regression", "supervised"}:
                score += 0.20
            if item.columns and 3 <= item.columns <= 100:
                score += 0.10
            if item.rows and 100 <= item.rows <= 500000:
                score += 0.05

            return max(0.0, min(score, 1.0))

        return 0.50

    def _score_metadata_quality(self, item: DatasetRecommendation) -> float:
        score = 0.2

        if item.name:
            score += 0.20
        if item.rows is not None:
            score += 0.20
        if item.columns is not None:
            score += 0.20
        if item.default_target_attribute:
            score += 0.20
        if item.url:
            score += 0.20

        return min(score, 1.0)

    def _score_usability(self, item: DatasetRecommendation) -> float:
        score = 0.3

        if item.url:
            score += 0.2
        if item.rows and item.columns:
            score += 0.2
        if item.format:
            score += 0.1
        if item.status:
            score += 0.1
        if item.default_target_attribute:
            score += 0.1

        return min(score, 1.0)

    def _build_recommendation_reason(
        self,
        item: DatasetRecommendation,
        intent: SearchIntent,
        all_items: List[DatasetRecommendation],
    ) -> str:
        query = intent.query.lower().strip()

        parts: List[str] = []

        if self._is_algorithm_query(query):
            parts.append(f"{item.name} is ranked highly because it is a structured tabular dataset")
            if item.task_type:
                parts.append(f"with a likely {item.task_type} setup")
            if item.rows and item.columns:
                parts.append(f"and about {item.rows} rows and {item.columns} columns")
            if item.default_target_attribute:
                parts.append(f"The default target is '{item.default_target_attribute}'")
        else:
            parts.append(
                f"{item.name} ranked well for your query based on relevance, usability, and learning value"
            )

        if len(all_items) > 1 and item.rank == 1:
            parts.append(f"It ranked above {len(all_items) - 1} other OpenML candidates")

        return " ".join(parts) + "."

    # =========================================================
    # DATASET EXPANSION / ANALYTICS
    # =========================================================

    def _build_dataset_analytics(
        self,
        dataset: Any,
        df: pd.DataFrame,
        preview_rows: int,
    ) -> DatasetAnalytics:
        row_count, col_count = df.shape

        columns: List[DatasetColumnProfile] = []
        for col in df.columns:
            series = df[col]
            missing_count = int(series.isna().sum())
            missing_percent = round((missing_count / row_count) * 100, 2) if row_count else 0.0
            unique_count = int(series.nunique(dropna=True))
            sample_values = self._safe_sample_values(series, limit=5)

            columns.append(
                DatasetColumnProfile(
                    name=col,
                    dtype=str(series.dtype),
                    missing_count=missing_count,
                    missing_percent=missing_percent,
                    unique_count=unique_count,
                    sample_values=sample_values,
                )
            )

        candidate_targets = self._detect_candidate_target_columns(
            df=df,
            default_target=dataset.default_target_attribute,
        )
        numeric_summary = self._build_numeric_summary(df)
        categorical_summary = self._build_categorical_summary(df)
        duplicate_rows = int(df.duplicated().sum())

        problem_type = self._infer_problem_type_from_dataframe(
            df=df,
            default_target=dataset.default_target_attribute,
        )
        recommended_metrics = self._recommend_metrics(problem_type)
        xgboost_fit = self._explain_xgboost_fit(df=df, problem_type=problem_type)

        summary = self._build_plain_english_summary(
            dataset_name=dataset.name,
            rows=row_count,
            columns=col_count,
            target=dataset.default_target_attribute,
            problem_type=problem_type,
        )

        self.logger.info(
            "[OpenML] analytics built | id=%s | name=%s | rows=%s | cols=%s | candidate_targets=%s | problem_type=%s",
            dataset.dataset_id,
            dataset.name,
            row_count,
            col_count,
            candidate_targets,
            problem_type,
        )

        return DatasetAnalytics(
            dataset_id=int(dataset.dataset_id),
            name=dataset.name,
            source="OpenML",
            url=f"https://www.openml.org/d/{dataset.dataset_id}",
            shape={"rows": int(row_count), "columns": int(col_count)},
            columns=columns,
            candidate_target_columns=candidate_targets,
            numeric_summary=numeric_summary,
            categorical_summary=categorical_summary,
            preview_rows=df.head(preview_rows).to_dict(orient="records"),
            duplicate_rows=duplicate_rows,
            plain_english_summary=summary,
            recommended_problem_type=problem_type,
            recommended_metrics=recommended_metrics,
            xgboost_fit_explanation=xgboost_fit,
        )

    # =========================================================
    # HELPERS
    # =========================================================

    def _is_algorithm_query(self, query: str) -> bool:
        algorithm_terms = {
            "xgboost",
            "random forest",
            "lightgbm",
            "catboost",
            "svm",
            "logistic regression",
            "linear regression",
            "decision tree",
        }
        return query in algorithm_terms

    def _build_tags(self, row: pd.Series) -> List[str]:
        tags = set()

        name = self._safe_str(row.get("name"))
        creator = self._safe_str(row.get("creator"))
        fmt = self._safe_str(row.get("format"))
        default_target = self._safe_str(row.get("default_target_attribute"))

        for value in [name, creator, fmt, default_target]:
            if value:
                parts = re.split(r"[\s_\-:/]+", value.lower())
                tags.update([part for part in parts if part])

        tags.add("openml")
        tags.add("tabular")

        return sorted(tags)

    def _build_search_match_reason(self, row: pd.Series, query: str) -> str:
        query_lower = query.lower().strip()
        name = self._safe_str(row.get("name"), "").lower()
        creator = self._safe_str(row.get("creator"), "").lower()
        target = self._safe_str(row.get("default_target_attribute"), "").lower()

        matched_fields = []
        if query_lower in name:
            matched_fields.append("name")
        if query_lower in creator:
            matched_fields.append("creator")
        if query_lower in target:
            matched_fields.append("default_target_attribute")

        if matched_fields:
            return f"Matched in {', '.join(matched_fields)}"

        return "Broad OpenML metadata match"

    def _infer_task_type_from_metadata(self, default_target_attribute: Optional[str]) -> Optional[str]:
        if default_target_attribute:
            return "supervised"
        return None

    def _estimate_difficulty(
        self,
        rows: Optional[int],
        columns: Optional[int],
    ) -> str:
        if rows is None or columns is None:
            return "unknown"

        if rows <= 100000 and columns <= 50:
            return "beginner"
        if rows <= 500000 and columns <= 200:
            return "intermediate"
        return "advanced"

    def _detect_candidate_target_columns(
        self,
        df: pd.DataFrame,
        default_target: Optional[str],
    ) -> List[str]:
        candidates: List[str] = []

        if default_target and default_target in df.columns:
            candidates.append(default_target)

        heuristics = ["target", "label", "class", "y", "output", "response"]
        for col in df.columns:
            col_lower = col.lower()
            if any(token == col_lower or token in col_lower for token in heuristics):
                if col not in candidates:
                    candidates.append(col)

        return candidates[:5]

    def _infer_problem_type_from_dataframe(
        self,
        df: pd.DataFrame,
        default_target: Optional[str],
    ) -> Optional[str]:
        target_col = None

        if default_target and default_target in df.columns:
            target_col = default_target
        else:
            candidates = self._detect_candidate_target_columns(df, default_target)
            if candidates:
                target_col = candidates[0]

        if not target_col:
            return None

        series = df[target_col].dropna()
        if series.empty:
            return None

        if pd.api.types.is_numeric_dtype(series):
            unique_count = series.nunique(dropna=True)
            if unique_count <= 20:
                return "classification"
            return "regression"

        return "classification"

    def _recommend_metrics(self, problem_type: Optional[str]) -> List[str]:
        if problem_type == "classification":
            return ["accuracy", "f1", "precision", "recall", "roc_auc"]
        if problem_type == "regression":
            return ["mae", "rmse", "r2"]
        return []

    def _explain_xgboost_fit(self, df: pd.DataFrame, problem_type: Optional[str]) -> str:
        numeric_cols = int(df.select_dtypes(include=["number"]).shape[1])
        categorical_cols = int(df.select_dtypes(exclude=["number"]).shape[1])

        parts = [
            "XGBoost is usually a strong choice for structured tabular data"
        ]

        parts.append(
            f"This dataset has {df.shape[0]} rows, {df.shape[1]} columns, "
            f"{numeric_cols} numeric columns, and {categorical_cols} non-numeric columns"
        )

        if problem_type == "classification":
            parts.append("It looks suitable for a classification workflow")
        elif problem_type == "regression":
            parts.append("It looks suitable for a regression workflow")
        else:
            parts.append("You may need to confirm the target column before training")

        if categorical_cols > 0:
            parts.append("Some preprocessing or encoding may be needed for categorical features")

        return ". ".join(parts) + "."

    def _build_numeric_summary(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        numeric_df = df.select_dtypes(include=["number"])
        if numeric_df.empty:
            return {}

        desc = numeric_df.describe().transpose()
        result: Dict[str, Dict[str, Any]] = {}

        for col, row in desc.iterrows():
            result[col] = {
                "count": self._safe_float(row.get("count")),
                "mean": self._safe_float(row.get("mean")),
                "std": self._safe_float(row.get("std")),
                "min": self._safe_float(row.get("min")),
                "25%": self._safe_float(row.get("25%")),
                "50%": self._safe_float(row.get("50%")),
                "75%": self._safe_float(row.get("75%")),
                "max": self._safe_float(row.get("max")),
            }

        return result

    def _build_categorical_summary(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        cat_df = df.select_dtypes(exclude=["number"])
        result: Dict[str, Dict[str, Any]] = {}

        if cat_df.empty:
            return result

        for col in cat_df.columns:
            series = df[col]
            top_values = (
                series.astype(str)
                .value_counts(dropna=False)
                .head(5)
                .to_dict()
            )

            result[col] = {
                "unique_count": int(series.nunique(dropna=True)),
                "top_values": top_values,
            }

        return result

    def _build_plain_english_summary(
        self,
        dataset_name: str,
        rows: int,
        columns: int,
        target: Optional[str],
        problem_type: Optional[str],
    ) -> str:
        parts = [
            f"{dataset_name} has {rows} rows and {columns} columns"
        ]

        if target:
            parts.append(f"The default target attribute is '{target}'")

        if problem_type:
            parts.append(f"It most likely fits a {problem_type} task")

        parts.append("This makes it suitable for initial EDA and baseline modeling")

        return ". ".join(parts) + "."

    def _safe_sample_values(self, series: pd.Series, limit: int = 5) -> List[Any]:
        values = series.dropna().head(limit).tolist()
        cleaned: List[Any] = []

        for value in values:
            if pd.isna(value):
                continue
            if hasattr(value, "item"):
                try:
                    value = value.item()
                except Exception:
                    pass
            cleaned.append(value)

        return cleaned

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        try:
            return int(value)
        except Exception:
            return None

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        try:
            return round(float(value), 6)
        except Exception:
            return None

    @staticmethod
    def _safe_str(value: Any, default: Optional[str] = None) -> Optional[str]:
        if value is None:
            return default
        try:
            if pd.isna(value):
                return default
        except Exception:
            pass

        text = str(value).strip()
        return text if text else default