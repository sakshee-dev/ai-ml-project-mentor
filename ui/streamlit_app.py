from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st

from app.services.dataset_aggregator import DatasetAggregatorService
from utils.global_variables import GlobalVariables
from utils.logger_setup import setup_logging


PROJECT_ROOT = Path(GlobalVariables.main_path).resolve()
LOGGER_CONFIG_PATH = Path(GlobalVariables.log_config_path).resolve()

setup_logging(
    project_root=PROJECT_ROOT,
    config_path=LOGGER_CONFIG_PATH,
    logger_name="log_console_file",
)

logger = logging.getLogger("log_console_file")
logger.info("[UI] Streamlit process started")


def initialize_state() -> None:
    try:
        defaults = {
            "search_results": [],
            "search_warnings": [],
            "selected_dataset_id": None,
            "selected_dataset_source": "openml",
            "selected_dataset_name": None,
            "dataset_details": None,
            "dataset_qa_response": None,
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
        logger.info("[UI] Session state initialized")
    except Exception as exc:
        logger.exception("[UI] initialize_state failed: %s", exc)
        raise


def make_dataframe_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert list/dict/object-heavy columns to strings
    so Streamlit/PyArrow can render them safely.
    """
    try:
        safe_df = df.copy()

        for col in safe_df.columns:
            safe_df[col] = safe_df[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False)
                if isinstance(x, (dict, list))
                else x
            )

        return safe_df

    except Exception as exc:
        logger.exception("[UI] make_dataframe_safe failed: %s", exc)
        return df.astype(str)


def render_search_results(results: list[dict]) -> None:
    try:
        logger.info("[UI] Rendering search results | count=%s", len(results))
        st.success(f"Found {len(results)} datasets")

        if st.session_state.selected_dataset_id:
            st.info(
                f"Selected dataset: {st.session_state.selected_dataset_name} "
                f"(ID: {st.session_state.selected_dataset_id})"
            )

        for item in results:
            with st.container():
                col1, col2 = st.columns([5, 1])

                with col1:
                    st.subheader(item.get("name", "Unknown Dataset"))
                    st.write(item.get("description") or "No description available.")
                    st.markdown(f"**Source:** {item.get('source', 'Unknown')}")
                    st.markdown(f"**Modality:** {item.get('modality', 'Unknown')}")
                    st.markdown(f"**Difficulty:** {item.get('difficulty', 'Unknown')}")

                    rows = item.get("rows")
                    columns = item.get("columns")
                    size_text = f"{rows or 'Unknown'} rows × {columns or 'Unknown'} columns"
                    st.markdown(f"**Size:** {size_text}")

                    score_final = item.get("score", {}).get("final", 0.0)
                    st.markdown(f"**Score:** {score_final:.2f}")

                    default_target = item.get("default_target_attribute")
                    if default_target:
                        st.markdown(f"**Default Target:** {default_target}")

                    recommendation_reason = item.get("recommendation_reason")
                    if recommendation_reason:
                        st.info(recommendation_reason)

                    tags = item.get("tags", [])
                    if tags:
                        st.write("**Tags:**", ", ".join(tags[:8]))

                    url = item.get("url")
                    if url:
                        st.markdown(f"[Open Dataset]({url})")

                with col2:
                    dataset_id = item.get("dataset_id")
                    dataset_source = item.get("source", "OpenML").lower()
                    dataset_name = item.get("name", "Unknown Dataset")
                    is_selected = st.session_state.selected_dataset_id == dataset_id

                    if is_selected:
                        st.success("Selected")

                    if st.button(
                        "Select" if not is_selected else "Reload",
                        key=f"select_{dataset_source}_{dataset_id}",
                        width="stretch",
                    ):
                        try:
                            logger.info(
                                "[UI] Dataset selected | id=%s | name=%s | source=%s",
                                dataset_id,
                                dataset_name,
                                dataset_source,
                            )

                            st.session_state.selected_dataset_id = dataset_id
                            st.session_state.selected_dataset_source = dataset_source
                            st.session_state.selected_dataset_name = dataset_name
                            st.session_state.dataset_qa_response = None

                            aggregator = DatasetAggregatorService()

                            with st.spinner("Loading dataset details..."):
                                details = aggregator.get_dataset_details(
                                    dataset_id=dataset_id,
                                    source=dataset_source,
                                    preview_rows=10,
                                )
                                st.session_state.dataset_details = details

                            logger.info("[UI] Dataset details loaded | dataset_id=%s", dataset_id)
                            st.rerun()

                        except Exception as exc:
                            logger.exception("[UI] Dataset select/load failed | dataset_id=%s | error=%s", dataset_id, exc)
                            st.error(f"Failed to load dataset details: {exc}")

                st.divider()

        st.subheader("Tabular View")
        safe_df = make_dataframe_safe(pd.DataFrame(results))
        st.dataframe(safe_df, width="stretch")

    except Exception as exc:
        logger.exception("[UI] render_search_results failed: %s", exc)
        st.error(f"Failed to render search results: {exc}")


def render_dataset_details(details: dict) -> None:
    try:
        logger.info("[UI] Rendering dataset details | name=%s", details.get("name"))
        st.header("Selected Dataset Analysis")
        st.subheader(details.get("name", "Dataset"))

        plain_summary = details.get("plain_english_summary")
        if plain_summary:
            st.write(plain_summary)

        shape = details.get("shape", {})
        rows = shape.get("rows", "Unknown")
        cols = shape.get("columns", "Unknown")
        st.markdown(f"**Shape:** {rows} rows × {cols} columns")

        recommended_problem_type = details.get("recommended_problem_type")
        if recommended_problem_type:
            st.markdown(f"**Recommended Problem Type:** {recommended_problem_type}")

        recommended_metrics = details.get("recommended_metrics", [])
        if recommended_metrics:
            st.markdown(f"**Recommended Metrics:** {', '.join(recommended_metrics)}")

        xgboost_fit_explanation = details.get("xgboost_fit_explanation")
        if xgboost_fit_explanation:
            st.info(xgboost_fit_explanation)

        candidate_targets = details.get("candidate_target_columns", [])
        if candidate_targets:
            st.markdown(f"**Candidate Target Columns:** {', '.join(candidate_targets)}")

        st.subheader("Preview Rows")
        preview_rows = details.get("preview_rows", [])
        if preview_rows:
            preview_df = make_dataframe_safe(pd.DataFrame(preview_rows))
            st.dataframe(preview_df, width="stretch")
        else:
            st.write("No preview rows available.")

        st.subheader("Column Profiles")
        columns_data = details.get("columns", [])
        if columns_data:
            columns_df = make_dataframe_safe(pd.DataFrame(columns_data))
            st.dataframe(columns_df, width="stretch")

        st.subheader("Numeric Summary")
        numeric_summary = details.get("numeric_summary", {})
        if numeric_summary:
            numeric_df = make_dataframe_safe(pd.DataFrame(numeric_summary).T)
            st.dataframe(numeric_df, width="stretch")
        else:
            st.write("No numeric columns found.")

        st.subheader("Categorical Summary")
        categorical_summary = details.get("categorical_summary", {})
        if categorical_summary:
            cat_rows = []
            for col_name, col_summary in categorical_summary.items():
                cat_rows.append(
                    {
                        "column": col_name,
                        "unique_count": col_summary.get("unique_count"),
                        "top_values": col_summary.get("top_values"),
                    }
                )
            cat_df = make_dataframe_safe(pd.DataFrame(cat_rows))
            st.dataframe(cat_df, width="stretch")
        else:
            st.write("No categorical summary available.")

    except Exception as exc:
        logger.exception("[UI] render_dataset_details failed: %s", exc)
        st.error(f"Failed to render dataset details: {exc}")


def render_dataset_qa(dataset_id: int, dataset_source: str) -> None:
    try:
        st.header("Ask About This Dataset")

        question = st.text_input(
            "Ask a question about the selected dataset",
            value="What is the target column?",
            key="dataset_question_input",
        )

        if st.button("Ask Mentor", key="ask_dataset_question"):
            try:
                logger.info(
                    "[UI] Ask Mentor clicked | dataset_id=%s | source=%s | question=%s",
                    dataset_id,
                    dataset_source,
                    question,
                )

                aggregator = DatasetAggregatorService()

                with st.spinner("Thinking..."):
                    qa_response = aggregator.ask_about_dataset(
                        dataset_id=dataset_id,
                        question=question,
                        source=dataset_source,
                    )
                    st.session_state.dataset_qa_response = qa_response

                logger.info("[UI] Ask Mentor response received | dataset_id=%s", dataset_id)

            except Exception as exc:
                logger.exception("[UI] Ask Mentor failed | dataset_id=%s | error=%s", dataset_id, exc)
                st.error(f"Failed to answer question: {exc}")

        qa_response = st.session_state.dataset_qa_response
        if qa_response:
            st.subheader("Mentor Response")
            st.write(f"**Question:** {qa_response.get('question', '')}")
            st.json(qa_response.get("answer", {}))

    except Exception as exc:
        logger.exception("[UI] render_dataset_qa failed: %s", exc)
        st.error(f"Failed to render dataset Q&A: {exc}")


def main() -> None:
    try:
        logger.info("[UI] Streamlit app initializing")

        st.set_page_config(
            page_title="AI ML Project Mentor",
            page_icon="🤖",
            layout="wide",
        )

        initialize_state()

        st.title("🤖 AI ML Project Mentor")
        st.write("Discover datasets and project ideas based on your interests and skill level.")

        with st.sidebar:
            st.header("Search Filters")
            query = st.text_input("What do you want to learn?", value="xgboost")
            level = st.selectbox("Skill Level", ["beginner", "intermediate", "advanced"])
            modality = st.selectbox("Modality", ["any", "tabular", "text", "image", "audio"])
            source = st.selectbox("Dataset Source", ["all", "openml", "huggingface", "uci", "kaggle"])
            limit = st.slider("Number of results", 5, 30, 10)

        search_clicked = st.button("Search Datasets")

        if search_clicked:
            try:
                logger.info(
                    "[UI] Search button clicked | query=%s | level=%s | modality=%s | source=%s | limit=%s",
                    query,
                    level,
                    modality,
                    source,
                    limit,
                )

                aggregator = DatasetAggregatorService()

                with st.spinner("Searching datasets..."):
                    results, warnings = aggregator.search_all_datasets(
                        query=query,
                        level=level,
                        modality=None if modality == "any" else modality,
                        source=None if source == "all" else source,
                        limit=limit,
                    )

                st.session_state.search_results = results
                st.session_state.search_warnings = warnings
                st.session_state.selected_dataset_id = None
                st.session_state.selected_dataset_source = "openml"
                st.session_state.selected_dataset_name = None
                st.session_state.dataset_details = None
                st.session_state.dataset_qa_response = None

                logger.info("[UI] Search complete | result_count=%s | warnings=%s", len(results), warnings)

            except Exception as exc:
                logger.exception("[UI] Search failed: %s", exc)
                st.error(f"Search failed: {exc}")

        for warning in st.session_state.search_warnings:
            st.warning(warning)

        results = st.session_state.search_results
        if results:
            render_search_results(results)
        elif st.session_state.search_warnings:
            st.warning("No datasets found.")

        selected_dataset_id = st.session_state.selected_dataset_id
        selected_dataset_source = st.session_state.selected_dataset_source
        details = st.session_state.dataset_details

        if selected_dataset_id and details:
            render_dataset_details(details)
            render_dataset_qa(selected_dataset_id, selected_dataset_source)

        logger.debug("[UI] Streamlit render cycle complete")

    except Exception as exc:
        logger.exception("[UI] Fatal app error: %s", exc)
        st.error("Unexpected application error occurred.")
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()