import streamlit as st
import pandas as pd

from app.services.dataset_aggregator import search_all_datasets


st.set_page_config(
    page_title="AI ML Project Mentor",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AI ML Project Mentor")
st.write("Discover datasets and project ideas based on your interests and skill level.")

with st.sidebar:
    st.header("Search Filters")
    query = st.text_input("What do you want to learn?", value="nlp")
    level = st.selectbox("Skill Level", ["beginner", "intermediate", "advanced"])
    modality = st.selectbox("Modality", ["any", "tabular", "text", "image", "audio"])
    source = st.selectbox("Dataset Source", ["all", "openml", "huggingface", "uci", "kaggle"])
    limit = st.slider("Number of results", 5, 30, 10)

search_clicked = st.button("Search Datasets")

if search_clicked:
    with st.spinner("Searching datasets..."):
        results, warnings = search_all_datasets(
        query=query,
        level=level,
        modality=None if modality == "any" else modality,
        source=None if source == "all" else source,
        limit=limit,
    )

    for warning in warnings:
        st.warning(warning)

    if not results:
        st.warning("No datasets found.")
    else:
        st.success(f"Found {len(results)} datasets")

        for item in results:
            with st.container():
                st.subheader(item.name)
                st.write(item.description or "No description available.")
                st.markdown(f"**Source:** {item.source}")
                st.markdown(f"**Modality:** {item.modality or 'Unknown'}")
                st.markdown(f"**Difficulty:** {item.difficulty or 'Unknown'}")
                st.markdown(f"**Size:** {item.size or 'Unknown'}")
                st.markdown(f"**Score:** {item.score:.2f}")

                if item.tags:
                    st.write("**Tags:**", ", ".join(item.tags[:8]))

                if item.url:
                    st.markdown(f"[Open Dataset]({item.url})")

                st.divider()

        st.subheader("Tabular View")
        df = pd.DataFrame([item.model_dump() for item in results])
        st.dataframe(df, use_container_width=True)