
import streamlit as st
import pandas as pd
import html
from transformers import pipeline

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Biomedical NER System",
    page_icon="🧬",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================
st.markdown("""
<style>
.main-title {
    font-size: 34px;
    font-weight: bold;
    color: #2563eb;
}
.entity {
    background-color: #dbeafe;
    color: #1e40af;
    padding: 3px 6px;
    border-radius: 5px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================
st.markdown(
    '<div class="main-title">🧬 Biomedical Named Entity Recognition</div>',
    unsafe_allow_html=True
)

st.write(
    "An NLP application that extracts specialized biomedical "
    "entities using a pretrained Transformer model."
)

st.divider()

# ==================================================
# MODEL
# ==================================================
MODEL_NAME = "d4data/biomedical-ner-all"


@st.cache_resource
def load_ner_model():
    ner = pipeline(
        task="ner",
        model=MODEL_NAME,
        aggregation_strategy="simple"
    )
    return ner


# ==================================================
# SAMPLE TEXT
# ==================================================
sample_text = """
The patient was diagnosed with diabetes and hypertension.
Metformin was prescribed to control blood glucose levels.
The patient also reported headache and mild fever.
"""


# ==================================================
# INPUT SECTION
# ==================================================
st.subheader("📝 Biomedical Text Input")

text_input = st.text_area(
    "Enter medical or biomedical text:",
    value=sample_text,
    height=180,
    placeholder="Enter your biomedical text here..."
)

uploaded_file = st.file_uploader(
    "Or upload a text document (.txt)",
    type=["txt"]
)

if uploaded_file is not None:
    try:
        uploaded_text = uploaded_file.read().decode(
            "utf-8",
            errors="ignore"
        )
        text_input = uploaded_text
        st.success("Text file uploaded successfully!")
    except Exception:
        st.error("Unable to read the uploaded file.")

# ==================================================
# ENTITY EXTRACTION
# ==================================================
if st.button("🔍 Extract Entities", type="primary"):

    if not text_input.strip():
        st.warning("Please enter or upload some text.")

    else:
        try:
            with st.spinner(
                "Loading biomedical Transformer model..."
            ):
                ner_model = load_ner_model()

            with st.spinner("Extracting biomedical entities..."):
                entities = ner_model(text_input)

            st.success("Entity extraction completed!")

            # ------------------------------------------
            # HIGHLIGHT ENTITIES
            # ------------------------------------------
            st.subheader("🎯 Highlighted Biomedical Text")

            valid_entities = []

            for entity in entities:
                start = entity.get("start")
                end = entity.get("end")

                if start is not None and end is not None:
                    valid_entities.append(entity)

            # Sort entities by character position
            valid_entities.sort(
                key=lambda item: item["start"]
            )

            highlighted_parts = []
            current_position = 0

            for entity in valid_entities:

                start = entity["start"]
                end = entity["end"]

                # Skip overlapping or invalid entities
                if start < current_position:
                    continue

                # Add normal text before entity
                normal_text = text_input[
                    current_position:start
                ]

                highlighted_parts.append(
                    html.escape(normal_text)
                )

                # Entity text
                entity_text = text_input[start:end]

                entity_label = entity.get(
                    "entity_group",
                    entity.get("entity", "ENTITY")
                )

                confidence = entity.get("score", 0)

                highlighted_parts.append(
                    f'<span class="entity" '
                    f'title="{html.escape(str(entity_label))}">'
                    f'{html.escape(entity_text)}'
                    f'</span>'
                )

                current_position = end

            # Add remaining text
            highlighted_parts.append(
                html.escape(text_input[current_position:])
            )

            highlighted_html = "".join(
                highlighted_parts
            )

            st.markdown(
                f'<div style="line-height:2.2;'
                f'font-size:17px;">'
                f'{highlighted_html}'
                f'</div>',
                unsafe_allow_html=True
            )

            # ------------------------------------------
            # ENTITY TABLE
            # ------------------------------------------
            st.subheader("📊 Extracted Entities")

            if entities:

                entity_data = []

                for entity in entities:

                    entity_data.append({
                        "Entity": entity.get("word", ""),
                        "Category": entity.get(
                            "entity_group",
                            entity.get("entity", "Unknown")
                        ),
                        "Confidence (%)": round(
                            float(entity.get("score", 0)) * 100,
                            2
                        )
                    })

                df = pd.DataFrame(entity_data)

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )

                # --------------------------------------
                # SUMMARY METRICS
                # --------------------------------------
                st.subheader("📈 Entity Statistics")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Total Entities",
                        len(df)
                    )

                with col2:
                    st.metric(
                        "Entity Categories",
                        df["Category"].nunique()
                    )

                # --------------------------------------
                # CATEGORY CHART
                # --------------------------------------
                st.subheader("Entity Category Distribution")

                category_counts = (
                    df["Category"]
                    .value_counts()
                )

                st.bar_chart(category_counts)

                # --------------------------------------
                # CSV DOWNLOAD
                # --------------------------------------
                csv_data = df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Entities CSV",
                    data=csv_data,
                    file_name="biomedical_entities.csv",
                    mime="text/csv"
                )

            else:
                st.warning(
                    "No biomedical entities were detected."
                )

        except Exception as error:
            st.error(
                "An error occurred during entity extraction."
            )

            st.code(str(error))

            st.info(
                "Check your internet connection. "
                "The Transformer model downloads during "
                "the first run."
            )

# ==================================================
# FOOTER
# ==================================================
st.divider()

st.caption(
    "NLP Capstone Project | "
    "Domain-Specific NER with Transformers | "
    "Biomedical Domain"
)
