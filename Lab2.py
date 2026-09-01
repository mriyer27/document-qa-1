import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Lab 2 — Document Summarizer
# ---------------------------------------------------------------------------

st.title("Lab 2 - Document Summarizer")
st.write(
    "Upload a PDF and get an AI-generated summary. "
    "Pick the language, the summary style, and the model in the sidebar."
)

# ---------------------------------------------------------------------------
# Part B: get the API key from Streamlit secrets instead of a text input.
# Add this to your .streamlit/secrets.toml (create the file if it doesn't exist):
#
#   OPENAI_API_KEY = "sk-...your-key-here..."
# ---------------------------------------------------------------------------
openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)

# ---------------------------------------------------------------------------
# Part C: sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Summary Options")

# Dropdown 1: language
language = st.sidebar.selectbox(
    "Summary language",
    ("English", "Spanish", "French", "German", "Chinese"),
)

# Dropdown 2: summary type
summary_type = st.sidebar.selectbox(
    "Summary type",
    (
        "Summarize the document in 100 words",
        "Summarize the document in 2 connecting paragraphs",
        "Summarize the document in 5 bullet points",
    ),
)

# Checkbox: pick between the "nano" (default) and "mini" (advanced) models
use_advanced_model = st.sidebar.checkbox("Use advanced model")
model = "gpt-5.5" if use_advanced_model else "gpt-5.4-mini"
st.sidebar.caption(f"Model in use: `{model}`")

# ---------------------------------------------------------------------------
# File upload + summary generation
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload a document (.pdf)", type=("pdf",))


def extract_pdf_text(file) -> str:
    """Pull the plain text out of every page of an uploaded PDF."""
    reader = PdfReader(file)
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages_text)


if uploaded_file:
    document_text = extract_pdf_text(uploaded_file)

    if not document_text.strip():
        st.warning(
            "Couldn't find any extractable text in that PDF "
            "(it may be a scanned image without OCR text)."
        )
    else:
        instruction = f"{summary_type}."
        prompt = (
            f"{instruction} Write the summary in {language}.\n\n"
            f"Here is the document:\n\n{document_text}"
        )

        messages = [{"role": "user", "content": prompt}]

        with st.spinner("Generating summary..."):
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )
            st.write_stream(stream)
else:
    st.info("Please upload a PDF document to generate a summary.", icon="📄")