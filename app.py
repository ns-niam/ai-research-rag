import streamlit as st

from app.ingest import extract_text
from app.chunking import chunk_text
from app.embeddings import build_index
from app.retrieval import retrieve


# ---------- PAGE ----------
st.set_page_config(
    page_title="AI Research RAG",
    page_icon="📚",
    layout="wide"
)

# ---------- HEADER ----------
st.title("📚 AI Research RAG")

st.markdown("""
### 🔍 Intelligent Research Assistant

Upload research papers and get **context-aware answers**
using Retrieval-Augmented Generation (RAG).
""")

# ---------- METRICS ----------
m1, m2, m3 = st.columns(3)

with m1:
    st.metric("📄 Documents", "∞")

with m2:
    st.metric("⚡ Vector Search", "FAISS")

with m3:
    st.metric("🧠 Mode", "RAG")

st.divider()


# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("⚙️ Features")

    st.markdown("""
### 🚀 Capabilities
- 📄 PDF Upload & Parsing  
- 🧠 Semantic Search  
- 🔍 Vector Retrieval  
- 📚 Context-Based Answers  

### 🧩 RAG Components
- Overlap Chunking  
- Embedding Search  
- FAISS Indexing  
- Grounded Responses  
""")

    st.divider()
    st.caption("💡 Tip: Ask specific questions for better answers.")


# ---------- UPLOAD ----------
st.subheader("📤 Upload Research Paper")

uploaded_file = st.file_uploader(
    "Upload your PDF file",
    type="pdf"
)

chunks = None
model = None
index = None


# ---------- PDF PROCESS ----------
if uploaded_file:

    st.success(f"Uploaded: {uploaded_file.name}")

    text = extract_text(uploaded_file)

    if text:

        st.divider()

        st.subheader("📄 Extracted Preview")
        st.write(text[:2500])

        chunks = chunk_text(text)

        # STATS
        s1, s2, s3 = st.columns(3)

        with s1:
            st.metric("Characters", len(text))

        with s2:
            st.metric("Pages", "Loaded")

        with s3:
            st.metric("Chunks", len(chunks))

        # VECTOR INDEX
        with st.spinner("⚡ Building vector index..."):
            model, index = build_index(chunks)

        st.success("Vector Index Ready ✅")


# ---------- ASK AI ----------
st.divider()

st.markdown("""
### 💬 Ask Questions

💡 Example:
- Summarize this document
- What are the main topics?
- Explain key concepts
""")

with st.form("qa_form"):
    query = st.text_input("Type your question...")
    submitted = st.form_submit_button("Ask AI")


if submitted:

    if not uploaded_file:
        st.warning("Upload a PDF first.")

    elif not query.strip():
        st.warning("Please enter a question.")

    elif index is not None:

        with st.spinner("🧠 Thinking..."):

            top_chunks = retrieve(
                query,
                chunks,
                model,
                index
            )

        context = " ".join(top_chunks)
        query_lower = query.lower()

        # ---------- ANSWER LOGIC ----------
        if "summary" in query_lower or "summarize" in query_lower:

            answer = (
                "### 📌 Summary\n\n"
                + context[:700]
                + "..."
            )

        elif "what is" in query_lower or "explain" in query_lower:

            answer = (
                "### 📖 Explanation\n\n"
                + top_chunks[0]
            )

        else:

            answer = (
                "### 📄 Answer\n\n"
                + top_chunks[0]
            )

        # ---------- OUTPUT ----------
        st.markdown(answer)

        st.subheader("📚 Retrieved Sources")

        for chunk in top_chunks:
            st.write(chunk)
            st.divider()