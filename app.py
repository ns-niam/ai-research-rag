import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# ---------- CACHED MODEL ----------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ---------- OVERLAP CHUNKING ----------
def chunk_text(text, size=1000, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap

    return chunks


# ---------- RETRIEVAL ----------
def retrieve_chunks(query, chunks, model, index, k=2):
    q_emb = model.encode([query])

    distances, indices = index.search(
        np.array(q_emb).astype("float32"), k
    )

    return [chunks[i] for i in indices[0]]


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

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    # CLEAN TEXT
    text = text.replace("\n", " ")
    text = " ".join(text.split())

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
            st.metric("Pages", len(reader.pages))

        with s3:
            st.metric("Chunks", len(chunks))

        # VECTOR INDEX
        with st.spinner("⚡ Building vector index..."):
            model = load_embedding_model()

            embeddings = model.encode(chunks)

            dim = embeddings.shape[1]

            index = faiss.IndexFlatL2(dim)

            index.add(
                np.array(embeddings).astype("float32")
            )

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
    query = st.text_input(
        "Type your question..."
    )

    submitted = st.form_submit_button("Ask AI")


if submitted:

    if not uploaded_file:
        st.warning("Upload a PDF first.")

    elif not query.strip():
        st.warning("Please enter a question.")

    elif index is not None:

        with st.spinner("🧠 Thinking..."):

            top_chunks = retrieve_chunks(
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