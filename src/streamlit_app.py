
import streamlit as st
import subprocess
import os
import io
from populate_db import main as populate_db
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function
from pdf2image import convert_from_path

from config import CHROMA_PATH, BASE_PROMPT, PERSONAS, COLORS

# Optional PDF generation
def create_pdf_bytes(md_text: str) -> bytes:
    try:
        from fpdf import FPDF
    except ImportError:
        st.error("Install fpdf (`pip install fpdf`) to enable PDF export.")
        return None
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in md_text.splitlines():
        pdf.multi_cell(0, 8, line)
    return pdf.output(dest="S").encode("latin-1")

# Cache PDF page images, but handle missing Poppler gracefully
@st.cache_data(show_spinner=False)
def get_pdf_page_image(pdf_path: str, page_number: int):
    """Convert a specific PDF page to an image, requires Poppler."""
    try:
        images = convert_from_path(
            pdf_path,
            first_page=page_number,
            last_page=page_number
        )
        return images[0] if images else None
    except Exception:
        st.warning(
            "PDF preview disabled (Poppler not installed or not in PATH)."
        )
        return None



# Helper to list Ollama models via CLI
@st.cache_data(show_spinner=False)
def get_available_models():
    try:
        result = subprocess.run(["ollama", "ls"], capture_output=True, text=True, check=True)
        lines = [l for l in result.stdout.splitlines() if l.strip()]
        if lines and lines[0].upper().startswith("NAME"):
            lines = lines[1:]
        return [line.split()[0] for line in lines]
    except Exception as e:
        st.warning(f"Could not list Ollama models: {e}")
        return []

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# App layout
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🔍 RAG Chatbot with Streamlit")

# Sidebar: upload, settings, export
with st.sidebar:
    st.header("Upload Documents")
    uploaded = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        os.makedirs("data", exist_ok=True)
        for f in uploaded:
            path = os.path.join("data", f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
        populate_db()
        st.success("Documents uploaded and indexed.")

    st.markdown("---")
    st.header("Settings & Export")
    persona = st.selectbox("Response Persona", list(PERSONAS.keys()))
    k = st.slider("Context chunks (k)", 1, 10, 5)
    model_opts = get_available_models() or ["deepseek-r1:8b"]
    model_name = st.selectbox("Ollama Model", model_opts)
    if st.button("Clear History"):
        st.session_state.history = []
        st.experimental_rerun()

    # Export chat buttons
    if st.session_state.history:
        md_lines = []
        for msg in st.session_state.history:
            prefix = "**User:**" if msg["role"] == "user" else "**Assistant:**"
            md_lines.append(f"{prefix} {msg['content']}")
        md_text = "\n\n".join(md_lines)
        st.download_button(
            "Download Chat as Markdown",
            data=md_text,
            file_name="chat.md",
            mime="text/markdown"
        )
        pdf_bytes = create_pdf_bytes(md_text)
        if pdf_bytes:
            st.download_button(
                "Download Chat as PDF",
                data=pdf_bytes,
                file_name="chat.pdf",
                mime="application/pdf"
            )

# Render chat history
for msg in st.session_state.history:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
query = st.chat_input("Ask a question...")
if query:
    # Echo user
    st.chat_message("user").write(query)
    st.session_state.history.append({"role": "user", "content": query})

    # Retrieval
    embedding_fn = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_fn)
    try:
        results = db.similarity_search_with_score(query, k=k)
    except Exception:
        st.warning("No documents in the vector store—upload & index PDFs first.")
        st.stop()
    if not results:
        st.warning("No matching chunks found.")
        st.stop()

    # Build context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_text = PERSONAS[persona] + "\n" + BASE_PROMPT
    prompt_template = ChatPromptTemplate.from_template(prompt_text)
    prompt = prompt_template.format(context=context_text, question=query)

    # LLM invocation
    model = Ollama(model=model_name)
    reply = model.invoke(prompt)
    st.chat_message("assistant").write(reply)
    st.session_state.history.append({"role": "assistant", "content": reply})

    # Sources with PDF preview and highlighted chunks
    st.markdown("**Sources & Context Snippets:**")
    for idx, (doc, score) in enumerate(results):
        sid = doc.metadata.get("id", "unknown")
        source_file, page_str, _ = sid.split(":")
        page_num = int(page_str)
        with st.expander(f"Source: {sid} (score: {score:.3f})"):
            # PDF page image
            pdf_path = os.path.join("data", source_file)
            img = get_pdf_page_image(pdf_path, page_num)
            if img:
                st.image(img, caption=f"Page {page_num}")
            # Highlight chunk
            color = COLORS[idx % len(COLORS)]
            st.markdown(
                f"<div style='background-color:{color}; padding:8px; border-radius:4px;'>{doc.page_content}</div>",
                unsafe_allow_html=True
            )

# Footer
st.markdown("---")
st.markdown("Built with [Streamlit](https://streamlit.io/) and LangChain.")
