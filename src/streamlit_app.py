import streamlit as st
import subprocess
import os
from populate_db import main as populate_db
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function

# Constants
CHROMA_PATH = "chroma_db"
BASE_PROMPT = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
PERSONAS = {
    "Concise": "Answer briefly and to the point.",
    "Detailed": "Provide a comprehensive, detailed answer.",
    "Expert": "Answer as an expert with advanced insights."
}

@st.cache_data(show_spinner=False)
def get_available_models():
    """List installed Ollama models via CLI."""
    try:
        result = subprocess.run(["ollama", "ls"], capture_output=True, text=True, check=True)
        lines = [l for l in result.stdout.splitlines() if l.strip()]
        if lines and lines[0].upper().startswith("NAME"):
            lines = lines[1:]
        return [line.split()[0] for line in lines]
    except Exception as e:
        st.warning(f"Could not list Ollama models: {e}")
        return []

# Initialize session
if "history" not in st.session_state:
    st.session_state.history = []

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🔍 RAG Chatbot with Streamlit")

# Sidebar: Upload & Settings
with st.sidebar:
    st.header("Upload Documents")
    uploaded = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        os.makedirs("data", exist_ok=True)
        for f in uploaded:
            with open(os.path.join("data", f.name), "wb") as out:
                out.write(f.getbuffer())
        populate_db()
        st.success("Documents uploaded and indexed.")

    st.markdown("---")
    st.header("Settings")
    persona = st.selectbox("Response Persona", list(PERSONAS.keys()))
    k = st.slider("Context chunks (k)", 1, 10, 5)
    model_opts = get_available_models() or ["deepseek-r1:8b"]
    model_name = st.selectbox("Ollama Model", model_opts)
    if st.button("Clear History"):
        st.session_state.history = []
        st.experimental_rerun()

# Display chat history
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

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    # Build prompt
    prompt_text = PERSONAS[persona] + "\n" + BASE_PROMPT
    prompt_template = ChatPromptTemplate.from_template(prompt_text)
    prompt = prompt_template.format(context=context_text, question=query)

    # LLM call (non-streaming)
    model = Ollama(model=model_name)
    reply = model.invoke(prompt)
    st.chat_message("assistant").write(reply)
    st.session_state.history.append({"role": "assistant", "content": reply})

    # Show sources
    st.markdown("**Sources & Context Snippets:**")
    for doc, score in results:
        sid = doc.metadata.get("id", "unknown")
        with st.expander(f"Source: {sid} (score: {score:.3f})"):
            st.write(doc.page_content)
