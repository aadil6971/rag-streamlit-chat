import streamlit as st
import subprocess
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Helper to list Ollama models via CLI\ n@st.cache_data(show_spinner=False)
def get_available_models():
    """Run `ollama ls` and return a list of model names."""
    try:
        result = subprocess.run(["ollama", "ls"], capture_output=True, text=True, check=True)
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        # Drop header line if it starts with 'NAME'
        if lines and lines[0].upper().startswith("NAME"):
            lines = lines[1:]
        # Each line begins with the model name
        models = [line.split()[0] for line in lines]
        return models
    except Exception as e:
        st.warning(f"Could not list Ollama models: {e}")
        return []

# Initialize session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🔍 RAG Chatbot with Streamlit")

# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    k = st.slider("Number of context chunks (k)", min_value=1, max_value=10, value=5)
    model_options = get_available_models() or ["deepseek-r1:8b"]
    model_name = st.selectbox("Ollama Model", model_options)
    if st.button("Clear chat history"):
        st.session_state.history = []
        st.experimental_rerun()

# Display past chat history
for msg in st.session_state.history:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
query = st.chat_input("Ask a question...")

if query:
    # Show user message immediately
    st.chat_message("user").write(query)
    st.session_state.history.append({"role": "user", "content": query})
    
    # Placeholder for assistant message
    assistant_msg = st.chat_message("assistant")
    with st.spinner("Thinking..."):
        # Prepare database and embeddings
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        results = db.similarity_search_with_score(query, k=k)
        
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
        
        # Format prompt
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query)
        
        # Invoke LLM
        model = Ollama(model=model_name)
        reply = model.invoke(prompt)

    # Write assistant reply
    assistant_msg.write(reply)
    st.session_state.history.append({"role": "assistant", "content": reply})

# Footer
st.markdown("---")
st.markdown("Built with [Streamlit](https://streamlit.io/) and LangChain.")
