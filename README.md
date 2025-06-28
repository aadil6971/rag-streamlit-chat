# 🔍 RAG Chatbot with Streamlit

A simple Retrieval-Augmented Generation (RAG) chat interface built with Streamlit and LangChain, using Ollama LLMs and Chroma vector store.

## Features

* **Interactive Chat UI**: Streamlit-based chat interface with message history.
* **Retrieval**: Uses Chroma to perform similarity search over pre-indexed documents.
* **Augmentation**: Constructs prompts with relevant context for more accurate answers.
* **LLM Integration**: Supports Ollama models via the `ollama` CLI.
* **Configurable**: Adjust number of context chunks (`k`) and select available Ollama model.
* **Clear History**: Option to reset the conversation.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10 or higher
* A virtual environment (recommended)
* [Streamlit](https://streamlit.io/)
* [Chroma DB](https://www.trychroma.com/)
* [Ollama](https://ollama.com/) CLI installed and running

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/aadil6971/rag-streamlit-chat.git
   cd rag-streamlit-chat
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate    # macOS/Linux
   .venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Populate Chroma DB**

   * Place your PDFs into the `data/` directory at the project root.
   * Run the existing script to index your documents into Chroma:

     ```bash
     cd src
     python populate_db.py
     ```

---

## 🎨 Configuration

* **Number of Context Chunks (`k`)**: Select how many top similar chunks to include in the prompt (1–10).
* **Model Selection**: Dropdown of models available from `ollama ls`.

These controls live in the sidebar.

---

## 💻 Running the App

```bash
streamlit run streamlit_app.py
```

This will open a browser window at `http://localhost:8501` with the chat UI.

---

## 🛠️ How It Works

1. **User Query**: You send a message via the chat input.
2. **Retrieval**: The app queries Chroma for the top `k` similar document chunks.
3. **Prompt Construction**: It builds a prompt template with the retrieved context.
4. **LLM Call**: Sends the prompt to the selected Ollama model.
5. **Display**: Shows a loading spinner while thinking, then renders the assistant’s reply and updates history.

---

## 🔧 Customization

* **Prompt Template**: Modify `PROMPT_TEMPLATE` in `streamlit_app.py`.
* **Embedding Function**: Adjust `get_embedding_function.py` to switch embedding providers.
* **Vector Store**: Swap Chroma for another `langchain`-supported store.

---
