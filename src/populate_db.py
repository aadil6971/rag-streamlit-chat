from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings.ollama import  OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from get_embedding_function import get_embedding_function
DATA_PATH = "data"
CHROMA_PATH = "chroma_db"

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)



def calculate_chunk_ids(chunks: list[Document]):
    last_page_id = None
    current_chunk_id = 0
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        if current_page_id == last_page_id:
            current_chunk_id += 1
        else:
            current_chunk_id = 0

        chunk_id = f"{source}:{page}:{current_chunk_id}"
        last_page_id = current_page_id
        
        chunk.metadata["id"] = chunk_id
    return chunks

def add_to_chroma(chunks:list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )
    chunks_with_ids = calculate_chunk_ids(chunks)
    
    
    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def main():
    print("Loading documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")

    print("Splitting documents into chunks...")
    chunks = split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    print("Adding chunks to Chroma DB...")
    add_to_chroma(chunks)
    print("Done!")

main()