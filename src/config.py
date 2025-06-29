CHROMA_PATH = "chroma_db"
DATA_PATH = "data"
BASE_PROMPT = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
PERSONAS = {
    "Concise": "Answer briefly and to the point.",
    "Detailed": "Provide a comprehensive, detailed answer.",
    "Expert": "Answer as an expert with advanced insights.",
    "Exact" : "Answer with exact number or fact only",
}
# Colors for highlighting chunks
COLORS = ["#fff2cc", "#fce5cd", "#d9ead3", "#cfe2f3", "#d9d2e9"]