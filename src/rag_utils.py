import os
import glob
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class LocalRAGRetriever:
    def __init__(self, knowledge_dir: str):
        self.knowledge_dir = knowledge_dir
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.text_chunks = []
        self.index = None
        self._build_index()

    def _read_documents(self):
        docs = []
        txt_files = glob.glob(os.path.join(self.knowledge_dir, "*.txt"))

        for file_path in txt_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    if text:
                        docs.extend(self._chunk_text(text, chunk_size=500, overlap=100))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        return docs

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100):
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap

        return chunks

    def _build_index(self):
        self.text_chunks = self._read_documents()

        if not self.text_chunks:
            print("No knowledge base files found.")
            return

        embeddings = self.embedding_model.encode(self.text_chunks, convert_to_numpy=True)
        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        print(f"RAG index built with {len(self.text_chunks)} chunks.")

    def retrieve(self, query: str, top_k: int = 3):
        if self.index is None or not self.text_chunks:
            return []

        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)

        retrieved_chunks = []
        for idx in indices[0]:
            if 0 <= idx < len(self.text_chunks):
                retrieved_chunks.append(self.text_chunks[idx])

        return retrieved_chunks