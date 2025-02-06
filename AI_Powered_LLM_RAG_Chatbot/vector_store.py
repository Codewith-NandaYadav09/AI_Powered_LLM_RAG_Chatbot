from langchain_community.embeddings import OpenAIEmbeddings
import faiss
import numpy as np
import pickle

class VectorStore:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.index = None
        self.documents = []
    
    def create_index(self, documents):
        self.documents = documents
        embeddings = self.embedding_model.embed_documents(
            [doc.page_content for doc in documents]
        )
        
        dimension = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings, dtype=np.float32))
    
    def search(self, query, k=3):
        query_embedding = self.embedding_model.embed_query(query)
        distances, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32), k
        )
        return [self.documents[i] for i in indices[0]]