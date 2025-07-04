from sentence_transformers import SentenceTransformer

class HuggingFaceEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=True).cpu().numpy()
    
    def embed_query(self, text):
        return self.model.encode([text], convert_to_tensor=True).cpu().numpy()[0]
