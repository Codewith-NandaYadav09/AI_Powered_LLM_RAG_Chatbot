# # from config import *
# from document_loader import DocumentProcessor
# from vector_store import VectorStore
# from chatbot import SustainabilityChatbot
# from langchain_community.embeddings import OpenAIEmbeddings

# def main():
#     # Initialize document processor
#     processor = DocumentProcessor(DOCUMENTS_DIR)
#     documents = processor.load_documents()
    
#     # Initialize vector store
#     embedding_model = OpenAIEmbeddings()
#     vector_store = VectorStore(embedding_model)
#     vector_store.create_index(documents)
    
#     # Initialize chatbot
#     chatbot = SustainabilityChatbot(vector_store)
    
#     # Interactive loop
#     print("Sustainability Chatbot initialized. Type 'quit' to exit.")
#     while True:
#         question = input("\nYour question: ")
#         if question.lower() == 'quit':
#             break
        
#         response = chatbot.get_response(question)
#         print(f"\nChatbot: {response}")

# if __name__ == "__main__":
#     main()

# main.py
from config import *
from document_loader import DocumentProcessor  # Using the same document loader as before
from vector_store import VectorStore  # Using the same vector store implementation
from embedding_model import HuggingFaceEmbeddings
from chatbot import SustainabilityChatbot

def main():
    # Initialize document processor
    processor = DocumentProcessor(DOCUMENTS_DIR)
    documents = processor.load_documents()
    
    # Initialize vector store with Hugging Face embeddings
    embedding_model = HuggingFaceEmbeddings()
    vector_store = VectorStore(embedding_model)
    vector_store.create_index(documents)
    
    # Initialize chatbot
    chatbot = SustainabilityChatbot(vector_store)
    
    # Interactive loop
    print("Sustainability Chatbot initialized. Type 'quit' to exit.")
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'quit':
            break
        
        response = chatbot.get_response(question)
        print(f"\nChatbot: {response}")

if __name__ == "__main__":
    main()