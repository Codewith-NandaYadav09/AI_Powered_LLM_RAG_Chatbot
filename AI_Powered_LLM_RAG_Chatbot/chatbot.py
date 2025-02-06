# from langchain_community.chat_models import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
# import google.generativeai as genai

# class SustainabilityChatbot:
#     def __init__(self, vector_store, model_name="gpt-3.5-turbo"):
#         self.vector_store = vector_store
#         self.llm = ChatOpenAI(model_name=model_name)
        
#         # Initialize Gemini Pro as backup
#         # genai.configure(api_key=GOOGLE_API_KEY)
#         # self.gemini = genai.GenerativeModel('gemini-pro')
        
#         self.prompt_template = ChatPromptTemplate.from_messages([
#             ("system", """You are a sustainability expert chatbot. Use the provided context to answer 
#             questions about carbon footprint reduction, ESG policies, and sustainability strategies. 
#             If the information isn't in the context, say so and provide general best practices."""),
#             ("user", "Context: {context}\n\nQuestion: {question}")
#         ])
    
#     def get_response(self, question):
#         # try:
#             # Get relevant documents from vector store
#         relevant_docs = self.vector_store.search(question)
#         context = "\n".join([doc.page_content for doc in relevant_docs])
        
#         # Generate response using primary model (OpenAI)
#         messages = self.prompt_template.format_messages(
#             context=context,
#             question=question
#         )
#         response = self.llm.invoke(messages)
#         return response.content
            
#         # except Exception as e:
#         #     # Fallback to Gemini if OpenAI fails
#         #     prompt = f"Context: {context}\n\nQuestion: {question}"
#         #     response = self.gemini.generate_content(prompt)
#         #     return response.text

# chatbot.py
import google.generativeai as genai
from langchain.prompts import PromptTemplate

class SustainabilityChatbot:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = HuggingFaceLLM()
        
        # Initialize Gemini Pro as backup
        genai.configure(api_key=GOOGLE_API_KEY)
        self.gemini = genai.GenerativeModel('gemini-pro')
        
        self.prompt_template = PromptTemplate(
            template="""You are a sustainability expert chatbot. Use the following context to answer 
            questions about carbon footprint reduction, ESG policies, and sustainability strategies. 
            If the information isn't in the context, say so and provide general best practices.

            Context: {context}

            Question: {question}

            Answer:""",
            input_variables=["context", "question"]
        )
    
    def get_response(self, question):
        try:
            # Get relevant documents from vector store
            relevant_docs = self.vector_store.search(question)
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            # Generate response using primary model (Hugging Face)
            prompt = self.prompt_template.format(
                context=context,
                question=question
            )
            response = self.llm.generate(prompt)
            return response
            
        except Exception as e:
            # Fallback to Gemini if Hugging Face fails
            prompt = f"Context: {context}\n\nQuestion: {question}"
            response = self.gemini.generate_content(prompt)
            return response.text