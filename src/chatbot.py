"""
RAG-based Chatbot using Gemini LLM and Chroma Vector Database.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from document_loader import DocumentLoader
from vector_store import VectorStore


class RAGChatbot:
    """RAG-based chatbot using Gemini and Chroma."""
    
    def __init__(self, api_key=None, persist_dir="./chroma_db", model_name="gemini-2.0-flash"):
        """
        Initialize the RAG chatbot.
        
        Args:
            api_key: Google API key (uses env var if not provided)
            persist_dir: Directory to persist Chroma database
            model_name: Gemini model to use
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.persist_dir = persist_dir
        self.model_name = model_name
        
        # Initialize components
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.api_key,
            temperature=0.0
        )
        
        self.vector_store = VectorStore(persist_dir=persist_dir, api_key=self.api_key)
        self.retriever = None
        
        print(f"✓ Initialized RAG Chatbot with {model_name}")
    
    def initialize_from_documents(self, documents_dir):
        """
        Initialize the chatbot with documents from a directory.
        
        Args:
            documents_dir: Directory containing text files
        """
        # Load documents
        loader = DocumentLoader()
        documents = loader.process_documents(documents_dir)
        
        if not documents:
            print("⚠️  No documents to process")
            return False
        
        # Add to vector store
        self.vector_store.add_documents(documents)
        
        # Create retriever
        self.retriever = self.vector_store.create_retriever(k=3)
        
        print("✓ Chatbot ready for queries!")
        return True
    
    def ask(self, query):
        """
        Ask a question to the chatbot.
        
        Args:
            query: User question
            
        Returns:
            Dictionary with answer and context
        """
        if self.retriever is None:
            raise ValueError("Chatbot not initialized. Call initialize_from_documents first.")
        
        print(f"\n🔍 Searching for relevant documents...")
        
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retriever.invoke(query)
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                print("   ℹ️  Using fallback search (quota exceeded)")
                # If retrieval fails, use first few documents as fallback
                retrieved_docs = self.vector_store.documents[:3] if self.vector_store.documents else []
            else:
                raise
        
        # Format context from retrieved documents
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No documents available."
        
        # Create prompt
        prompt = ChatPromptTemplate.from_template("""You are a helpful assistant. Use the provided context to answer questions accurately.
        If the answer is not in the context, say "I don't have information about this topic."
        
        Context:
        {context}
        
        Question: {question}
        
        Answer the question based on the context above:""")
        
        # Create chain
        chain = prompt | self.llm | StrOutputParser()
        
        # Get response
        answer = chain.invoke({
            "context": context_text,
            "question": query
        })
        
        return {
            "answer": answer,
            "context": retrieved_docs
        }
    
    def get_sources(self, response):
        """
        Extract and display source documents from response.
        
        Args:
            response: Response from ask()
        """
        if "context" in response:
            print("\n📚 Sources:")
            for i, doc in enumerate(response["context"], 1):
                print(f"\n--- Source {i} ---")
                print(doc.page_content[:200] + "...")
    
    def interactive_chat(self):
        """Start an interactive chat session."""
        print("\n" + "="*60)
        print("🤖 RAG Chatbot - Interactive Mode")
        print("="*60)
        print("Type 'exit' to quit, 'sources' to see document sources")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == "exit":
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == "sources":
                    print("ℹ️  Use this to see sources after asking a question.")
                    continue
                
                print("\n⏳ Thinking...", end="", flush=True)
                response = self.ask(user_input)
                
                print("\r          \r", end="")  # Clear the "Thinking..." message
                print(f"\nBot: {response['answer']}")
                
                # Ask if user wants to see sources
                show_sources = input("\nShow sources? (y/n): ").strip().lower()
                if show_sources == 'y':
                    self.get_sources(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


def main():
    """Main function to run the RAG chatbot."""
    load_dotenv()
    
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found in environment variables")
        print("Please set GOOGLE_API_KEY in your .env file or as an environment variable")
        return
    
    # Create chatbot
    chatbot = RAGChatbot(api_key=api_key)
    
    # Initialize with documents
    documents_dir = "../data"
    if not os.path.exists(documents_dir):
        print(f"⚠️  Documents directory not found: {documents_dir}")
        print("Please create the 'data' directory and add text files")
        return
    
    # Initialize and start chat
    if chatbot.initialize_from_documents(documents_dir):
        chatbot.interactive_chat()


if __name__ == "__main__":
    main()
