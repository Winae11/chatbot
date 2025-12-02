"""
Module for managing vector database and embeddings using Chroma.
"""

import os
import shutil
import pickle
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document


class SimpleVectorStore:
    """Simple in-memory vector store for document embeddings."""
    
    def __init__(self, persist_dir="./chroma_db", api_key=None):
        """
        Initialize the vector store.
        
        Args:
            persist_dir: Directory to persist database
            api_key: Google API key (uses GOOGLE_API_KEY env var if not provided)
        """
        self.persist_dir = persist_dir
        self.api_key = api_key
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        self.documents = []
        self.embeddings_list = []
        self.embeddings_file = os.path.join(persist_dir, "embeddings.pkl")
        self.documents_file = os.path.join(persist_dir, "documents.pkl")
    
    def _simple_hash_embedding(self, text):
        """Generate a simple embedding using hash-based approach (fallback when API unavailable)."""
        # Create a simple vector based on text characteristics
        embedding = [0.0] * 768  # Standard embedding dimension
        
        # Use character codes to create a deterministic embedding
        for i, char in enumerate(text[:100]):  # Use first 100 chars
            idx = (i + ord(char)) % 768
            embedding[idx] += ord(char) / 256.0
        
        # Normalize
        magnitude = sum(x**2 for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding
    
    def add_documents(self, documents):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document chunks to add
            
        Returns:
            Self for chaining
        """
        print(f"\n🔄 Adding {len(documents)} documents to vector store...")
        
        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_dir, exist_ok=True)
        
        self.documents = documents
        
        # Generate embeddings for all documents
        print("⏳ Generating embeddings (this may take a moment)...")
        self.embeddings_list = []
        use_fallback = False
        
        for i, doc in enumerate(documents):
            try:
                if not use_fallback:
                    embedding = self.embeddings.embed_query(doc.page_content)
                else:
                    embedding = self._simple_hash_embedding(doc.page_content)
                
                self.embeddings_list.append(embedding)
                if (i + 1) % 5 == 0:
                    print(f"   ✓ Embedded {i + 1}/{len(documents)} documents")
            except Exception as e:
                error_msg = str(e)
                if "quota" in error_msg.lower() or "429" in error_msg:
                    print(f"   ⚠️  API quota exceeded, using fallback embedding method")
                    use_fallback = True
                    embedding = self._simple_hash_embedding(doc.page_content)
                    self.embeddings_list.append(embedding)
                else:
                    print(f"   ✗ Error embedding document {i}: {e}")
                    self.embeddings_list.append(None)
        
        # Save to disk
        self._save_to_disk()
        print(f"✓ Documents added and embeddings saved to: {self.persist_dir}")
        return self
    
    def _save_to_disk(self):
        """Save embeddings and documents to disk."""
        try:
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings_list, f)
            with open(self.documents_file, 'wb') as f:
                pickle.dump(self.documents, f)
        except Exception as e:
            print(f"⚠️  Warning: Could not save to disk: {e}")
    
    def _load_from_disk(self):
        """Load embeddings and documents from disk."""
        try:
            if os.path.exists(self.embeddings_file) and os.path.exists(self.documents_file):
                with open(self.embeddings_file, 'rb') as f:
                    self.embeddings_list = pickle.load(f)
                with open(self.documents_file, 'rb') as f:
                    self.documents = pickle.load(f)
                print(f"✓ Loaded {len(self.documents)} documents from cache")
                return True
        except Exception as e:
            print(f"⚠️  Warning: Could not load from disk: {e}")
        return False
    
    def similarity_search(self, query, k=3):
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        if not self.documents:
            # Try loading from disk
            if not self._load_from_disk():
                raise ValueError("No documents in vector store")
        
        # Generate embedding for query
        try:
            query_embedding = self.embeddings.embed_query(query)
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                print("   ℹ️  Using fallback embedding (quota exceeded)")
                query_embedding = self._simple_hash_embedding(query)
            else:
                raise
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings_list):
            if doc_embedding is None:
                continue
            
            # Cosine similarity
            dot_product = sum(a * b for a, b in zip(query_embedding, doc_embedding))
            magnitude_query = sum(a**2 for a in query_embedding) ** 0.5
            magnitude_doc = sum(b**2 for b in doc_embedding) ** 0.5
            
            if magnitude_query > 0 and magnitude_doc > 0:
                similarity = dot_product / (magnitude_query * magnitude_doc)
            else:
                similarity = 0
            
            similarities.append((i, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k documents
        results = []
        for idx, _ in similarities[:k]:
            results.append(self.documents[idx])
        
        return results if results else self.documents[:k]
    
    def as_retriever(self, search_type="similarity", search_kwargs=None):
        """
        Create a retriever for the vector store.
        
        Args:
            search_type: Type of search (only "similarity" supported)
            search_kwargs: Search kwargs (k parameter for number of results)
            
        Returns:
            Retriever object
        """
        if search_kwargs is None:
            search_kwargs = {"k": 3}
        
        k = search_kwargs.get("k", 3)
        
        class Retriever:
            def __init__(self, store, k):
                self.store = store
                self.k = k
            
            def invoke(self, query):
                return self.store.similarity_search(query, k=self.k)
        
        return Retriever(self, k)
    
    def create_retriever(self, k=3):
        """
        Create a retriever for the vector store (alias for compatibility).
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            Retriever object
        """
        return self.as_retriever(search_kwargs={"k": k})
    
    def clear(self):
        """Clear the vector store."""
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
            print(f"✓ Cleared vector store at: {self.persist_dir}")
        self.documents = []
        self.embeddings_list = []


# Alias for compatibility
VectorStore = SimpleVectorStore
