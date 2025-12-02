"""
Module for loading and processing documents for RAG system.
"""

import os
from pathlib import Path
from langchain_core.documents import Document


class DocumentLoader:
    """Handles loading and processing of documents."""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        """
        Initialize the document loader.
        
        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks to maintain context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def _split_text(self, text, chunk_size, chunk_overlap):
        """Simple text splitter using character-based chunking."""
        chunks = []
        
        if not text or len(text) == 0:
            return chunks
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk_size, save current chunk
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Keep last part for overlap
                current_chunk = paragraph[-chunk_overlap:] if len(paragraph) > chunk_overlap else paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n"
                current_chunk += paragraph
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text[:chunk_size]]
    
    def load_documents_from_directory(self, directory_path):
        """
        Load all text files from a directory.
        
        Args:
            directory_path: Path to directory containing text files
            
        Returns:
            List of loaded documents
        """
        documents = []
        data_dir = Path(directory_path)
        
        # Load all .txt files from the directory
        for file_path in data_dir.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                doc = Document(page_content=content, metadata={"source": str(file_path)})
                documents.append(doc)
                print(f"✓ Loaded: {file_path.name}")
            except Exception as e:
                print(f"✗ Error loading {file_path.name}: {e}")
        
        return documents
    
    def split_documents(self, documents):
        """
        Split documents into chunks for vector embedding.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        chunks = []
        
        for doc in documents:
            text_chunks = self._split_text(doc.page_content, self.chunk_size, self.chunk_overlap)
            for chunk_text in text_chunks:
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata=doc.metadata
                )
                chunks.append(chunk_doc)
        
        print(f"\n✓ Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks
    
    def process_documents(self, directory_path):
        """
        Complete pipeline: load and split documents.
        
        Args:
            directory_path: Path to directory with documents
            
        Returns:
            List of processed document chunks
        """
        print(f"\n📂 Loading documents from: {directory_path}")
        documents = self.load_documents_from_directory(directory_path)
        
        if not documents:
            print("⚠️  No documents found!")
            return []
        
        print(f"\n✓ Loaded {len(documents)} documents")
        chunks = self.split_documents(documents)
        return chunks
