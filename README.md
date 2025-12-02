# RAG-Based Chatbot with Gemini and Chroma

A simple yet powerful Retrieval-Augmented Generation (RAG) chatbot that combines Google's Gemini LLM with Chroma vector database for intelligent document-grounded conversations.

## 📚 What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that enhances large language models (LLMs) by augmenting them with external knowledge. Here's how it works:

```
User Question → Vector Search → Relevant Documents → LLM + Context → Answer
```

### Key Components:

1. **Document Preparation**: Split documents into chunks and generate embeddings
2. **Vector Storage**: Store embeddings in Chroma vector database
3. **Retrieval**: Find relevant documents based on semantic similarity
4. **Generation**: Pass retrieved documents to Gemini LLM for response generation

### Why RAG?

- ✅ Reduces hallucinations by grounding responses in real data
- ✅ Works with current information without model retraining
- ✅ Enables knowledge over large document collections
- ✅ Sources are traceable for verification
- ✅ More cost-effective with smaller LLMs
- ✅ Privacy-preserving with local document storage

## 🏗️ Architecture

### Project Structure

```
chatbot/
├── src/
│   ├── chatbot.py          # Main RAG chatbot implementation
│   ├── document_loader.py  # Document loading and chunking
│   └── vector_store.py     # Chroma vector database management
├── data/
│   ├── python_guide.txt    # Sample documents
│   ├── ml_guide.txt
│   └── vector_db_rag.txt
├── chroma_db/              # Persisted vector database (created after first run)
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create this)
└── README.md              # This file
```

### Component Overview

#### 1. **DocumentLoader** (`document_loader.py`)
- Loads text files from a directory
- Splits documents into semantic chunks
- Maintains context with overlapping chunks

#### 2. **VectorStore** (`vector_store.py`)
- Manages Chroma database operations
- Generates embeddings using Gemini's embedding model
- Provides retrieval interface for similarity search
- Persists data to disk for reuse

#### 3. **RAGChatbot** (`chatbot.py`)
- Combines all components
- Creates retrieval chain
- Handles user queries and generates responses
- Interactive chat interface

## 🔧 Installation

### 1. Prerequisites
- Python 3.8 or higher
- Google API key with Gemini access

### 2. Get Your Google API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the API key

### 3. Setup Project

```bash
# Clone or download the project
cd chatbot

# Create a virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure Environment

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your actual Google API key.

## 🚀 Usage

### Run the Chatbot

```bash
# Make sure you're in the src directory or adjust the path
cd src
python chatbot.py
```

### Interactive Chat

Once running, you can:

1. **Ask questions** about the content in the `data/` directory
2. **View sources** - See which documents the answer came from
3. **Type 'exit'** to quit the chat

Example queries:
- "What is Python?"
- "Explain machine learning"
- "What is RAG and how does it work?"
- "Tell me about vector databases"

### Sample Interaction

```
🤖 RAG Chatbot - Interactive Mode
============================================================

You: What is a vector embedding?
⏳ Thinking...

Bot: A vector embedding is a numerical representation of text, images, or 
other data in a high-dimensional space. They capture semantic meaning and 
relationships. For example, the word "king" and "queen" would have similar 
embeddings, while "king" and "table" would be more different.

Show sources? (y/n): y

📚 Sources:

--- Source 1 ---
Vector embeddings are numerical representations of text, images, or other data 
in a high-dimensional space. They capture semantic meaning and relationships...
```

## 📖 Understanding the Code

### How Document Chunking Works

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Size of each chunk
    chunk_overlap=200,    # Overlap to maintain context
    separators=["\n\n", "\n", " ", ""]  # Split priority
)
```

This ensures:
- Documents are split into manageable pieces
- Context is preserved across chunks
- Semantic boundaries are respected

### How Vector Embedding Works

```python
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)
```

This converts text to high-dimensional vectors:
- Similar texts have similar vectors
- Used for similarity search
- Enables semantic understanding

### How Retrieval Works

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Return top 3 most similar documents
)
```

The retriever:
1. Converts your question to a vector
2. Calculates similarity to all document chunks
3. Returns the 3 most similar chunks
4. Passes them to the LLM for context

### How Generation Works

The LLM receives:
1. Your question
2. Retrieved document chunks as context
3. System prompt instructing it to use the context
4. Generates a response grounded in the provided information

## 🎯 Adding Your Own Documents

1. Create `.txt` files in the `data/` directory
2. Add any text content you want to search
3. The chatbot will automatically:
   - Load all `.txt` files
   - Generate embeddings
   - Store in vector database
   - Make them searchable

Example:

```bash
# Create data/company_docs.txt with your content
echo "Your documentation here" > data/company_docs.txt

# Run the chatbot - it will automatically include the new file
python chatbot.py
```

## 🔄 How Chroma Database Works

### First Run
```
Load Documents → Generate Embeddings → Create Chroma DB → Store on Disk
```

### Subsequent Runs
```
Load from Disk → Use Existing Embeddings → Search Faster
```

The `chroma_db/` directory contains:
- Indexed vectors for fast search
- Metadata about documents
- Embeddings for all chunks

To reset and rebuild:
```bash
# Delete the chroma_db directory
rm -r chroma_db  # On Windows: rmdir /s chroma_db
```

## 🧠 Key Concepts Explained

### Embeddings
Vector representations of text that capture semantic meaning. Similar texts have similar embeddings.

### Vector Distance
Measures similarity between embeddings. Smaller distance = more similar.

### Retrieval
Finding relevant documents by comparing vector distances.

### Augmentation
Adding retrieved documents as context to the LLM prompt.

### Generation
LLM using the context to generate an informed response.

## ⚙️ Configuration

### Adjust Number of Retrieved Documents

In `chatbot.py`, change the `k` parameter:

```python
retriever = self.vector_store.create_retriever(k=5)  # Retrieve top 5 instead of 3
```

### Adjust Chunk Size

In `document_loader.py`:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,      # Larger chunks
    chunk_overlap=500,    # More overlap
)
```

### Change LLM Temperature

In `chatbot.py`:

```python
self.llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.3,  # Lower = more deterministic, Higher = more creative
)
```

## 🐛 Troubleshooting

### "GOOGLE_API_KEY not found"
- Make sure you created `.env` file in project root
- Check the API key is correct
- Verify it has Gemini API access enabled

### "No documents found"
- Ensure `.txt` files are in the `data/` directory
- Check file encoding is UTF-8
- Verify files have readable text content

### Slow first run
- First run generates embeddings - this takes time
- Subsequent runs use cached embeddings and are much faster

### Import errors
- Reinstall requirements: `pip install -r requirements.txt`
- Make sure you're using the virtual environment
- Check Python version (3.8+)

## 📚 External Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Chroma Documentation](https://docs.trychroma.com/)
- [Google Gemini API](https://ai.google.dev/)
- [RAG Paper](https://arxiv.org/abs/2005.11401)

## 🎓 Learning Path

1. **Start here**: Run the basic chatbot with sample documents
2. **Explore**: Modify configuration and see the effects
3. **Extend**: Add your own documents and test retrieval
4. **Optimize**: Tune parameters for better results
5. **Advanced**: Integrate with your own applications

## 📝 Example Customization

### Create a Domain-Specific RAG

```python
# Create data/medical_docs.txt with medical information
# The chatbot automatically becomes a medical Q&A assistant

# Or create data/legal_docs.txt
# Now it answers legal questions
```

### Production Deployment

```python
# Use different vector database in production
from langchain.vectorstores import Pinecone
# Or Weaviate, Milvus, etc.

# Add authentication, logging, monitoring
# Use async processing for scalability
# Deploy with FastAPI or similar
```

## 📄 License

This project is provided as-is for educational purposes.

## 🤝 Contributing

Feel free to extend this project with:
- Support for PDFs, web pages, databases
- Multi-language support
- Advanced filtering
- Caching for repeated queries
- Integration with other LLMs
- REST API wrapper

---

**Happy Learning! 🚀**

For questions about RAG, Chroma, or Gemini, check the knowledge base within the `data/` directory by asking the chatbot!
"# chatbot" 
