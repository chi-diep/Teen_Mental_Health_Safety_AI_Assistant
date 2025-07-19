# Teen Safety & Mental Health AI Assistant

A web app using a hybrid RAG pipeline with Flask, JSON-based knowledge base, LangChain, Ollama, and HuggingFace embeddings.

## Why This Project Matters

Many parents and caregivers want to support teens through digital safety, mental health, and crisis navigation, but often lack reliable tools or real-time resources.

**Guardiané** is an AI-powered assistant that answers natural language questions such as:

- "What should I know about cyberbullying?"
- "Where can I find support hotlines?"
- "How can I talk to my teen about screen addiction?"

The app combines Retrieval-Augmented Generation (RAG) with lightweight statistical logic to provide empathetic and accurate responses instantly.

## Data Sources

The system uses structured JSON files as its knowledge base:

- `faq.json`: Curated Q&A covering safety, addiction, parenting, and wellness.
- `hotlines.json`: A list of verified support organizations with contact details.

These files are embedded into a vector store using MiniLM for semantic retrieval.

## How the Hybrid RAG Works

### Data Loading

- Parses and preprocesses entries from the `faq.json` and `hotlines.json` files.
- Converts them into natural language chunks stored as LangChain `Document` objects.

### Embedding and Storage

- Uses HuggingFace `all-MiniLM-L6-v2` model to generate embeddings.
- Stores embeddings in ChromaDB for persistent vector search.

### Answering Pipeline

- If a question involves statistical keywords (count, total, max, etc.), the app uses direct JSON processing.
- For contextual or open-ended questions, the app uses Ollama with a TinyLLaMA model to generate conversational answers via LangChain.

## Example Queries

| Category            | Sample Question                                          |
|---------------------|----------------------------------------------------------|
| Statistical         | How many hotlines are available?                         |
| Practical Guidance  | How do I talk to my teen about screen time?              |
| Resource Retrieval  | Where can I get help for anxiety or depression?          |
| Conversational      | Hello, who are you?                                      |

## Technologies Used

- Python
- Flask
- LangChain and LangChain Ollama
- HuggingFace sentence-transformers
- ChromaDB
- Ollama (TinyLLaMA model)
- HTML, CSS, and JavaScript (for frontend)
- PWA manifest and service worker
