# RAG Pipeline Test

A RAG (Retrieval-Augmented Generation) pipeline implementation supporting both Ollama and OpenAI language models for testing and comparison.

## Features

- **Dual LLM Support**: Switch between Ollama (local) and OpenAI (cloud) models
- **Document Ingestion**: Process PDF documents using Docling for advanced parsing
- **Vector Database**: ChromaDB for document storage and retrieval
- **Hierarchical Chunking**: Parent-child document structure for better context
- **FastAPI Backend**: RESTful API with interactive documentation
- **Web Interface**: Simple frontend for testing queries

## Installation

1. Clone the repository and navigate to the project directory:
```bash
cd RAGPipelineTest
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables by copying the example file:
```bash
cp .env.example .env
```

4. Edit `.env` with your configuration (see Configuration section below).

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure the following variables:

#### Required (Common)
- `CHROMA_HOST`: ChromaDB host (default: `localhost:8000`)
- `COLLECTION_NAME`: Name for child document collection
- `COLLECTION_NAME_PARENTS`: Name for parent document collection
- `CHUNK_SIZE`: Document chunk size (default: `1000`)
- `CHUNK_OVERLAP`: Chunk overlap size (default: `100`)

#### Ollama Configuration (Default)
- `OLLAMA_BASE_URL`: Ollama API URL (default: `http://localhost:11434`)
- `GENERATION_MODEL`: Ollama model for text generation (e.g., `llama3.2:latest`)
- `EMBEDDING_MODEL`: Ollama model for embeddings (e.g., `nomic-embed-text:latest`)

#### OpenAI Configuration (Optional)
- `OPENAI_API_KEY`: Your OpenAI API key (required when using `--openai`)
- `OPENAI_MODEL`: OpenAI model name (default: `gpt-3.5-turbo`)
- `OPENAI_EMBEDDING_MODEL`: OpenAI embedding model (default: `text-embedding-ada-002`)

### Prerequisites

#### For Ollama (Default)
1. Install [Ollama](https://ollama.ai/)
2. Pull required models:
```bash
ollama pull llama3.2:latest
ollama pull nomic-embed-text:latest
```
3. Start Ollama service (usually runs automatically)

#### For OpenAI
1. Get an API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Set `OPENAI_API_KEY` in your `.env` file
3. Install OpenAI dependencies (included in requirements.txt):
```bash
pip install langchain-openai
```

#### ChromaDB
Start ChromaDB server:
```bash
# Using Docker (recommended)
docker run -p 8000:8000 chromadb/chroma

# Or using Docker Compose (if available)
docker-compose up chroma
```

## Usage

### 1. Ingest Documents

Process a PDF document to prepare it for RAG queries:

#### Using Ollama Embeddings (Default)
```bash
python rag_cli.py ingest path/to/your/document.pdf
```

#### Using OpenAI Embeddings
```bash
python rag_cli.py ingest path/to/your/document.pdf --openai
```

This will:
- Parse the PDF using Docling
- Extract page and structural information
- Create hierarchical chunks (articles and paragraphs)
- Generate embeddings using the specified provider (Ollama or OpenAI)
- Store documents in ChromaDB collections

**Note**: When using `--openai`, make sure your `OPENAI_API_KEY` is configured in your `.env` file.

### 2. Start the Server

#### Using Ollama (Default)
```bash
python rag_cli.py server
```

#### Using OpenAI
```bash
python rag_cli.py server --openai
```

#### Custom Port
```bash
python rag_cli.py server --port 8500 --openai
```

### 3. Test the API

Once the server is running:

- **Interactive API Docs**: http://localhost:8500/docs
- **Web Interface**: http://localhost:8500/
- **Health Check**: http://localhost:8500/ask (POST)

### Example API Usage

```bash
curl -X POST "http://localhost:8500/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What is the speed limit in urban areas?",
       "k": 5
     }'
```

## LLM Abstraction

The application uses an abstraction layer that allows seamless switching between different LLM providers:

### Architecture

```python
# Abstract base class
class LLMProvider(ABC):
    def invoke(self, messages, **kwargs) -> str: ...
    def get_embedding_model(self): ...

# Concrete implementations
class OllamaProvider(LLMProvider): ...
class OpenAIProvider(LLMProvider): ...
```

### Supported Providers

| Provider | Models | Pros | Cons |
|----------|---------|------|------|
| **Ollama** | llama3.2, mistral, nomic-embed-text | Free, Private, Local | Requires local setup, slower |
| **OpenAI** | GPT-3.5/4, text-embedding-ada-002 | Fast, High quality | Costs money, requires API key |

### Testing the Abstraction

Run the test script to verify both providers work:

```bash
python test_llm_abstraction.py
```

## Project Structure

```
RAGPipelineTest/
├── api_server.py              # FastAPI server with LLM abstraction
├── rag_cli.py                 # CLI for ingestion and server management
├── chroma_client.py           # ChromaDB client configuration
├── test_llm_abstraction.py    # Test script for LLM providers
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── frontend/                 # Web interface
└── README.md                # This file
```

## Advanced Usage

### Query Processing Pipeline

1. **Document Ingestion**: Parse PDF, create embeddings (Ollama or OpenAI)
2. **Query Preprocessing**: Extract filters and rewrite for synonyms
3. **Retrieval**: Search child chunks, expand to parent documents
4. **Generation**: Use LLM to generate response with citations
5. **Response**: Return answer with source documents

### Mixed Provider Usage

You can mix and match providers for different operations:

```bash
# Ingest with OpenAI embeddings
python rag_cli.py ingest document.pdf --openai

# Run server with Ollama LLM (but OpenAI embeddings from ingestion)
python rag_cli.py server

# Or run server with OpenAI LLM too
python rag_cli.py server --openai
```

**Important**: The embedding model used for ingestion should ideally match the one used for query-time retrieval for best results, though cross-provider compatibility is generally maintained.

### Custom Models

#### Ollama
Add custom models to your Ollama installation:
```bash
ollama pull custom-model:latest
```
Update `GENERATION_MODEL` in `.env`.

#### OpenAI
Use different OpenAI models by updating `.env`:
```
OPENAI_MODEL=gpt-4
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

### Performance Tuning

- **Chunk Size**: Adjust `CHUNK_SIZE` based on your documents
- **Retrieval Count**: Modify `k` parameter in queries
- **Temperature**: Lower for factual responses, higher for creative
- **Model Selection**: GPT-4 for quality, GPT-3.5 for speed
- **Embedding Strategy**: 
  - OpenAI embeddings are faster and often higher quality
  - Ollama embeddings are free and private
  - Consider using OpenAI for ingestion, Ollama for generation to balance cost/quality

## Troubleshooting

### Common Issues

1. **ChromaDB Connection Error**
   - Ensure ChromaDB is running on the configured port
   - Check `CHROMA_HOST` in your `.env`

2. **Ollama Model Not Found**
   - Pull the required models: `ollama pull model-name`
   - Verify model names with: `ollama list`

3. **OpenAI API Errors**
   - Check your API key is valid and has credits
   - Verify model names are correct
   - Check API rate limits

4. **Import Errors**
   - Install all dependencies: `pip install -r requirements.txt`
   - For OpenAI: `pip install langchain-openai`

5. **Embedding Compatibility**
   - If you ingested with `--openai` but run server without it, ensure compatibility
   - For best results, use the same embedding provider for ingestion and queries
   - Mixed setups work but may have slightly reduced retrieval accuracy

### Debug Mode

Enable debug logging by setting environment variables:
```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_VERBOSE=true
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation at `/docs`
3. Run the test script to verify configuration
4. Open an issue on the repository