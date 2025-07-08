# HubSpot RAG Generator

A Retrieval-Augmented Generation (RAG) system that crawls HubSpot's knowledge base and provides intelligent Q&A capabilities using Anthropic's Claude.

## Features

- **Intelligent Crawling**: Uses crawl4ai to systematically crawl HubSpot's knowledge base
- **Smart Processing**: Chunks documents optimally for RAG with overlap handling
- **Vector Search**: ChromaDB-powered semantic search with sentence transformers
- **Claude Integration**: Uses Anthropic's Claude for intelligent answer generation
- **Multiple Interfaces**: Command-line, web interface, and REST API
- **Category Filtering**: Filter searches by HubSpot product areas
- **Source Attribution**: All answers include source links and confidence scores

## Prerequisites

- Python 3.8+
- Anthropic API key
- At least 2GB of free disk space for the vector database

## Installation

1. **Clone or download the project**
2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

5. **Install Playwright browsers** (required for crawl4ai):
   ```bash
   playwright install
   ```

## Quick Start

### Option 1: Run Full Pipeline (Recommended)
```bash
python main.py full
```
This will:
1. Crawl HubSpot knowledge base
2. Process and chunk documents
3. Index in vector database
4. Prepare the system for queries

### Option 2: Run Components Separately
```bash
# Step 1: Crawl HubSpot knowledge base
python main.py crawl

# Step 2: Process documents into chunks
python main.py process

# Step 3: Index in vector database
python main.py index
```

## Usage

### Command Line Interface
```bash
python main.py chat
```

### Web Interface
1. Start the API server:
   ```bash
   python main.py api
   ```

2. Open `web_interface.html` in your browser

### REST API
Start the server:
```bash
python main.py api --host 0.0.0.0 --port 8000
```

API endpoints:
- `POST /query` - Query the RAG system
- `GET /health` - Health check
- `GET /stats` - Vector store statistics

Example API request:
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "How do I create a contact in HubSpot?"}'
```

## Configuration

### Environment Variables
- `ANTHROPIC_API_KEY`: Your Anthropic API key (required)
- `CHROMA_PERSIST_DIRECTORY`: Vector database directory (default: `./chroma_db`)
- `MAX_CRAWL_DEPTH`: Maximum crawl depth (default: 3)
- `CRAWL_DELAY`: Delay between requests in seconds (default: 1)

### Customization
- **Chunk size**: Modify `chunk_size` in `data_processor.py`
- **Embedding model**: Change `SentenceTransformer` model in `vector_store.py`
- **Claude model**: Update model name in `rag_system.py`

## Project Structure

```
hubspot-rag-generator/
├── src/
│   ├── crawler.py          # Web crawler using crawl4ai
│   ├── data_processor.py   # Document processing and chunking
│   ├── vector_store.py     # ChromaDB vector database
│   └── rag_system.py       # RAG system with Claude integration
├── data/                   # Crawled data and processed chunks
├── chroma_db/             # Vector database files
├── main.py                # Main orchestrator script
├── web_interface.html     # Simple web interface
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
└── README.md             # This file
```

## Commands Reference

```bash
# Full pipeline
python main.py full [--max-articles 100]

# Individual components
python main.py crawl [--max-articles 100]
python main.py process
python main.py index

# Interfaces
python main.py chat
python main.py api [--host 0.0.0.0] [--port 8000]
```

## Example Questions

The system can answer questions like:
- "How do I create a new contact in HubSpot?"
- "What are the different deal stages in HubSpot?"
- "How do I set up email automation workflows?"
- "How do I integrate HubSpot with other tools?"
- "What is the difference between contacts and companies?"

## Troubleshooting

### Common Issues

1. **"No module named 'crawl4ai'"**
   - Make sure you're in the virtual environment
   - Run `pip install -r requirements.txt`

2. **"API server is not running"**
   - Start the API server with `python main.py api`
   - Check if port 8000 is available

3. **"Vector store is empty"**
   - Run the full pipeline first: `python main.py full`
   - Or run components separately: crawl → process → index

4. **Crawling fails**
   - Check your internet connection
   - Some sites may block automated requests
   - Try reducing `MAX_CRAWL_DEPTH` in .env

5. **Out of memory during indexing**
   - Reduce chunk size in `data_processor.py`
   - Process smaller batches

### Performance Tips

- **Faster crawling**: Reduce `CRAWL_DELAY` (but be respectful)
- **Better chunks**: Adjust `chunk_size` and `chunk_overlap`
- **Smaller database**: Limit `max_articles` during crawling
- **Faster search**: Use smaller embedding model in `vector_store.py`

## Development

### Adding New Features

1. **Custom crawling targets**: Modify `start_urls` in `crawler.py`
2. **Different LLM**: Replace Anthropic client in `rag_system.py`
3. **Advanced chunking**: Enhance `DocumentProcessor` class
4. **Better UI**: Improve `web_interface.html`

### Testing

```bash
# Test individual components
python src/crawler.py
python src/data_processor.py
python src/vector_store.py
python src/rag_system.py
```

## License

MIT License - feel free to use and modify as needed.

## Contributing

1. Fork the project
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the logs in the `logs/` directory
3. Open an issue on the project repository

---

**Note**: This tool is for educational and research purposes. Please respect HubSpot's terms of service and robots.txt when crawling their content.