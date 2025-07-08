# Claude AI Assistant Instructions

This file contains instructions and context for Claude AI to better assist with this project.

## Project Overview

This is a **Multi-Site RAG (Retrieval-Augmented Generation) System** designed to crawl multiple websites and create a unified, searchable knowledge base. While initially focused on HubSpot's knowledge base, the system is architected to be extensible for any documentation or content website.

### Key Features
- **Multi-site crawling** with configurable strategies (breadth-first, depth-first, sitemap-based)
- **Intelligent content extraction** using CSS selectors and content cleaning
- **URL normalization and deduplication** to prevent duplicate crawling
- **Vector embeddings** using ChromaDB and sentence-transformers
- **RESTful API** with FastAPI for programmatic access
- **Web interface** for easy interaction and management
- **Configurable crawling limits** and politeness settings
- **Content filtering** for language, file types, and URL patterns

## Architecture

### Core Components
- `src/generic_crawler.py` - Main crawling engine with multiple strategies
- `src/multi_site_vector_store.py` - Vector database management with ChromaDB
- `src/site_config.py` - Site configuration management (loads from JSON)
- `src/multi_site_main.py` - CLI interface and system orchestration
- `src/site_management_api.py` - FastAPI REST endpoints
- `multi_site_web_interface.html` - Browser-based UI

### Data Storage
- `data/sites_config.json` - Site configurations (JSON format)
- `data/*_articles.json` - Crawled article data per site
- `chroma_db/` - Vector embeddings and search index
- `logs/` - System and crawl logs

## Development Commands

### Setup
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### CLI Operations
```bash
# Initial setup with HubSpot site
python src/multi_site_main.py setup

# Crawl all active sites
python src/multi_site_main.py crawl-all

# Crawl specific site
python src/multi_site_main.py crawl --site-id <site-id>

# Start API server
python src/multi_site_main.py api

# Search content
python src/multi_site_main.py search --query "your search query"

# List configured sites
python src/multi_site_main.py list

# System statistics
python src/multi_site_main.py stats
```

### API Server
```bash
# Start on default port (8001)
python src/multi_site_main.py api

# Custom host/port
python src/multi_site_main.py api --host 0.0.0.0 --port 8080
```

## Configuration Management

### Site Configuration Structure
Sites are configured in `data/sites_config.json` with the following structure:
- **Basic info**: name, description, base_url, start_urls
- **Crawl strategy**: breadth_first, depth_first, sitemap, url_list
- **Limits**: max_articles, max_depth, delay_seconds, timeout_seconds
- **Content selectors**: CSS selectors for title, content, description, category
- **URL filtering**: allowed_patterns, blocked_patterns (regex)
- **Processing options**: chunk_size, chunk_overlap, language

### Adding New Sites
```bash
python src/multi_site_main.py add-site \
  --template documentation \
  --name "Your Site Name" \
  --url "https://example.com" \
  --start-urls "https://example.com/docs" \
  --description "Site description"
```

## Code Style Guidelines

### Python Standards
- Follow PEP 8 style guidelines
- Use type hints for all function parameters and returns
- Implement proper error handling with try/catch blocks
- Use async/await for I/O operations
- Log important operations and errors appropriately

### URL Handling
- Always use `_normalize_url()` for logging and deduplication
- Implement proper URL validation and filtering
- Respect robots.txt and crawl delays
- Handle relative URLs properly with `urljoin()`

### Configuration
- Use JSON for configuration, not Python code
- Validate configuration on load
- Provide sensible defaults for optional settings
- Use dataclasses for type safety

## Important Implementation Notes

### Crawling Best Practices
- **URL Normalization**: All URLs are normalized (removing query params, fragments) for deduplication
- **Politeness**: Respect crawl delays, robots.txt, and rate limits
- **Content Quality**: Filter for English content, meaningful word counts, and valid HTML
- **Error Handling**: Log failures but continue crawling other URLs
- **Memory Management**: Process articles in batches to avoid memory issues

### Vector Store Management
- **Embeddings**: Use sentence-transformers with 'all-MiniLM-L6-v2' model
- **Chunking**: Split articles into overlapping chunks for better retrieval
- **Metadata**: Store comprehensive metadata for filtering and ranking
- **Collections**: Separate site-specific and global collections for flexible querying

### Security Considerations
- Validate all user inputs, especially URLs and site configurations
- Sanitize HTML content before processing
- Implement proper authentication for API endpoints (when needed)
- Avoid exposing internal paths or sensitive information in logs

## Debugging and Monitoring

### Log Files
- `logs/multi_site_rag.log` - Main system logs
- `logs/audit.log` - Configuration changes and system events

### Common Issues
- **robots.txt blocking**: Check `respect_robots_txt` setting in site config
- **Empty content**: Verify CSS selectors match target site structure
- **Memory issues**: Reduce `max_articles` or `chunk_size` for large crawls
- **Import errors**: Ensure virtual environment is activated and dependencies installed

### Performance Monitoring
- Monitor crawl completion rates and article counts
- Check vector store size and query performance
- Review failed URLs and error patterns in logs

## Future Enhancements

### Planned Features
- **Authentication integration** for protected content
- **Incremental crawling** to update only changed content
- **Content classification** and automatic tagging
- **Multi-language support** beyond English
- **Advanced search features** with filters and ranking
- **Scheduled crawling** with cron-like functionality

### Extension Points
- **New crawl strategies** in `generic_crawler.py`
- **Custom content extractors** for specific site types
- **Additional vector store backends** beyond ChromaDB
- **Custom embedding models** for domain-specific content
- **Integration APIs** for external systems