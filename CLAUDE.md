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
- **Crawl job management** with pause/stop/resume functionality
- **Progress tracking** with detailed status reporting
- **Error handling** and recovery mechanisms
- **Security features** including rate limiting and input validation

## Architecture

### Core Components
- `src/generic_crawler.py` - Main crawling engine with multiple strategies
- `src/multi_site_vector_store.py` - Vector database management with ChromaDB
- `src/site_config.py` - Site configuration management (loads from JSON)
- `src/multi_site_main.py` - CLI interface and system orchestration
- `src/site_management_api.py` - FastAPI REST endpoints with crawl job management
- `src/crawl_progress.py` - Progress tracking and job state management
- `src/enhanced_api.py` - Enhanced API server with security and monitoring
- `src/security.py` - Security utilities and rate limiting
- `src/performance.py` - Performance monitoring and optimization
- `src/error_handling.py` - Centralized error handling and logging
- `src/advanced_chunking.py` - Advanced text chunking strategies
- `multi_site_web_interface.html` - Browser-based UI

### Data Storage
- `data/sites_config.json` - Site configurations (JSON format)
- `data/*_articles.json` - Crawled article data per site
- `data/progress/` - Crawl job progress files
- `chroma_db/` - Vector embeddings and search index
- `logs/` - System and crawl logs
- `backups/` - System configuration backups
- `tests/` - Unit tests and test data

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

# Site management
python src/multi_site_main.py add-site --template documentation --name "Site Name" --url "https://example.com" --start-urls "https://example.com/docs"

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

# Backup system
python src/multi_site_main.py backup
```

### API Server
```bash
# Start on default port (8001)
python src/multi_site_main.py api

# Custom host/port
python src/multi_site_main.py api --host 0.0.0.0 --port 8080

# Enhanced API with security features
python src/enhanced_api.py --enable-auth --port 8000
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
- **Parallelization**: max_concurrent_requests, max_concurrent_sites, enable_parallel_processing, batch_size, thread_pool_size

### Adding New Sites
```bash
python src/multi_site_main.py add-site \
  --template documentation \
  --name "Your Site Name" \
  --url "https://example.com" \
  --start-urls "https://example.com/docs" \
  --description "Site description"
```

### Available Templates
- **documentation**: For documentation websites (GitBook, Notion, etc.)
- **blog**: For blog and news websites
- **support**: For support and FAQ websites

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
- **Parallelization**: Configure concurrent requests and sites based on server capacity and politeness requirements

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

## API Endpoints

### Site Management
- `GET /sites` - List all site configurations
- `POST /sites` - Create new site configuration
- `GET /sites/{site_id}` - Get specific site configuration
- `PUT /sites/{site_id}` - Update site configuration
- `DELETE /sites/{site_id}` - Delete site configuration
- `POST /sites/from-template` - Create site from template

### Crawl Job Management
- `POST /crawl/start` - Start crawling job
- `GET /crawl/{job_id}/status` - Get crawl job status
- `GET /crawl/{job_id}/progress` - Get detailed progress information
- `POST /crawl/{job_id}/pause` - Pause crawl job
- `POST /crawl/{job_id}/stop` - Stop crawl job
- `POST /crawl/{job_id}/resume` - Resume paused crawl job
- `POST /crawl/{job_id}/recover` - Recover interrupted crawl job
- `GET /crawl/jobs` - List all crawl jobs
- `GET /crawl/recoverable` - Get recoverable jobs

### System Health
- `GET /health` - System health check
- `GET /templates` - Get available site templates

## Debugging and Monitoring

### Log Files
- `logs/multi_site_rag.log` - Main system logs
- `logs/audit.log` - Configuration changes and system events

### Common Issues
- **robots.txt blocking**: Check `respect_robots_txt` setting in site config
- **Empty content**: Verify CSS selectors match target site structure
- **Memory issues**: Reduce `max_articles` or `chunk_size` for large crawls
- **Import errors**: Ensure virtual environment is activated and dependencies installed
- **Job interruptions**: Use the recovery endpoints to resume interrupted crawls

### Performance Monitoring
- Monitor crawl completion rates and article counts
- Check vector store size and query performance
- Review failed URLs and error patterns in logs
- Track job progress and completion times

## Future Enhancements

### Planned Features
- **Authentication integration** for protected content
- **Incremental crawling** to update only changed content
- **Content classification** and automatic tagging
- **Multi-language support** beyond English
- **Advanced search features** with filters and ranking
- **Scheduled crawling** with cron-like functionality
- **Real-time websocket updates** for crawl progress
- **Distributed crawling** across multiple workers

### Extension Points
- **New crawl strategies** in `generic_crawler.py`
- **Custom content extractors** for specific site types
- **Additional vector store backends** beyond ChromaDB
- **Custom embedding models** for domain-specific content
- **Integration APIs** for external systems
- **Custom progress tracking** and notification systems
- **Advanced chunking strategies** for different content types

## Testing

### Unit Tests
- `tests/test_url_normalization.py` - URL normalization tests
- `tests/test_pause_stop_resume.py` - Crawl job control tests

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_url_normalization.py

# Run with coverage
python -m pytest --cov=src tests/
```

## Dependencies

### Required Dependencies
- `crawl4ai==0.6.3` - AI-powered web crawling
- `chromadb==0.5.23` - Vector database
- `fastapi==0.115.6` - Web framework
- `uvicorn==0.34.0` - ASGI server
- `sentence-transformers==3.3.1` - Text embeddings
- `beautifulsoup4==4.13.4` - HTML parsing
- `requests==2.32.4` - HTTP requests
- `pydantic==2.11.7` - Data validation
- `python-dotenv==1.1.1` - Environment variables
- `psutil==7.0.0` - System monitoring
- `bleach==6.2.0` - HTML sanitization
- `langdetect==1.0.9` - Language detection
- `spacy==3.8.2` - NLP processing
- `scikit-learn==1.6.0` - Machine learning utilities
- `numpy==2.3.1` - Numerical computing
- `anthropic==0.45.0` - AI model integration

## Parallelization Configuration

### Parallelization Settings
The system supports various parallelization configurations to optimize crawling performance:

#### Configuration Options
- **max_concurrent_requests**: Number of simultaneous HTTP requests per site (default: 5, range: 1-20)
- **max_concurrent_sites**: Number of sites to crawl simultaneously (default: 3, range: 1-10)
- **enable_parallel_processing**: Enable/disable parallel processing (default: true)
- **batch_size**: Number of URLs to process in each batch (default: 10, range: 1-100)
- **thread_pool_size**: Size of thread pool for concurrent operations (default: 4, range: 1-16)

#### Template Defaults
- **Documentation sites**: 5 concurrent requests, 2 concurrent sites, batch size 15
- **Blog sites**: 3 concurrent requests, 2 concurrent sites, batch size 8
- **Support sites**: 4 concurrent requests, 2 concurrent sites, batch size 12

#### Best Practices
- **Start conservative**: Begin with lower concurrency settings and increase based on performance
- **Monitor server response**: Watch for rate limiting or server overload indicators
- **Respect robots.txt**: Parallelization should not violate crawl-delay directives
- **Balance politeness and speed**: Higher concurrency may trigger anti-bot measures
- **Consider site capacity**: Popular sites can handle more concurrent requests
- **Test incrementally**: Increase settings gradually while monitoring success rates