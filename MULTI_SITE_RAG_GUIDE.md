# 🌐 Multi-Site RAG System Guide

A flexible, scalable RAG (Retrieval-Augmented Generation) system that can crawl and index multiple websites simultaneously, providing unified search across all your knowledge sources.

## ✨ Features

### 🚀 **Multi-Site Management**
- Configure unlimited websites with individual settings
- Pre-built templates for common site types (Documentation, Blog, Support)
- Flexible crawling strategies (Breadth-first, Depth-first, Sitemap, URL list)
- Site-specific content selectors and filters

### 🤖 **Advanced Crawling**
- Respects robots.txt and rate limits
- Handles JavaScript-rendered content
- Automatic content extraction and cleaning
- Duplicate detection and filtering
- Configurable crawl depth and article limits

### 🔍 **Intelligent Search**
- Multi-site semantic search using sentence transformers
- Site-specific or cross-site queries
- Category-based filtering
- Similarity scoring and ranking
- Real-time search across all indexed content

### 📊 **Management Interface**
- Modern web interface for site configuration
- Real-time crawl monitoring and status tracking
- Comprehensive statistics and analytics
- Template-based quick setup
- RESTful API for programmatic access

### 🔧 **Enterprise Ready**
- Persistent vector database with ChromaDB
- Configurable chunking and embedding strategies
- Backup and restore functionality
- Comprehensive logging and error handling
- Scalable architecture for high-volume deployments

## 🛠️ Installation & Setup

### Prerequisites
```bash
# Python 3.8+
python --version

# Virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies
```bash
# Install core dependencies
pip install -r requirements-simple.txt

# Install additional crawling dependencies
pip install crawl4ai python-dotenv beautifulsoup4 requests

# Install Playwright browsers for JavaScript rendering
playwright install
```

### Environment Configuration
Your `.env` file should contain:
```env
# Required: Your Anthropic API Key
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: System Configuration
MAX_CRAWL_DEPTH=3
CRAWL_DELAY=1
USER_AGENT=MultiSiteRAGBot/1.0
LOG_LEVEL=INFO
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

## 🚀 Quick Start

### Option 1: Web Interface (Recommended)
```bash
# 1. Initial setup
python src/multi_site_main.py setup

# 2. Start the API server
python src/multi_site_main.py api

# 3. Open web interface
open multi_site_web_interface.html
```

### Option 2: Command Line
```bash
# Add HubSpot Knowledge Base
python src/multi_site_main.py add-site \
  --template documentation \
  --name "HubSpot Knowledge Base" \
  --url "https://knowledge.hubspot.com" \
  --start-urls "https://knowledge.hubspot.com/articles" \
  --description "HubSpot's comprehensive knowledge base"

# Crawl all sites
python src/multi_site_main.py crawl-all

# Sync with vector store
python src/multi_site_main.py sync

# Search across all sites
python src/multi_site_main.py search --query "how to create contacts"
```

## 📖 Detailed Usage

### Site Configuration Templates

#### Documentation Sites
Perfect for GitBook, Notion, API docs, and technical documentation:
```python
# Optimized for structured documentation
- Content selectors: .content, .documentation-content, article
- Strategy: Breadth-first crawling
- Default categories: Getting Started, API, Tutorials, FAQ
```

#### Blog Sites
Ideal for news sites, company blogs, and article collections:
```python
# Optimized for article-based content
- Content selectors: .post-content, .article-content, .entry-content
- Strategy: Sitemap-based crawling
- Default categories: News, Tutorials, Updates, Tips
```

#### Support Sites
Designed for help centers, FAQ pages, and support documentation:
```python
# Optimized for Q&A and support content
- Content selectors: .answer, .faq-content, .support-content
- Strategy: Breadth-first crawling
- Default categories: FAQ, Troubleshooting, How-to, Known Issues
```

### Advanced Site Configuration

#### Custom Content Selectors
```python
{
    "selectors": {
        "title": "h1, .page-title, .article-title",
        "content": ".main-content, article, .post-body",
        "description": ".description, .summary, .excerpt",
        "category": ".category, .tag, .section",
        "exclude": [".sidebar", ".footer", ".ads", ".comments"]
    }
}
```

#### Crawling Limits
```python
{
    "limits": {
        "max_articles": 500,        # Maximum articles to crawl
        "max_depth": 4,             # Maximum link depth
        "delay_seconds": 1.0,       # Delay between requests
        "timeout_seconds": 30,      # Request timeout
        "max_file_size_mb": 10,     # Maximum file size
        "respect_robots_txt": true, # Respect robots.txt
        "follow_redirects": true    # Follow HTTP redirects
    }
}
```

#### URL Filtering
```python
{
    "allowed_patterns": [
        "https://example.com/docs/.*",
        "https://example.com/blog/.*"
    ],
    "blocked_patterns": [
        ".*\\.pdf$",
        ".*\\.zip$",
        ".*/privacy.*",
        ".*/terms.*"
    ]
}
```

### Search and Query

#### Basic Search
```python
# Search across all sites
results = system.search("machine learning tutorials")

# Search specific sites
results = system.search("API documentation", site_ids=["site_1", "site_2"])

# Limit results
results = system.search("python examples", n_results=5)
```

#### API Search
```bash
# Search via API
curl -X GET "http://localhost:8001/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "user authentication", "site_ids": ["hubspot_site"], "n_results": 10}'
```

#### Advanced Filtering
```python
# Filter by category
results = vector_store.query_global(
    query="troubleshooting",
    where={"category": "FAQ"},
    n_results=10
)

# Filter by content type
results = vector_store.query_global(
    query="API endpoints",
    where={"content_type": "documentation"},
    n_results=10
)
```

## 🔧 API Reference

### Site Management
```bash
# List all sites
GET /sites

# Get specific site
GET /sites/{site_id}

# Create new site
POST /sites
{
    "name": "Example Site",
    "base_url": "https://example.com",
    "start_urls": ["https://example.com/docs"],
    "crawl_strategy": "breadth_first",
    "content_type": "documentation"
}

# Update site
PUT /sites/{site_id}

# Delete site
DELETE /sites/{site_id}
```

### Crawling
```bash
# Start crawl job
POST /crawl/start
{
    "site_ids": ["site_1", "site_2"],
    "force_recrawl": false
}

# Get crawl status
GET /crawl/{job_id}/status

# List all crawl jobs
GET /crawl/jobs
```

### Templates
```bash
# Get available templates
GET /templates

# Create site from template
POST /sites/from-template
{
    "template_name": "documentation",
    "name": "My Docs",
    "base_url": "https://docs.example.com",
    "start_urls": ["https://docs.example.com/getting-started"]
}
```

## 🎯 Use Cases

### Technical Documentation Hub
```python
# Add multiple documentation sites
sites = [
    {"name": "Python Docs", "url": "https://docs.python.org"},
    {"name": "Django Docs", "url": "https://docs.djangoproject.com"},
    {"name": "React Docs", "url": "https://react.dev"}
]

# Unified search across all documentation
results = system.search("authentication middleware")
```

### Company Knowledge Base
```python
# Add internal and external knowledge sources
sites = [
    {"name": "Internal Wiki", "url": "https://wiki.company.com"},
    {"name": "Support Center", "url": "https://help.company.com"},
    {"name": "Product Blog", "url": "https://blog.company.com"}
]

# Search across all company knowledge
results = system.search("product roadmap Q4")
```

### Research and News Aggregation
```python
# Add multiple news and research sources
sites = [
    {"name": "Tech News", "url": "https://technews.com"},
    {"name": "Research Papers", "url": "https://arxiv.org"},
    {"name": "Industry Reports", "url": "https://reports.industry.com"}
]

# Search across all sources
results = system.search("artificial intelligence trends 2024")
```

## 📊 Monitoring and Analytics

### System Statistics
```python
# Get comprehensive system stats
stats = system.get_system_stats()

# Example output:
{
    "sites": {
        "total": 5,
        "active": 4,
        "by_type": {
            "documentation": 3,
            "blog": 1,
            "support": 1
        }
    },
    "vector_store": {
        "total_documents": 1250,
        "total_sites": 4
    }
}
```

### Crawl Monitoring
```python
# Monitor crawl progress
job_status = await api_call(f'/crawl/{job_id}/status')

# Example output:
{
    "job_id": "crawl_20241201_143022_0",
    "status": "completed",
    "total_articles": 145,
    "sites_crawled": 3,
    "started_at": "2024-12-01T14:30:22",
    "completed_at": "2024-12-01T14:45:33"
}
```

## 🔐 Security and Best Practices

### Rate Limiting
```python
# Configure respectful crawling
"limits": {
    "delay_seconds": 1.0,        # Minimum delay between requests
    "max_articles": 10000,        # Limit to prevent overload
    "respect_robots_txt": false   # Always respect robots.txt
}
```

### Content Filtering
```python
# Exclude sensitive or irrelevant content
"selectors": {
    "exclude": [
        ".comments",      # User comments
        ".ads",          # Advertisement content
        ".sidebar",      # Navigation sidebars
        ".footer",       # Footer content
        ".login-form"    # Login forms
    ]
}
```

### Data Privacy
```python
# Block personal or sensitive pages
"blocked_patterns": [
    ".*/login.*",
    ".*/admin.*",
    ".*/profile.*",
    ".*/settings.*",
    ".*/private.*"
]
```

## 🛠️ Troubleshooting

### Common Issues

#### Crawling Failures
```bash
# Check site accessibility
curl -I "https://example.com"

# Verify robots.txt
curl "https://example.com/robots.txt"

# Check logs
tail -f logs/multi_site_rag.log
```

#### API Connection Issues
```bash
# Test API connectivity
curl http://localhost:8001/health

# Check API logs
python src/multi_site_main.py api --reload
```

#### Vector Store Issues
```bash
# Rebuild vector store
python src/multi_site_main.py sync

# Check vector store stats
python src/multi_site_main.py stats
```

### Performance Optimization

#### Crawling Performance
```python
# Optimize crawling settings
"limits": {
    "max_articles": 200,    # Reduce for faster crawls
    "max_depth": 2,         # Reduce depth for focused crawling
    "delay_seconds": 0.5,   # Reduce delay for faster crawling
    "timeout_seconds": 15   # Reduce timeout for faster failures
}
```

#### Search Performance
```python
# Optimize chunk size for search
"chunk_size": 1000,    # Smaller chunks for precise search
"chunk_overlap": 200,  # Overlap for context preservation
```

## 🔄 Backup and Recovery

### System Backup
```bash
# Create full system backup
python src/multi_site_main.py backup

# Backup specific site
python src/multi_site_main.py backup --site-id site_123
```

### Recovery
```python
# Restore from backup
system.restore_site_data("backups/site_backup_20241201.json")

# Rebuild vector store from crawled data
system.sync_vector_store()
```

## 📈 Scaling and Performance

### Horizontal Scaling
- Deploy multiple crawler instances
- Use Redis for job queuing
- Implement load balancing for API
- Use distributed vector stores

### Vertical Scaling
- Increase memory for larger vector stores
- Use GPU acceleration for embeddings
- Optimize database connections
- Implement caching strategies

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repo-url>
cd multi-site-rag

# Create development environment
python -m venv dev-env
source dev-env/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

### Adding New Templates
```python
# Create new template in site_config.py
SITE_TEMPLATES["custom_type"] = {
    "selectors": {
        "title": "custom-title-selector",
        "content": "custom-content-selector"
    },
    "limits": {
        "max_articles": 150,
        "max_depth": 3
    }
}
```

## 📞 Support

For questions and support:
1. Check this guide and troubleshooting section
2. Review logs in `logs/multi_site_rag.log`
3. Test with the included HubSpot example
4. Check API documentation at `http://localhost:8001/docs`

## 🔄 Migration from Single-Site

### Migrating Existing Setup
```bash
# 1. Backup existing data
cp -r chroma_db chroma_db_backup

# 2. Run migration script
python src/migrate_to_multi_site.py

# 3. Verify migration
python src/multi_site_main.py stats
```

### Configuration Migration
```python
# Old single-site config
OLD_CONFIG = {
    "url": "https://knowledge.hubspot.com",
    "max_articles": 100
}

# New multi-site config
NEW_CONFIG = {
    "name": "HubSpot Knowledge Base",
    "base_url": "https://knowledge.hubspot.com",
    "start_urls": ["https://knowledge.hubspot.com/articles"],
    "limits": {"max_articles": 100}
}
```

---

## 🎉 You're Ready!

Your multi-site RAG system is now configured and ready to crawl multiple websites simultaneously. Start by adding your first site through the web interface or command line, then explore the powerful search capabilities across all your knowledge sources.

**Next Steps:**
1. 🌐 Add your first site using templates
2. 🤖 Start crawling and indexing content  
3. 🔍 Search across all sites from one interface
4. 📊 Monitor performance and optimize settings
5. 🚀 Scale up with additional sites and content sources 