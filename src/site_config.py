#!/usr/bin/env python3
"""
Site Configuration Management for Multi-Site RAG System

This module provides classes and utilities for managing website configurations
stored in JSON format. It handles loading, saving, validation, and manipulation
of site configurations while maintaining type safety and data integrity.
"""

import json
import os
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CrawlStrategy(Enum):
    """Different crawling strategies available"""
    BREADTH_FIRST = "breadth_first"  # Crawl all pages at depth 1, then depth 2, etc.
    DEPTH_FIRST = "depth_first"      # Follow links deep before exploring siblings
    SITEMAP = "sitemap"              # Use sitemap.xml for URL discovery
    URL_LIST = "url_list"            # Use predefined list of URLs


class ContentType(Enum):
    """Types of content that can be crawled"""
    GENERAL = "general"              # General web content
    DOCUMENTATION = "documentation"  # Technical documentation
    BLOG = "blog"                   # Blog posts and articles
    SUPPORT = "support"             # Support and FAQ content
    API = "api"                     # API documentation
    TUTORIAL = "tutorial"           # Tutorial and guide content


@dataclass
class SiteSelectors:
    """CSS selectors for extracting content from web pages"""
    title: Optional[str] = "h1, .page-title, .doc-title"
    content: Optional[str] = ".content, .documentation-content, article"
    description: Optional[str] = ".description, .summary, .lead"
    category: Optional[str] = None
    exclude: List[str] = field(default_factory=list)


@dataclass
class AuthConfig:
    """Authentication configuration for SSO-protected sites"""
    requires_sso: bool = False
    user_data_dir: Optional[str] = None
    login_url: Optional[str] = None
    auth_test_url: Optional[str] = None
    session_timeout_hours: int = 24
    auth_type: str = "sso"  # sso, basic, oauth, etc.


@dataclass
class CrawlLimits:
    """Limits and constraints for crawling operations"""
    max_articles: int = 10000
    max_depth: int = 4
    delay_seconds: float = 0.5
    timeout_seconds: int = 30
    max_file_size_mb: int = 1000
    respect_robots_txt: bool = True
    follow_redirects: bool = True
    
    # Parallelization settings
    max_concurrent_requests: int = 6
    max_concurrent_sites: int = 3
    enable_parallel_processing: bool = True
    batch_size: int = 10
    thread_pool_size: int = 6


@dataclass
class SiteConfig:
    """Configuration for a single website to be crawled"""
    # Basic identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    base_url: str = ""
    start_urls: List[str] = field(default_factory=list)
    
    # Crawling configuration
    crawl_strategy: CrawlStrategy = CrawlStrategy.BREADTH_FIRST
    content_type: ContentType = ContentType.GENERAL
    limits: CrawlLimits = field(default_factory=CrawlLimits)
    
    # Authentication configuration
    auth_config: AuthConfig = field(default_factory=AuthConfig)
    
    # Content extraction
    selectors: SiteSelectors = field(default_factory=SiteSelectors)
    
    # URL filtering
    allowed_patterns: List[str] = field(default_factory=list)  # Regex patterns
    blocked_patterns: List[str] = field(default_factory=list)  # Regex patterns
    
    # Categories and classification
    categories: List[str] = field(default_factory=list)
    auto_categorize: bool = True
    
    # Processing options
    chunk_size: int = 1000
    chunk_overlap: int = 200
    language: str = "en"
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_crawled: Optional[datetime] = None
    is_active: bool = True
    
    # Statistics
    total_articles: int = 0
    total_chunks: int = 0
    last_crawl_status: str = "not_started"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        
        # Convert datetime objects to ISO strings
        for field_name in ['created_at', 'updated_at', 'last_crawled']:
            if data[field_name] and isinstance(data[field_name], datetime):
                data[field_name] = data[field_name].isoformat()
            elif data[field_name] is None:
                data[field_name] = None
        
        # Convert enums to values
        data['crawl_strategy'] = self.crawl_strategy.value
        data['content_type'] = self.content_type.value
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SiteConfig':
        """Create SiteConfig from dictionary (loaded from JSON)"""
        # Handle datetime fields
        for field_name in ['created_at', 'updated_at', 'last_crawled']:
            if data.get(field_name):
                if isinstance(data[field_name], str):
                    data[field_name] = datetime.fromisoformat(data[field_name])
            else:
                data[field_name] = None
        
        # Handle enum fields
        if 'crawl_strategy' in data:
            data['crawl_strategy'] = CrawlStrategy(data['crawl_strategy'])
        if 'content_type' in data:
            data['content_type'] = ContentType(data['content_type'])
        
        # Handle nested objects
        if 'limits' in data and isinstance(data['limits'], dict):
            data['limits'] = CrawlLimits(**data['limits'])
        
        if 'selectors' in data and isinstance(data['selectors'], dict):
            data['selectors'] = SiteSelectors(**data['selectors'])
        
        if 'auth_config' in data and isinstance(data['auth_config'], dict):
            data['auth_config'] = AuthConfig(**data['auth_config'])
        
        return cls(**data)
    
    def update(self, **kwargs):
        """Update site configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.now()


class SiteConfigManager:
    """Manages loading, saving, and manipulating site configurations"""
    
    def __init__(self, config_file: str = "data/sites_config.json"):
        self.config_file = config_file
        self.sites: Dict[str, SiteConfig] = {}
        self._ensure_config_file()
        self.load_configs()
    
    def _ensure_config_file(self):
        """Ensure the configuration file and directory exist"""
        config_path = Path(self.config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not config_path.exists():
            # Create empty configuration file
            empty_config = {
                "sites": [],
                "updated_at": datetime.now().isoformat()
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(empty_config, f, indent=2, ensure_ascii=False)
            logger.info(f"Created new configuration file: {self.config_file}")
    
    def load_configs(self):
        """Load site configurations from JSON file"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.sites = {}
            for site_data in data.get('sites', []):
                site_config = SiteConfig.from_dict(site_data)
                self.sites[site_config.id] = site_config
            
            logger.info(f"Loaded {len(self.sites)} site configurations")
            
        except Exception as e:
            logger.error(f"Error loading site configurations: {str(e)}")
            self.sites = {}
    
    def save_configs(self):
        """Save all site configurations to JSON file"""
        try:
            # Create backup
            self._create_backup()
            
            data = {
                "sites": [site.to_dict() for site in self.sites.values()],
                "updated_at": datetime.now().isoformat()
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(self.sites)} site configurations")
            
        except Exception as e:
            logger.error(f"Error saving site configurations: {str(e)}")
            raise
    
    def _create_backup(self):
        """Create a backup of the current configuration file"""
        if os.path.exists(self.config_file):
            backup_dir = Path("backups")
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"sites_config_backup_{timestamp}.json"
            
            import shutil
            shutil.copy2(self.config_file, backup_file)
    
    def add_site(self, site_config: SiteConfig) -> str:
        """Add a new site configuration"""
        if not site_config.id:
            site_config.id = str(uuid.uuid4())
        
        site_config.created_at = datetime.now()
        site_config.updated_at = datetime.now()
        
        self.sites[site_config.id] = site_config
        self.save_configs()
        
        logger.info(f"Added new site: {site_config.name} ({site_config.id})")
        return site_config.id
    
    def update_site(self, site_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing site configuration"""
        if site_id not in self.sites:
            logger.warning(f"Site not found: {site_id}")
            return False
        
        site = self.sites[site_id]
        
        # Handle special cases for nested objects
        if 'limits' in updates and isinstance(updates['limits'], dict):
            site.limits = CrawlLimits(**updates['limits'])
            del updates['limits']
        
        if 'selectors' in updates and isinstance(updates['selectors'], dict):
            site.selectors = SiteSelectors(**updates['selectors'])
            del updates['selectors']
        
        if 'auth_config' in updates and isinstance(updates['auth_config'], dict):
            site.auth_config = AuthConfig(**updates['auth_config'])
            del updates['auth_config']
        
        # Handle enum conversions
        if 'crawl_strategy' in updates:
            if isinstance(updates['crawl_strategy'], str):
                updates['crawl_strategy'] = CrawlStrategy(updates['crawl_strategy'])
        
        if 'content_type' in updates:
            if isinstance(updates['content_type'], str):
                updates['content_type'] = ContentType(updates['content_type'])
        
        # Apply updates
        site.update(**updates)
        self.save_configs()
        
        logger.info(f"Updated site configuration: {site.name}")
        return True
    
    def delete_site(self, site_id: str) -> bool:
        """Delete a site configuration"""
        if site_id not in self.sites:
            logger.warning(f"Site not found: {site_id}")
            return False
        
        site_name = self.sites[site_id].name
        del self.sites[site_id]
        self.save_configs()
        
        logger.info(f"Deleted site: {site_name} ({site_id})")
        return True
    
    def get_site(self, site_id: str) -> Optional[SiteConfig]:
        """Get a specific site configuration"""
        return self.sites.get(site_id)
    
    def get_all_sites(self) -> List[SiteConfig]:
        """Get all site configurations"""
        return list(self.sites.values())
    
    def get_active_sites(self) -> List[SiteConfig]:
        """Get only active site configurations"""
        return [site for site in self.sites.values() if site.is_active]
    
    def get_sites_by_type(self, content_type: ContentType) -> List[SiteConfig]:
        """Get sites filtered by content type"""
        return [site for site in self.sites.values() if site.content_type == content_type]
    
    def backup_configs(self, backup_path: str):
        """Create a backup of configurations to a specific path"""
        backup_file = Path(backup_path) / "sites_config.json"
        backup_file.parent.mkdir(parents=True, exist_ok=True)
        
        import shutil
        shutil.copy2(self.config_file, backup_file)
        logger.info(f"Backed up configurations to: {backup_file}")


def create_site_from_template(template_name: str, 
                             name: str,
                             description: str,
                             base_url: str,
                             start_urls: List[str]) -> SiteConfig:
    """Create a new site configuration from a predefined template"""
    
    templates = {
        "documentation": {
            "content_type": ContentType.DOCUMENTATION,
            "crawl_strategy": CrawlStrategy.BREADTH_FIRST,
            "limits": CrawlLimits(
                max_articles=10000,
                max_depth=4,
                delay_seconds=1.0,
                respect_robots_txt=True,
                max_concurrent_requests=6,
                max_concurrent_sites=2,
                enable_parallel_processing=True,
                batch_size=15,
                thread_pool_size=4
            ),
            "selectors": SiteSelectors(
                title="h1, .page-title, .doc-title",
                content=".content, .documentation-content, article, .doc-content",
                description=".description, .summary, .lead",
                exclude=["nav", "header", "footer", ".sidebar"]
            ),
            "categories": ["Getting Started", "API", "Tutorials", "FAQ"],
            "blocked_patterns": [r".*\.(pdf|jpg|jpeg|png|gif|svg|css|js|ico)$"]
        },
        
        "blog": {
            "content_type": ContentType.BLOG,
            "crawl_strategy": CrawlStrategy.BREADTH_FIRST,
            "limits": CrawlLimits(
                max_articles=200,
                max_depth=3,
                delay_seconds=1.5,
                max_concurrent_requests=3,
                max_concurrent_sites=2,
                enable_parallel_processing=True,
                batch_size=8,
                thread_pool_size=3
            ),
            "selectors": SiteSelectors(
                title="h1, .post-title, .article-title",
                content=".post-content, .article-content, article",
                description=".excerpt, .summary, .lead",
                category=".category, .tag",
                exclude=["nav", "sidebar", "footer", ".comments"]
            ),
            "categories": ["News", "Updates", "How-to", "Case Studies"],
            "blocked_patterns": [r".*\.(pdf|jpg|jpeg|png|gif|css|js)$"]
        },
        
        "support": {
            "content_type": ContentType.SUPPORT,
            "crawl_strategy": CrawlStrategy.BREADTH_FIRST,
            "limits": CrawlLimits(
                max_articles=300,
                max_depth=3,
                delay_seconds=1.0,
                max_concurrent_requests=4,
                max_concurrent_sites=2,
                enable_parallel_processing=True,
                batch_size=12,
                thread_pool_size=3
            ),
            "selectors": SiteSelectors(
                title="h1, .question, .faq-title",
                content=".answer, .solution, .content",
                category=".category, .topic",
                exclude=["nav", "sidebar", "footer"]
            ),
            "categories": ["FAQ", "Troubleshooting", "How-to", "Support"],
            "blocked_patterns": [r".*\.(pdf|jpg|png|gif|css|js)$"]
        }
    }
    
    if template_name not in templates:
        raise ValueError(f"Unknown template: {template_name}. Available: {list(templates.keys())}")
    
    template = templates[template_name]
    
    # Create site configuration from template
    site_config = SiteConfig(
        name=name,
        description=description,
        base_url=base_url,
        start_urls=start_urls,
        content_type=template["content_type"],
        crawl_strategy=template["crawl_strategy"],
        limits=template["limits"],
        selectors=template["selectors"],
        categories=template["categories"],
        blocked_patterns=template.get("blocked_patterns", []),
        auto_categorize=True,
        is_active=True
    )
    
    logger.info(f"Created site from template '{template_name}': {name}")
    return site_config


def get_available_templates() -> Dict[str, Dict[str, Any]]:
    """Get information about available site templates"""
    return {
        "documentation": {
            "name": "Documentation Site",
            "description": "For technical documentation, guides, and API docs",
            "content_type": "documentation",
            "typical_selectors": {
                "title": "h1, .page-title, .doc-title",
                "content": ".content, .documentation-content, article"
            }
        },
        "blog": {
            "name": "Blog Site",
            "description": "For blog posts, news, and article content",
            "content_type": "blog",
            "typical_selectors": {
                "title": "h1, .post-title, .article-title",
                "content": ".post-content, .article-content, article"
            }
        },
        "support": {
            "name": "Support Site",
            "description": "For FAQ, support articles, and help content",
            "content_type": "support",
            "typical_selectors": {
                "title": "h1, .question, .faq-title",
                "content": ".answer, .solution, .content"
            }
        }
    } 