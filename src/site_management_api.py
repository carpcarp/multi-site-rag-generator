#!/usr/bin/env python3
"""
Site Management API for Multi-Site RAG System
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

from site_config import (
    SiteConfig, SiteConfigManager, CrawlStrategy, ContentType, 
    SiteSelectors, CrawlLimits, create_site_from_template
)
from generic_crawler import GenericWebCrawler, MultiSiteCrawler
from crawl_progress import CrawlProgressManager, JobProgress, CrawlJobStatus, SiteCrawlStatus

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class SiteSelectorsRequest(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    exclude: List[str] = Field(default_factory=list)

class CrawlLimitsRequest(BaseModel):
    max_articles: int = Field(100, ge=1, le=10000)
    max_depth: int = Field(3, ge=1, le=10)
    delay_seconds: float = Field(1.0, ge=0.1, le=10.0)
    timeout_seconds: int = Field(30, ge=5, le=300)
    max_file_size_mb: int = Field(10, ge=1, le=100)
    respect_robots_txt: bool = True
    follow_redirects: bool = True

class SiteConfigRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field("", max_length=1000)
    base_url: str = Field(..., pattern=r'^https?://.+')
    start_urls: List[str] = Field(..., min_items=1, max_items=50)
    
    crawl_strategy: CrawlStrategy = CrawlStrategy.BREADTH_FIRST
    content_type: ContentType = ContentType.GENERAL
    limits: CrawlLimitsRequest = Field(default_factory=CrawlLimitsRequest)
    selectors: SiteSelectorsRequest = Field(default_factory=SiteSelectorsRequest)
    
    allowed_patterns: List[str] = Field(default_factory=list, max_items=20)
    blocked_patterns: List[str] = Field(default_factory=list, max_items=20)
    categories: List[str] = Field(default_factory=list, max_items=50)
    auto_categorize: bool = True
    
    chunk_size: int = Field(1000, ge=100, le=4000)
    chunk_overlap: int = Field(200, ge=0, le=1000)
    language: str = Field("en", min_length=2, max_length=5)
    is_active: bool = True

    @validator('start_urls')
    def validate_start_urls(cls, v):
        for url in v:
            if not url.startswith(('http://', 'https://')):
                raise ValueError(f"Invalid URL: {url}")
        return v

    @validator('chunk_overlap')
    def validate_chunk_overlap(cls, v, values):
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError("Chunk overlap must be less than chunk size")
        return v

class CrawlLimitsResponse(BaseModel):
    max_articles: int
    max_depth: int
    delay_seconds: float
    timeout_seconds: int
    max_file_size_mb: int
    respect_robots_txt: bool
    follow_redirects: bool

class SiteSelectorsResponse(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    exclude: List[str] = Field(default_factory=list)

class SiteConfigResponse(BaseModel):
    id: str
    name: str
    description: str
    base_url: str
    start_urls: List[str]
    crawl_strategy: str
    content_type: str
    limits: CrawlLimitsResponse
    selectors: SiteSelectorsResponse
    allowed_patterns: List[str]
    blocked_patterns: List[str]
    categories: List[str]
    auto_categorize: bool
    chunk_size: int
    chunk_overlap: int
    language: str
    is_active: bool
    total_articles: int
    total_chunks: int
    last_crawl_status: str
    created_at: str
    updated_at: str
    last_crawled: Optional[str] = None

class CrawlJobRequest(BaseModel):
    site_ids: List[str] = Field(..., min_items=1, max_items=10)
    force_recrawl: bool = False

class CrawlJobResponse(BaseModel):
    job_id: str
    status: str
    site_ids: List[str]
    started_at: str
    estimated_duration: Optional[int] = None

class CrawlStatusResponse(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    site_results: Dict[str, Dict[str, Any]]
    started_at: str
    completed_at: Optional[str] = None
    total_articles: int = 0
    errors: List[str] = Field(default_factory=list)

class TemplateRequest(BaseModel):
    template_name: str = Field(..., pattern=r'^(documentation|blog|support)$')
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field("", max_length=1000)
    base_url: str = Field(..., pattern=r'^https?://.+')
    start_urls: List[str] = Field(..., min_items=1, max_items=50)

# Enhanced crawl job manager with progress tracking
class CrawlJobManager:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.active_jobs: Dict[str, asyncio.Task] = {}
        self.progress_manager = CrawlProgressManager()
    
    def create_job(self, site_ids: List[str], site_configs: Dict[str, Any]) -> str:
        job_id = f"crawl_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.jobs)}"
        
        # Create progress tracking
        progress = self.progress_manager.create_job(job_id, site_ids, site_configs)
        
        # Maintain backward compatibility with simple job dict
        self.jobs[job_id] = {
            'status': 'pending',
            'site_ids': site_ids,
            'started_at': datetime.now().isoformat(),
            'site_results': {},
            'errors': [],
            'progress': progress
        }
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)
    
    def get_job_progress(self, job_id: str) -> Optional[JobProgress]:
        """Get detailed progress information"""
        return self.progress_manager.get_job_progress(job_id)
    
    def update_job(self, job_id: str, updates: Dict[str, Any]):
        if job_id in self.jobs:
            self.jobs[job_id].update(updates)
    
    def set_job_running(self, job_id: str, task: asyncio.Task):
        self.active_jobs[job_id] = task
        self.update_job(job_id, {'status': 'running'})
        
        # Update progress status
        progress = self.progress_manager.get_job_progress(job_id)
        if progress:
            progress.status = CrawlJobStatus.RUNNING
            self.progress_manager.save_progress(job_id)
    
    def complete_job(self, job_id: str, results: Dict[str, Any]):
        if job_id in self.active_jobs:
            del self.active_jobs[job_id]
        
        self.update_job(job_id, {
            'status': 'completed',
            'completed_at': datetime.now().isoformat(),
            'site_results': results,
            'total_articles': sum(len(articles) for articles in results.values())
        })
        
        # Complete progress tracking
        self.progress_manager.complete_job(job_id)
    
    def fail_job(self, job_id: str, error: str):
        if job_id in self.active_jobs:
            del self.active_jobs[job_id]
        
        self.update_job(job_id, {
            'status': 'failed',
            'completed_at': datetime.now().isoformat(),
        })
        
        if job_id in self.jobs:
            self.jobs[job_id]['errors'].append(error)
        
        # Fail progress tracking
        self.progress_manager.fail_job(job_id, error)
    
    def get_recoverable_jobs(self) -> List[str]:
        """Get jobs that can be recovered from interruption"""
        return self.progress_manager.get_recoverable_jobs()

# Global instances
job_manager = CrawlJobManager()
config_manager = SiteConfigManager()

def create_site_management_api() -> FastAPI:
    """Create the site management API"""
    
    app = FastAPI(
        title="Multi-Site RAG - Site Management API",
        description="API for managing website configurations and crawling jobs",
        version="1.0.0"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Site configuration endpoints
    @app.get("/sites", response_model=List[SiteConfigResponse])
    async def list_sites(active_only: bool = Query(False)):
        """List all site configurations"""
        try:
            if active_only:
                sites = config_manager.get_active_sites()
            else:
                sites = config_manager.get_all_sites()
            
            return [
                SiteConfigResponse(
                    id=site.id,
                    name=site.name,
                    description=site.description,
                    base_url=site.base_url,
                    start_urls=site.start_urls,
                    crawl_strategy=site.crawl_strategy.value,
                    content_type=site.content_type.value,
                    limits=CrawlLimitsResponse(
                        max_articles=site.limits.max_articles,
                        max_depth=site.limits.max_depth,
                        delay_seconds=site.limits.delay_seconds,
                        timeout_seconds=site.limits.timeout_seconds,
                        max_file_size_mb=site.limits.max_file_size_mb,
                        respect_robots_txt=site.limits.respect_robots_txt,
                        follow_redirects=site.limits.follow_redirects
                    ),
                    selectors=SiteSelectorsResponse(
                        title=site.selectors.title,
                        content=site.selectors.content,
                        description=site.selectors.description,
                        category=site.selectors.category,
                        exclude=site.selectors.exclude
                    ),
                    allowed_patterns=site.allowed_patterns,
                    blocked_patterns=site.blocked_patterns,
                    categories=site.categories,
                    auto_categorize=site.auto_categorize,
                    chunk_size=site.chunk_size,
                    chunk_overlap=site.chunk_overlap,
                    language=site.language,
                    is_active=site.is_active,
                    total_articles=site.total_articles,
                    total_chunks=site.total_chunks,
                    last_crawl_status=site.last_crawl_status,
                    created_at=site.created_at.isoformat(),
                    updated_at=site.updated_at.isoformat(),
                    last_crawled=site.last_crawled.isoformat() if site.last_crawled else None
                )
                for site in sites
            ]
        except Exception as e:
            logger.error(f"Error listing sites: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/sites/{site_id}", response_model=SiteConfigResponse)
    async def get_site(site_id: str):
        """Get a specific site configuration"""
        site = config_manager.get_site(site_id)
        if not site:
            raise HTTPException(status_code=404, detail="Site not found")
        
        return SiteConfigResponse(
            id=site.id,
            name=site.name,
            description=site.description,
            base_url=site.base_url,
            start_urls=site.start_urls,
            crawl_strategy=site.crawl_strategy.value,
            content_type=site.content_type.value,
            limits=CrawlLimitsResponse(
                max_articles=site.limits.max_articles,
                max_depth=site.limits.max_depth,
                delay_seconds=site.limits.delay_seconds,
                timeout_seconds=site.limits.timeout_seconds,
                max_file_size_mb=site.limits.max_file_size_mb,
                respect_robots_txt=site.limits.respect_robots_txt,
                follow_redirects=site.limits.follow_redirects
            ),
            selectors=SiteSelectorsResponse(
                title=site.selectors.title,
                content=site.selectors.content,
                description=site.selectors.description,
                category=site.selectors.category,
                exclude=site.selectors.exclude
            ),
            allowed_patterns=site.allowed_patterns,
            blocked_patterns=site.blocked_patterns,
            categories=site.categories,
            auto_categorize=site.auto_categorize,
            chunk_size=site.chunk_size,
            chunk_overlap=site.chunk_overlap,
            language=site.language,
            is_active=site.is_active,
            total_articles=site.total_articles,
            total_chunks=site.total_chunks,
            last_crawl_status=site.last_crawl_status,
            created_at=site.created_at.isoformat(),
            updated_at=site.updated_at.isoformat(),
            last_crawled=site.last_crawled.isoformat() if site.last_crawled else None
        )
    
    @app.post("/sites", response_model=SiteConfigResponse)
    async def create_site(site_request: SiteConfigRequest):
        """Create a new site configuration"""
        try:
            # Convert request to SiteConfig
            site_config = SiteConfig(
                name=site_request.name,
                description=site_request.description,
                base_url=site_request.base_url,
                start_urls=site_request.start_urls,
                crawl_strategy=site_request.crawl_strategy,
                content_type=site_request.content_type,
                limits=CrawlLimits(**site_request.limits.dict()),
                selectors=SiteSelectors(**site_request.selectors.dict()),
                allowed_patterns=site_request.allowed_patterns,
                blocked_patterns=site_request.blocked_patterns,
                categories=site_request.categories,
                auto_categorize=site_request.auto_categorize,
                chunk_size=site_request.chunk_size,
                chunk_overlap=site_request.chunk_overlap,
                language=site_request.language,
                is_active=site_request.is_active
            )
            
            site_id = config_manager.add_site(site_config)
            site = config_manager.get_site(site_id)
            
            return SiteConfigResponse(
                id=site.id,
                name=site.name,
                description=site.description,
                base_url=site.base_url,
                start_urls=site.start_urls,
                crawl_strategy=site.crawl_strategy.value,
                content_type=site.content_type.value,
                limits=CrawlLimitsResponse(
                    max_articles=site.limits.max_articles,
                    max_depth=site.limits.max_depth,
                    delay_seconds=site.limits.delay_seconds,
                    timeout_seconds=site.limits.timeout_seconds,
                    max_file_size_mb=site.limits.max_file_size_mb,
                    respect_robots_txt=site.limits.respect_robots_txt,
                    follow_redirects=site.limits.follow_redirects
                ),
                selectors=SiteSelectorsResponse(
                    title=site.selectors.title,
                    content=site.selectors.content,
                    description=site.selectors.description,
                    category=site.selectors.category,
                    exclude=site.selectors.exclude
                ),
                allowed_patterns=site.allowed_patterns,
                blocked_patterns=site.blocked_patterns,
                categories=site.categories,
                auto_categorize=site.auto_categorize,
                chunk_size=site.chunk_size,
                chunk_overlap=site.chunk_overlap,
                language=site.language,
                is_active=site.is_active,
                total_articles=site.total_articles,
                total_chunks=site.total_chunks,
                last_crawl_status=site.last_crawl_status,
                created_at=site.created_at.isoformat(),
                updated_at=site.updated_at.isoformat(),
                last_crawled=site.last_crawled.isoformat() if site.last_crawled else None
            )
            
        except Exception as e:
            logger.error(f"Error creating site: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.put("/sites/{site_id}", response_model=SiteConfigResponse)
    async def update_site(site_id: str, site_request: SiteConfigRequest):
        """Update an existing site configuration"""
        if not config_manager.get_site(site_id):
            raise HTTPException(status_code=404, detail="Site not found")
        
        try:
            updates = {
                'name': site_request.name,
                'description': site_request.description,
                'base_url': site_request.base_url,
                'start_urls': site_request.start_urls,
                'crawl_strategy': site_request.crawl_strategy,
                'content_type': site_request.content_type,
                'limits': CrawlLimits(**site_request.limits.dict()),
                'selectors': SiteSelectors(**site_request.selectors.dict()),
                'allowed_patterns': site_request.allowed_patterns,
                'blocked_patterns': site_request.blocked_patterns,
                'categories': site_request.categories,
                'auto_categorize': site_request.auto_categorize,
                'chunk_size': site_request.chunk_size,
                'chunk_overlap': site_request.chunk_overlap,
                'language': site_request.language,
                'is_active': site_request.is_active
            }
            
            config_manager.update_site(site_id, updates)
            site = config_manager.get_site(site_id)
            
            return SiteConfigResponse(
                id=site.id,
                name=site.name,
                description=site.description,
                base_url=site.base_url,
                start_urls=site.start_urls,
                crawl_strategy=site.crawl_strategy.value,
                content_type=site.content_type.value,
                limits=CrawlLimitsResponse(
                    max_articles=site.limits.max_articles,
                    max_depth=site.limits.max_depth,
                    delay_seconds=site.limits.delay_seconds,
                    timeout_seconds=site.limits.timeout_seconds,
                    max_file_size_mb=site.limits.max_file_size_mb,
                    respect_robots_txt=site.limits.respect_robots_txt,
                    follow_redirects=site.limits.follow_redirects
                ),
                selectors=SiteSelectorsResponse(
                    title=site.selectors.title,
                    content=site.selectors.content,
                    description=site.selectors.description,
                    category=site.selectors.category,
                    exclude=site.selectors.exclude
                ),
                allowed_patterns=site.allowed_patterns,
                blocked_patterns=site.blocked_patterns,
                categories=site.categories,
                auto_categorize=site.auto_categorize,
                chunk_size=site.chunk_size,
                chunk_overlap=site.chunk_overlap,
                language=site.language,
                is_active=site.is_active,
                total_articles=site.total_articles,
                total_chunks=site.total_chunks,
                last_crawl_status=site.last_crawl_status,
                created_at=site.created_at.isoformat(),
                updated_at=site.updated_at.isoformat(),
                last_crawled=site.last_crawled.isoformat() if site.last_crawled else None
            )
            
        except Exception as e:
            logger.error(f"Error updating site: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.delete("/sites/{site_id}")
    async def delete_site(site_id: str):
        """Delete a site configuration"""
        if not config_manager.get_site(site_id):
            raise HTTPException(status_code=404, detail="Site not found")
        
        try:
            config_manager.delete_site(site_id)
            return {"message": "Site deleted successfully"}
        except Exception as e:
            logger.error(f"Error deleting site: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Template endpoints
    @app.post("/sites/from-template", response_model=SiteConfigResponse)
    async def create_site_from_template_endpoint(template_request: TemplateRequest):
        """Create a site from a template"""
        try:
            site_config = create_site_from_template(
                template_request.template_name,
                name=template_request.name,
                description=template_request.description,
                base_url=template_request.base_url,
                start_urls=template_request.start_urls
            )
            
            site_id = config_manager.add_site(site_config)
            site = config_manager.get_site(site_id)
            
            return SiteConfigResponse(
                id=site.id,
                name=site.name,
                description=site.description,
                base_url=site.base_url,
                start_urls=site.start_urls,
                crawl_strategy=site.crawl_strategy.value,
                content_type=site.content_type.value,
                limits=CrawlLimitsResponse(
                    max_articles=site.limits.max_articles,
                    max_depth=site.limits.max_depth,
                    delay_seconds=site.limits.delay_seconds,
                    timeout_seconds=site.limits.timeout_seconds,
                    max_file_size_mb=site.limits.max_file_size_mb,
                    respect_robots_txt=site.limits.respect_robots_txt,
                    follow_redirects=site.limits.follow_redirects
                ),
                selectors=SiteSelectorsResponse(
                    title=site.selectors.title,
                    content=site.selectors.content,
                    description=site.selectors.description,
                    category=site.selectors.category,
                    exclude=site.selectors.exclude
                ),
                allowed_patterns=site.allowed_patterns,
                blocked_patterns=site.blocked_patterns,
                categories=site.categories,
                auto_categorize=site.auto_categorize,
                chunk_size=site.chunk_size,
                chunk_overlap=site.chunk_overlap,
                language=site.language,
                is_active=site.is_active,
                total_articles=site.total_articles,
                total_chunks=site.total_chunks,
                last_crawl_status=site.last_crawl_status,
                created_at=site.created_at.isoformat(),
                updated_at=site.updated_at.isoformat(),
                last_crawled=site.last_crawled.isoformat() if site.last_crawled else None
            )
            
        except Exception as e:
            logger.error(f"Error creating site from template: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/templates")
    async def get_templates():
        """Get available site templates"""
        return {
            "templates": [
                {
                    "name": "documentation",
                    "description": "For documentation websites (GitBook, Notion, etc.)",
                    "default_selectors": {
                        "title": "h1, .page-title, .doc-title",
                        "content": ".content, .documentation-content, article, .doc-content",
                        "description": ".description, .summary, .lead"
                    },
                    "default_categories": ["Getting Started", "API", "Tutorials", "FAQ"]
                },
                {
                    "name": "blog",
                    "description": "For blog and news websites",
                    "default_selectors": {
                        "title": "h1, .post-title, .article-title",
                        "content": ".post-content, .article-content, .entry-content",
                        "category": ".category, .tag, .post-category"
                    },
                    "default_categories": ["News", "Tutorials", "Updates", "Tips"]
                },
                {
                    "name": "support",
                    "description": "For support and FAQ websites",
                    "default_selectors": {
                        "title": "h1, .question, .faq-title",
                        "content": ".answer, .faq-content, .support-content",
                        "category": ".category, .topic"
                    },
                    "default_categories": ["FAQ", "Troubleshooting", "How-to", "Known Issues"]
                }
            ]
        }
    
    # Crawling endpoints
    @app.post("/crawl/start", response_model=CrawlJobResponse)
    async def start_crawl(
        crawl_request: CrawlJobRequest,
        background_tasks: BackgroundTasks
    ):
        """Start a crawling job for specified sites"""
        try:
            # Validate site IDs and collect site configs
            site_configs = {}
            for site_id in crawl_request.site_ids:
                site = config_manager.get_site(site_id)
                if not site:
                    raise HTTPException(status_code=404, detail=f"Site {site_id} not found")
                site_configs[site_id] = {
                    'name': site.name,
                    'limits': {
                        'max_articles': site.limits.max_articles,
                        'max_depth': site.limits.max_depth
                    }
                }
            
            # Create crawl job with progress tracking
            job_id = job_manager.create_job(crawl_request.site_ids, site_configs)
            
            # Start background crawling task
            background_tasks.add_task(
                run_crawl_job, 
                job_id, 
                crawl_request.site_ids,
                crawl_request.force_recrawl
            )
            
            return CrawlJobResponse(
                job_id=job_id,
                status="pending",
                site_ids=crawl_request.site_ids,
                started_at=datetime.now().isoformat(),
                estimated_duration=len(crawl_request.site_ids) * 60  # Rough estimate
            )
            
        except Exception as e:
            logger.error(f"Error starting crawl: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/crawl/{job_id}/status", response_model=CrawlStatusResponse)
    async def get_crawl_status(job_id: str):
        """Get the status of a crawl job"""
        job = job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Crawl job not found")
        
        return CrawlStatusResponse(
            job_id=job_id,
            status=job['status'],
            site_results=job.get('site_results', {}),
            started_at=job['started_at'],
            completed_at=job.get('completed_at'),
            total_articles=job.get('total_articles', 0),
            errors=job.get('errors', [])
        )
    
    @app.get("/crawl/jobs")
    async def list_crawl_jobs():
        """List all crawl jobs"""
        return {
            "jobs": [
                {
                    "job_id": job_id,
                    "status": job['status'],
                    "site_ids": job['site_ids'],
                    "started_at": job['started_at'],
                    "completed_at": job.get('completed_at'),
                    "total_articles": job.get('total_articles', 0)
                }
                for job_id, job in job_manager.jobs.items()
            ]
        }
    
    @app.get("/crawl/{job_id}/progress")
    async def get_detailed_progress(job_id: str):
        """Get detailed progress information for a crawl job"""
        progress = job_manager.get_job_progress(job_id)
        if not progress:
            raise HTTPException(status_code=404, detail="Crawl job not found")
        
        # Convert to serializable format
        site_progresses = {}
        for site_id, site_progress in progress.site_progresses.items():
            site_progresses[site_id] = {
                "site_id": site_progress.site_id,
                "site_name": site_progress.site_name,
                "status": site_progress.status.value,
                "progress_percentage": site_progress.progress_percentage,
                "current_depth": site_progress.current_depth,
                "max_depth": site_progress.max_depth,
                "discovered_urls": site_progress.total_urls_discovered,
                "crawled_urls": site_progress.total_urls_crawled,
                "failed_urls": site_progress.total_urls_failed,
                "articles_found": site_progress.articles_found,
                "articles_processed": site_progress.articles_processed,
                "chunks_created": site_progress.chunks_created,
                "estimated_time_remaining": site_progress.estimated_time_remaining,
                "start_time": site_progress.start_time.isoformat() if site_progress.start_time else None,
                "completion_time": site_progress.completion_time.isoformat() if site_progress.completion_time else None,
                "errors": site_progress.errors
            }
        
        return {
            "job_id": progress.job_id,
            "status": progress.status.value,
            "overall_progress_percentage": progress.overall_progress_percentage,
            "sites_completed": progress.sites_completed,
            "sites_failed": progress.sites_failed,
            "total_articles": progress.total_articles,
            "total_chunks": progress.total_chunks,
            "total_errors": progress.total_errors,
            "estimated_time_remaining": progress.estimated_time_remaining,
            "start_time": progress.start_time.isoformat(),
            "last_update": progress.last_update.isoformat(),
            "completion_time": progress.completion_time.isoformat() if progress.completion_time else None,
            "site_progresses": site_progresses
        }
    
    @app.get("/crawl/recoverable")
    async def get_recoverable_jobs():
        """Get jobs that can be recovered from interruption"""
        recoverable_jobs = job_manager.get_recoverable_jobs()
        return {
            "recoverable_jobs": recoverable_jobs,
            "count": len(recoverable_jobs)
        }
    
    @app.post("/crawl/{job_id}/recover")
    async def recover_crawl_job(
        job_id: str,
        background_tasks: BackgroundTasks
    ):
        """Recover and resume an interrupted crawl job"""
        progress = job_manager.get_job_progress(job_id)
        if not progress:
            raise HTTPException(status_code=404, detail="Crawl job not found")
        
        if progress.status not in [CrawlJobStatus.RUNNING, CrawlJobStatus.PAUSED]:
            raise HTTPException(status_code=400, detail="Job cannot be recovered - not in recoverable state")
        
        # Extract site IDs that haven't completed
        incomplete_sites = [
            site_id for site_id, site_progress in progress.site_progresses.items()
            if site_progress.status not in [SiteCrawlStatus.COMPLETED, SiteCrawlStatus.FAILED, SiteCrawlStatus.SKIPPED]
        ]
        
        if not incomplete_sites:
            raise HTTPException(status_code=400, detail="No incomplete sites to recover")
        
        # Resume the job
        background_tasks.add_task(
            run_crawl_job,
            job_id,
            incomplete_sites,
            force_recrawl=False
        )
        
        return {
            "message": f"Recovery started for job {job_id}",
            "incomplete_sites": incomplete_sites,
            "sites_to_recover": len(incomplete_sites)
        }
    
    # Health check
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "total_sites": len(config_manager.get_all_sites()),
            "active_sites": len(config_manager.get_active_sites()),
            "active_crawl_jobs": len(job_manager.active_jobs),
            "timestamp": datetime.now().isoformat()
        }
    
    return app

async def run_crawl_job(job_id: str, site_ids: List[str], force_recrawl: bool = False):
    """Background task to run a crawl job with progress tracking"""
    try:
        logger.info(f"Starting crawl job {job_id} for sites: {site_ids}")
        
        # Create multi-site crawler with progress tracking
        progress_manager = job_manager.progress_manager
        multi_crawler = MultiSiteCrawler(config_manager, progress_manager)
        
        # Start crawling
        results = {}
        for site_id in site_ids:
            try:
                site_config = config_manager.get_site(site_id)
                if not site_config:
                    continue
                
                # Skip if recently crawled successfully and not forcing recrawl
                if (not force_recrawl and 
                    site_config.last_crawled and 
                    site_config.last_crawl_status == "completed" and
                    site_config.total_articles > 0):  # Only skip if previous crawl found content
                    
                    time_since_crawl = datetime.now() - site_config.last_crawled
                    if time_since_crawl.total_seconds() < 3600:  # 1 hour
                        logger.info(f"Skipping {site_config.name} - recently crawled successfully")
                        progress_manager.update_site_progress(
                            job_id, site_id, 
                            status=SiteCrawlStatus.SKIPPED,
                            completion_time=datetime.now()
                        )
                        continue
                
                # Crawl the site with progress tracking
                articles = await multi_crawler.crawl_site_by_id(site_id, job_id)
                results[site_id] = articles
                
                logger.info(f"Crawled {len(articles)} articles from {site_config.name}")
                
            except Exception as e:
                logger.error(f"Error crawling site {site_id}: {str(e)}")
                results[site_id] = []
                
                # Track error in progress system
                progress_manager.fail_site(job_id, site_id, str(e))
                
                # Maintain backward compatibility
                if job_id in job_manager.jobs:
                    job_manager.jobs[job_id]['errors'].append(f"Site {site_id}: {str(e)}")
        
        # Complete the job
        job_manager.complete_job(job_id, results)
        logger.info(f"Crawl job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error in crawl job {job_id}: {str(e)}")
        job_manager.fail_job(job_id, str(e))

def main():
    """Run the site management API server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Site Management API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8001, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run the API
    app = create_site_management_api()
    
    logger.info(f"Starting Site Management API on {args.host}:{args.port}")
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main() 