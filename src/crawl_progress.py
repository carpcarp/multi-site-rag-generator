#!/usr/bin/env python3
"""
Crawl Progress Tracking and Recovery System

This module provides incremental progress tracking for crawl jobs,
allowing for recovery and resumption of interrupted crawls.
"""

import json
import os
import time
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class CrawlJobStatus(Enum):
    """Status of a crawl job"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SiteCrawlStatus(Enum):
    """Status of an individual site crawl within a job"""
    PENDING = "pending"
    DISCOVERING = "discovering"  # Finding URLs to crawl
    CRAWLING = "crawling"        # Actively crawling pages
    PROCESSING = "processing"    # Processing crawled content
    PAUSED = "paused"           # Crawl paused by user
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class CrawlProgress:
    """Progress tracking for a single site crawl"""
    site_id: str
    site_name: str
    status: SiteCrawlStatus
    
    # URL tracking
    discovered_urls: Set[str]
    crawled_urls: Set[str]
    failed_urls: Set[str]
    remaining_urls: Set[str]
    
    # Content tracking
    articles_found: int = 0
    articles_processed: int = 0
    chunks_created: int = 0
    
    # Progress metrics
    current_depth: int = 0
    max_depth: int = 3
    start_time: Optional[datetime] = None
    last_update: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    
    # Error tracking
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.start_time is None:
            self.start_time = datetime.now()
        self.last_update = datetime.now()
    
    @property
    def total_urls_discovered(self) -> int:
        return len(self.discovered_urls)
    
    @property
    def total_urls_crawled(self) -> int:
        return len(self.crawled_urls)
    
    @property
    def total_urls_failed(self) -> int:
        return len(self.failed_urls)
    
    @property
    def progress_percentage(self) -> float:
        """Calculate crawl progress as percentage"""
        if not self.discovered_urls:
            return 0.0
        return (len(self.crawled_urls) / len(self.discovered_urls)) * 100
    
    @property
    def estimated_time_remaining(self) -> Optional[float]:
        """Estimate time remaining based on current progress"""
        if not self.start_time or self.total_urls_crawled == 0:
            return None
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.total_urls_crawled / elapsed  # URLs per second
        
        if rate > 0 and self.remaining_urls:
            return len(self.remaining_urls) / rate
        return None


@dataclass
class JobProgress:
    """Progress tracking for an entire crawl job"""
    job_id: str
    status: CrawlJobStatus
    site_progresses: Dict[str, CrawlProgress]
    
    start_time: datetime
    last_update: datetime
    completion_time: Optional[datetime] = None
    
    # Job-level metrics
    total_articles: int = 0
    total_chunks: int = 0
    total_errors: int = 0
    
    # Recovery information
    checkpoint_interval: int = 30  # seconds
    last_checkpoint: Optional[datetime] = None
    
    def __post_init__(self):
        self.last_update = datetime.now()
    
    @property
    def overall_progress_percentage(self) -> float:
        """Calculate overall job progress"""
        if not self.site_progresses:
            return 0.0
        
        total_progress = sum(progress.progress_percentage for progress in self.site_progresses.values())
        return total_progress / len(self.site_progresses)
    
    @property
    def sites_completed(self) -> int:
        return sum(1 for p in self.site_progresses.values() if p.status == SiteCrawlStatus.COMPLETED)
    
    @property
    def sites_failed(self) -> int:
        return sum(1 for p in self.site_progresses.values() if p.status == SiteCrawlStatus.FAILED)
    
    @property
    def estimated_time_remaining(self) -> Optional[float]:
        """Estimate total time remaining for job"""
        site_estimates = [p.estimated_time_remaining for p in self.site_progresses.values() 
                         if p.estimated_time_remaining is not None]
        
        if site_estimates:
            # Return the maximum estimate (assuming sites run sequentially)
            return max(site_estimates)
        return None


class CrawlProgressManager:
    """Manages crawl progress tracking and persistence"""
    
    def __init__(self, progress_dir: str = "data/progress"):
        self.progress_dir = Path(progress_dir)
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache of active jobs
        self.active_jobs: Dict[str, JobProgress] = {}
        
        # Load any existing progress files
        self._load_active_jobs()
    
    def _get_progress_file(self, job_id: str) -> Path:
        """Get the progress file path for a job"""
        return self.progress_dir / f"{job_id}.json"
    
    def _serialize_sets(self, obj):
        """Custom serializer for sets in JSON"""
        if isinstance(obj, set):
            return sorted(list(obj))
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, (CrawlJobStatus, SiteCrawlStatus)):
            return obj.value
        if isinstance(obj, dict):
            return {k: self._serialize_sets(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._serialize_sets(v) for v in obj]
        # Let dataclass asdict() handle object serialization instead of recursive __dict__ access
        return obj
    
    def _deserialize_sets(self, obj, progress_data=None):
        """Custom deserializer for sets from JSON"""
        if progress_data is None:
            progress_data = obj
        
        # Convert URL lists back to sets
        for field in ['discovered_urls', 'crawled_urls', 'failed_urls', 'remaining_urls']:
            if field in progress_data:
                progress_data[field] = set(progress_data[field])
        
        # Convert datetime strings back to datetime objects
        for field in ['start_time', 'last_update', 'completion_time', 'last_checkpoint']:
            if field in progress_data and progress_data[field]:
                progress_data[field] = datetime.fromisoformat(progress_data[field])
        
        return progress_data
    
    def create_job(self, job_id: str, site_ids: List[str], site_configs: Dict[str, Any]) -> JobProgress:
        """Create a new crawl job with progress tracking"""
        site_progresses = {}
        
        for site_id in site_ids:
            site_config = site_configs.get(site_id)
            if site_config:
                progress = CrawlProgress(
                    site_id=site_id,
                    site_name=site_config.get('name', 'Unknown Site'),
                    status=SiteCrawlStatus.PENDING,
                    discovered_urls=set(),
                    crawled_urls=set(),
                    failed_urls=set(),
                    remaining_urls=set(),
                    max_depth=site_config.get('limits', {}).get('max_depth', 3)
                )
                site_progresses[site_id] = progress
        
        job_progress = JobProgress(
            job_id=job_id,
            status=CrawlJobStatus.PENDING,
            site_progresses=site_progresses,
            start_time=datetime.now(),
            last_update=datetime.now()
        )
        
        self.active_jobs[job_id] = job_progress
        self.save_progress(job_id)
        
        logger.info(f"Created crawl job {job_id} with {len(site_ids)} sites")
        return job_progress
    
    def update_site_progress(self, job_id: str, site_id: str, **updates) -> bool:
        """Update progress for a specific site"""
        if job_id not in self.active_jobs:
            logger.warning(f"Job {job_id} not found in active jobs")
            return False
        
        job_progress = self.active_jobs[job_id]
        if site_id not in job_progress.site_progresses:
            logger.warning(f"Site {site_id} not found in job {job_id}")
            return False
        
        site_progress = job_progress.site_progresses[site_id]
        
        # Update fields
        for field, value in updates.items():
            if hasattr(site_progress, field):
                setattr(site_progress, field, value)
        
        # Update timestamps
        site_progress.last_update = datetime.now()
        job_progress.last_update = datetime.now()
        
        # Check if we should save a checkpoint
        self._maybe_checkpoint(job_id)
        
        return True
    
    def add_discovered_urls(self, job_id: str, site_id: str, urls: List[str]) -> bool:
        """Add newly discovered URLs to the progress"""
        if job_id not in self.active_jobs:
            return False
        
        site_progress = self.active_jobs[job_id].site_progresses.get(site_id)
        if not site_progress:
            return False
        
        new_urls = set(urls) - site_progress.crawled_urls - site_progress.failed_urls
        site_progress.discovered_urls.update(new_urls)
        site_progress.remaining_urls.update(new_urls)
        
        self.update_site_progress(job_id, site_id, status=SiteCrawlStatus.DISCOVERING)
        return True
    
    def mark_url_crawled(self, job_id: str, site_id: str, url: str, success: bool = True) -> bool:
        """Mark a URL as crawled (successfully or failed)"""
        if job_id not in self.active_jobs:
            return False
        
        site_progress = self.active_jobs[job_id].site_progresses.get(site_id)
        if not site_progress:
            return False
        
        site_progress.remaining_urls.discard(url)
        
        if success:
            site_progress.crawled_urls.add(url)
            site_progress.articles_found += 1
        else:
            site_progress.failed_urls.add(url)
        
        # Update status if actively crawling
        if site_progress.status == SiteCrawlStatus.PENDING:
            site_progress.status = SiteCrawlStatus.CRAWLING
        
        self.update_site_progress(job_id, site_id)
        return True
    
    def complete_site(self, job_id: str, site_id: str, articles_count: int, chunks_count: int = 0) -> bool:
        """Mark a site as completed"""
        updates = {
            'status': SiteCrawlStatus.COMPLETED,
            'completion_time': datetime.now(),
            'articles_processed': articles_count,
            'chunks_created': chunks_count
        }
        
        success = self.update_site_progress(job_id, site_id, **updates)
        
        if success:
            # Update job-level metrics
            job_progress = self.active_jobs[job_id]
            job_progress.total_articles += articles_count
            job_progress.total_chunks += chunks_count
            
            # Check if all sites are complete
            all_complete = all(
                p.status in [SiteCrawlStatus.COMPLETED, SiteCrawlStatus.FAILED, SiteCrawlStatus.SKIPPED]
                for p in job_progress.site_progresses.values()
            )
            
            if all_complete:
                self.complete_job(job_id)
        
        return success
    
    def fail_site(self, job_id: str, site_id: str, error: str) -> bool:
        """Mark a site as failed"""
        if job_id not in self.active_jobs:
            return False
        
        site_progress = self.active_jobs[job_id].site_progresses.get(site_id)
        if not site_progress:
            return False
        
        site_progress.status = SiteCrawlStatus.FAILED
        site_progress.completion_time = datetime.now()
        site_progress.errors.append(error)
        
        # Update job-level error count
        self.active_jobs[job_id].total_errors += 1
        
        self.update_site_progress(job_id, site_id)
        return True
    
    def complete_job(self, job_id: str) -> bool:
        """Mark a job as completed"""
        if job_id not in self.active_jobs:
            return False
        
        job_progress = self.active_jobs[job_id]
        job_progress.status = CrawlJobStatus.COMPLETED
        job_progress.completion_time = datetime.now()
        job_progress.last_update = datetime.now()
        
        self.save_progress(job_id)
        logger.info(f"Crawl job {job_id} completed")
        return True
    
    def fail_job(self, job_id: str, error: str) -> bool:
        """Mark a job as failed"""
        if job_id not in self.active_jobs:
            return False
        
        job_progress = self.active_jobs[job_id]
        job_progress.status = CrawlJobStatus.FAILED
        job_progress.completion_time = datetime.now()
        job_progress.last_update = datetime.now()
        
        self.save_progress(job_id)
        logger.error(f"Crawl job {job_id} failed: {error}")
        return True
    
    def get_job_progress(self, job_id: str) -> Optional[JobProgress]:
        """Get progress for a specific job"""
        return self.active_jobs.get(job_id)
    
    def get_site_progress(self, job_id: str, site_id: str) -> Optional[CrawlProgress]:
        """Get progress for a specific site within a job"""
        job_progress = self.active_jobs.get(job_id)
        if job_progress:
            return job_progress.site_progresses.get(site_id)
        return None
    
    def _maybe_checkpoint(self, job_id: str):
        """Save a checkpoint if enough time has passed"""
        job_progress = self.active_jobs.get(job_id)
        if not job_progress:
            return
        
        now = datetime.now()
        if (job_progress.last_checkpoint is None or 
            (now - job_progress.last_checkpoint).total_seconds() >= job_progress.checkpoint_interval):
            
            self.save_progress(job_id)
            job_progress.last_checkpoint = now
    
    def save_progress(self, job_id: str) -> bool:
        """Save progress to disk"""
        if job_id not in self.active_jobs:
            return False
        
        try:
            progress_file = self._get_progress_file(job_id)
            job_progress = self.active_jobs[job_id]
            
            # Convert to serializable format
            data = self._serialize_sets(asdict(job_progress))
            
            with open(progress_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved progress for job {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save progress for job {job_id}: {str(e)}")
            return False
    
    def load_progress(self, job_id: str) -> Optional[JobProgress]:
        """Load progress from disk"""
        try:
            progress_file = self._get_progress_file(job_id)
            if not progress_file.exists():
                return None
            
            with open(progress_file, 'r') as f:
                data = json.load(f)
            
            # Deserialize the data
            self._deserialize_sets(None, data)
            
            # Reconstruct site progresses
            site_progresses = {}
            for site_id, site_data in data['site_progresses'].items():
                self._deserialize_sets(None, site_data)
                site_data['status'] = SiteCrawlStatus(site_data['status'])
                progress = CrawlProgress(**site_data)
                site_progresses[site_id] = progress
            
            # Reconstruct job progress
            data['site_progresses'] = site_progresses
            data['status'] = CrawlJobStatus(data['status'])
            job_progress = JobProgress(**data)
            
            self.active_jobs[job_id] = job_progress
            logger.info(f"Loaded progress for job {job_id}")
            return job_progress
            
        except Exception as e:
            logger.error(f"Failed to load progress for job {job_id}: {str(e)}")
            return None
    
    def _load_active_jobs(self):
        """Load all active job progress files"""
        if not self.progress_dir.exists():
            return
        
        for progress_file in self.progress_dir.glob("*.json"):
            job_id = progress_file.stem
            try:
                progress = self.load_progress(job_id)
                if progress and progress.status in [CrawlJobStatus.RUNNING, CrawlJobStatus.PAUSED]:
                    logger.info(f"Loaded active job {job_id} for potential recovery")
            except Exception as e:
                logger.warning(f"Failed to load progress file {progress_file}: {str(e)}")
    
    def get_recoverable_jobs(self) -> List[str]:
        """Get list of jobs that can be recovered"""
        recoverable = []
        for job_id, progress in self.active_jobs.items():
            if progress.status in [CrawlJobStatus.RUNNING, CrawlJobStatus.PAUSED]:
                recoverable.append(job_id)
        return recoverable
    
    def cleanup_completed_jobs(self, max_age_days: int = 7):
        """Clean up old completed job progress files"""
        cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 3600)
        
        for progress_file in self.progress_dir.glob("*.json"):
            try:
                if progress_file.stat().st_mtime < cutoff_time:
                    job_id = progress_file.stem
                    if job_id in self.active_jobs:
                        progress = self.active_jobs[job_id]
                        if progress.status in [CrawlJobStatus.COMPLETED, CrawlJobStatus.FAILED]:
                            progress_file.unlink()
                            del self.active_jobs[job_id]
                            logger.info(f"Cleaned up old progress file for job {job_id}")
            except Exception as e:
                logger.warning(f"Failed to cleanup progress file {progress_file}: {str(e)}")

    def pause_job(self, job_id: str) -> bool:
        """Pause a running crawl job"""
        if job_id not in self.active_jobs:
            logger.warning(f"Job {job_id} not found")
            return False
        
        job_progress = self.active_jobs[job_id]
        
        if job_progress.status != CrawlJobStatus.RUNNING:
            logger.warning(f"Job {job_id} is not running (status: {job_progress.status.value})")
            return False
        
        # Pause the job
        job_progress.status = CrawlJobStatus.PAUSED
        job_progress.last_update = datetime.now()
        
        # Pause all active sites in the job
        for site_progress in job_progress.site_progresses.values():
            if site_progress.status in [SiteCrawlStatus.DISCOVERING, SiteCrawlStatus.CRAWLING, SiteCrawlStatus.PROCESSING]:
                site_progress.status = SiteCrawlStatus.PAUSED
                site_progress.last_update = datetime.now()
        
        # Save current state
        self.save_progress(job_id)
        logger.info(f"Paused crawl job {job_id}")
        return True

    def stop_job(self, job_id: str) -> bool:
        """Stop a crawl job (cancel it)"""
        if job_id not in self.active_jobs:
            logger.warning(f"Job {job_id} not found")
            return False
        
        job_progress = self.active_jobs[job_id]
        
        if job_progress.status in [CrawlJobStatus.COMPLETED, CrawlJobStatus.FAILED, CrawlJobStatus.CANCELLED]:
            logger.warning(f"Job {job_id} is already finished (status: {job_progress.status.value})")
            return False
        
        # Stop the job
        job_progress.status = CrawlJobStatus.CANCELLED
        job_progress.completion_time = datetime.now()
        job_progress.last_update = datetime.now()
        
        # Mark all incomplete sites as cancelled (failed)
        for site_progress in job_progress.site_progresses.values():
            if site_progress.status not in [SiteCrawlStatus.COMPLETED, SiteCrawlStatus.FAILED]:
                site_progress.status = SiteCrawlStatus.FAILED
                site_progress.completion_time = datetime.now()
                site_progress.last_update = datetime.now()
                site_progress.errors.append("Job cancelled by user")
        
        # Save final state
        self.save_progress(job_id)
        logger.info(f"Stopped crawl job {job_id}")
        return True

    def resume_job(self, job_id: str) -> bool:
        """Resume a paused crawl job"""
        if job_id not in self.active_jobs:
            logger.warning(f"Job {job_id} not found")
            return False
        
        job_progress = self.active_jobs[job_id]
        
        if job_progress.status != CrawlJobStatus.PAUSED:
            logger.warning(f"Job {job_id} is not paused (status: {job_progress.status.value})")
            return False
        
        # Resume the job
        job_progress.status = CrawlJobStatus.RUNNING
        job_progress.last_update = datetime.now()
        
        # Resume all paused sites in the job
        for site_progress in job_progress.site_progresses.values():
            if site_progress.status == SiteCrawlStatus.PAUSED:
                # Determine what state to resume to based on progress
                if site_progress.discovered_urls and site_progress.remaining_urls:
                    site_progress.status = SiteCrawlStatus.CRAWLING
                elif site_progress.discovered_urls and not site_progress.remaining_urls:
                    site_progress.status = SiteCrawlStatus.PROCESSING
                else:
                    site_progress.status = SiteCrawlStatus.DISCOVERING
                site_progress.last_update = datetime.now()
        
        # Save state
        self.save_progress(job_id)
        logger.info(f"Resumed crawl job {job_id}")
        return True

    def get_resumable_sites(self, job_id: str) -> List[str]:
        """Get list of sites that can be resumed for a job"""
        if job_id not in self.active_jobs:
            return []
        
        job_progress = self.active_jobs[job_id]
        resumable_sites = []
        
        for site_id, site_progress in job_progress.site_progresses.items():
            if site_progress.status in [SiteCrawlStatus.PAUSED, SiteCrawlStatus.DISCOVERING, SiteCrawlStatus.CRAWLING]:
                resumable_sites.append(site_id)
        
        return resumable_sites