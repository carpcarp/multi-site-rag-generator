#!/usr/bin/env python3
"""
Unit tests for pause, stop, and resume functionality in the crawl system.
Tests the control mechanisms for managing running crawl jobs.
"""

import unittest
import asyncio
import sys
import os
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime
import tempfile
import shutil

# Add the src directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from generic_crawler import GenericWebCrawler
from site_config import SiteConfig, CrawlStrategy, ContentType, CrawlLimits
from crawl_progress import CrawlProgressManager, CrawlJobStatus, SiteCrawlStatus
from site_management_api import CrawlJobManager


class TestPauseStopResume(unittest.TestCase):
    """Test pause, stop, and resume functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directory for progress files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test site configuration
        self.site_config = SiteConfig(
            id="test-site-1",
            name="Test Site",
            description="Test site for pause/stop/resume",
            base_url="https://example.com",
            start_urls=["https://example.com"],
            crawl_strategy=CrawlStrategy.BREADTH_FIRST,
            content_type=ContentType.GENERAL,
            limits=CrawlLimits(max_articles=10, max_depth=2)
        )
        
        # Create progress manager with temp directory
        self.progress_manager = CrawlProgressManager(progress_dir=self.temp_dir)
        
        # Create job manager
        self.job_manager = CrawlJobManager()
        self.job_manager.progress_manager = self.progress_manager
        
        # Create test crawler
        self.crawler = GenericWebCrawler(
            self.site_config, 
            self.progress_manager, 
            job_id="test-job-1"
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_crawler_control_flags(self):
        """Test that control flags work correctly"""
        # Initially, crawler should be in normal state
        self.assertFalse(self.crawler._should_stop)
        self.assertFalse(self.crawler._should_pause)
        self.assertEqual(self.crawler._check_control_signals(), "continue")
        
        # Test pause signal
        self.crawler.pause_crawl()
        self.assertTrue(self.crawler._should_pause)
        self.assertEqual(self.crawler._check_control_signals(), "pause")
        
        # Test resume from pause
        self.crawler.resume_crawl()
        self.assertFalse(self.crawler._should_pause)
        self.assertEqual(self.crawler._check_control_signals(), "continue")
        
        # Test stop signal
        self.crawler.stop_crawl()
        self.assertTrue(self.crawler._should_stop)
        self.assertEqual(self.crawler._check_control_signals(), "stop")
        
        # Stop should override pause
        self.crawler.pause_crawl()
        self.assertTrue(self.crawler._should_pause)
        self.assertTrue(self.crawler._should_stop)
        self.assertEqual(self.crawler._check_control_signals(), "stop")
    
    async def test_wait_while_paused(self):
        """Test the pause waiting mechanism"""
        # Test that wait_while_paused returns immediately when not paused
        start_time = asyncio.get_event_loop().time()
        await self.crawler._wait_while_paused()
        end_time = asyncio.get_event_loop().time()
        self.assertLess(end_time - start_time, 0.1, "Should return immediately when not paused")
        
        # Test that it waits when paused but exits when resumed
        self.crawler.pause_crawl()
        
        async def resume_after_delay():
            await asyncio.sleep(0.2)  # Wait a bit then resume
            self.crawler.resume_crawl()
        
        # Start resume task
        resume_task = asyncio.create_task(resume_after_delay())
        
        # This should wait until resumed
        start_time = asyncio.get_event_loop().time()
        await self.crawler._wait_while_paused()
        end_time = asyncio.get_event_loop().time()
        
        await resume_task  # Clean up
        self.assertGreater(end_time - start_time, 0.15, "Should wait while paused")
        self.assertFalse(self.crawler._should_pause, "Should be resumed")
        
        # Test that it exits immediately when stopped
        self.crawler.pause_crawl()
        self.crawler.stop_crawl()
        
        start_time = asyncio.get_event_loop().time()
        await self.crawler._wait_while_paused()
        end_time = asyncio.get_event_loop().time()
        self.assertLess(end_time - start_time, 0.1, "Should exit immediately when stopped")
    
    def test_progress_manager_pause_job(self):
        """Test progress manager pause functionality"""
        # Create a test job
        job_id = "test-job-pause"
        site_configs = {
            "site1": {"name": "Test Site 1", "limits": {"max_articles": 10, "max_depth": 2}}
        }
        
        progress = self.progress_manager.create_job(job_id, ["site1"], site_configs)
        progress.status = CrawlJobStatus.RUNNING
        
        # Set site to crawling state
        self.progress_manager.update_site_progress(
            job_id, "site1", status=SiteCrawlStatus.CRAWLING
        )
        
        # Test pause
        success = self.progress_manager.pause_job(job_id)
        self.assertTrue(success)
        
        # Check job status
        updated_progress = self.progress_manager.get_job_progress(job_id)
        self.assertEqual(updated_progress.status, CrawlJobStatus.PAUSED)
        
        # Check site status
        site_progress = self.progress_manager.get_site_progress(job_id, "site1")
        self.assertEqual(site_progress.status, SiteCrawlStatus.PAUSED)
        
        # Test that pausing a non-running job fails
        self.progress_manager.update_site_progress(
            job_id, "site1", status=SiteCrawlStatus.COMPLETED
        )
        updated_progress.status = CrawlJobStatus.COMPLETED
        
        success = self.progress_manager.pause_job(job_id)
        self.assertFalse(success)
    
    def test_progress_manager_stop_job(self):
        """Test progress manager stop functionality"""
        # Create a test job
        job_id = "test-job-stop"
        site_configs = {
            "site1": {"name": "Test Site 1", "limits": {"max_articles": 10, "max_depth": 2}}
        }
        
        progress = self.progress_manager.create_job(job_id, ["site1"], site_configs)
        progress.status = CrawlJobStatus.RUNNING
        
        # Set site to crawling state
        self.progress_manager.update_site_progress(
            job_id, "site1", status=SiteCrawlStatus.CRAWLING
        )
        
        # Test stop
        success = self.progress_manager.stop_job(job_id)
        self.assertTrue(success)
        
        # Check job status
        updated_progress = self.progress_manager.get_job_progress(job_id)
        self.assertEqual(updated_progress.status, CrawlJobStatus.CANCELLED)
        self.assertIsNotNone(updated_progress.completion_time)
        
        # Check site status
        site_progress = self.progress_manager.get_site_progress(job_id, "site1")
        self.assertEqual(site_progress.status, SiteCrawlStatus.FAILED)
        self.assertIn("Job cancelled by user", site_progress.errors)
        
        # Test that stopping an already finished job fails
        success = self.progress_manager.stop_job(job_id)
        self.assertFalse(success)
    
    def test_progress_manager_resume_job(self):
        """Test progress manager resume functionality"""
        # Create a test job
        job_id = "test-job-resume"
        site_configs = {
            "site1": {"name": "Test Site 1", "limits": {"max_articles": 10, "max_depth": 2}}
        }
        
        progress = self.progress_manager.create_job(job_id, ["site1"], site_configs)
        progress.status = CrawlJobStatus.PAUSED
        
        # Set site to paused state with some discovered URLs
        site_progress = self.progress_manager.get_site_progress(job_id, "site1")
        site_progress.status = SiteCrawlStatus.PAUSED
        site_progress.discovered_urls.add("https://example.com/page1")
        site_progress.remaining_urls.add("https://example.com/page1")
        
        # Test resume
        success = self.progress_manager.resume_job(job_id)
        self.assertTrue(success)
        
        # Check job status
        updated_progress = self.progress_manager.get_job_progress(job_id)
        self.assertEqual(updated_progress.status, CrawlJobStatus.RUNNING)
        
        # Check site status (should resume to crawling since it has remaining URLs)
        site_progress = self.progress_manager.get_site_progress(job_id, "site1")
        self.assertEqual(site_progress.status, SiteCrawlStatus.CRAWLING)
        
        # Test resuming a non-paused job
        success = self.progress_manager.resume_job(job_id)
        self.assertFalse(success)
    
    def test_get_resumable_sites(self):
        """Test getting list of resumable sites"""
        # Create a test job with multiple sites
        job_id = "test-job-resumable"
        site_configs = {
            "site1": {"name": "Test Site 1", "limits": {"max_articles": 10, "max_depth": 2}},
            "site2": {"name": "Test Site 2", "limits": {"max_articles": 10, "max_depth": 2}},
            "site3": {"name": "Test Site 3", "limits": {"max_articles": 10, "max_depth": 2}}
        }
        
        progress = self.progress_manager.create_job(job_id, ["site1", "site2", "site3"], site_configs)
        
        # Set different states for sites
        self.progress_manager.update_site_progress(
            job_id, "site1", status=SiteCrawlStatus.PAUSED
        )
        self.progress_manager.update_site_progress(
            job_id, "site2", status=SiteCrawlStatus.COMPLETED
        )
        self.progress_manager.update_site_progress(
            job_id, "site3", status=SiteCrawlStatus.CRAWLING
        )
        
        # Get resumable sites
        resumable = self.progress_manager.get_resumable_sites(job_id)
        
        # Should include paused and crawling sites, but not completed ones
        self.assertIn("site1", resumable)  # Paused
        self.assertIn("site3", resumable)  # Crawling
        self.assertNotIn("site2", resumable)  # Completed
    
    def test_job_manager_crawler_registration(self):
        """Test crawler registration and control in job manager"""
        job_id = "test-job-registration"
        site_id = "test-site"
        
        # Test registration
        self.job_manager.register_crawler(job_id, site_id, self.crawler)
        self.assertIn(job_id, self.job_manager.active_crawlers)
        self.assertIn(site_id, self.job_manager.active_crawlers[job_id])
        self.assertEqual(self.job_manager.active_crawlers[job_id][site_id], self.crawler)
        
        # Test pause signal
        self.job_manager.pause_job_crawlers(job_id)
        self.assertTrue(self.crawler._should_pause)
        
        # Test resume signal
        self.job_manager.resume_job_crawlers(job_id)
        self.assertFalse(self.crawler._should_pause)
        
        # Test stop signal
        self.job_manager.stop_job_crawlers(job_id)
        self.assertTrue(self.crawler._should_stop)
        
        # Test unregistration
        self.job_manager.unregister_crawler(job_id, site_id)
        self.assertNotIn(job_id, self.job_manager.active_crawlers)
    
    def test_site_status_resume_logic(self):
        """Test the logic for determining what status to resume to"""
        job_id = "test-resume-logic"
        site_configs = {
            "site1": {"name": "Test Site 1", "limits": {"max_articles": 10, "max_depth": 2}}
        }
        
        progress = self.progress_manager.create_job(job_id, ["site1"], site_configs)
        progress.status = CrawlJobStatus.PAUSED
        
        site_progress = self.progress_manager.get_site_progress(job_id, "site1")
        site_progress.status = SiteCrawlStatus.PAUSED
        
        # Test case 1: No URLs discovered yet -> should resume to DISCOVERING
        success = self.progress_manager.resume_job(job_id)
        self.assertTrue(success)
        site_progress = self.progress_manager.get_site_progress(job_id, "site1")
        self.assertEqual(site_progress.status, SiteCrawlStatus.DISCOVERING)
        
        # Reset to paused
        progress.status = CrawlJobStatus.PAUSED
        site_progress.status = SiteCrawlStatus.PAUSED
        
        # Test case 2: URLs discovered but remaining -> should resume to CRAWLING
        site_progress.discovered_urls.add("https://example.com/page1")
        site_progress.remaining_urls.add("https://example.com/page1")
        
        success = self.progress_manager.resume_job(job_id)
        self.assertTrue(success)
        site_progress = self.progress_manager.get_site_progress(job_id, "site1")
        self.assertEqual(site_progress.status, SiteCrawlStatus.CRAWLING)
        
        # Reset to paused
        progress.status = CrawlJobStatus.PAUSED
        site_progress.status = SiteCrawlStatus.PAUSED
        
        # Test case 3: URLs discovered but none remaining -> should resume to PROCESSING
        site_progress.remaining_urls.clear()
        
        success = self.progress_manager.resume_job(job_id)
        self.assertTrue(success)
        site_progress = self.progress_manager.get_site_progress(job_id, "site1")
        self.assertEqual(site_progress.status, SiteCrawlStatus.PROCESSING)


class AsyncTestCase(unittest.TestCase):
    """Base class for async test cases"""
    
    def setUp(self):
        """Set up async test environment"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up async test environment"""
        self.loop.close()
    
    def run_async(self, coro):
        """Helper to run async functions in tests"""
        return self.loop.run_until_complete(coro)


class TestAsyncPauseStopResume(AsyncTestCase):
    """Test async pause, stop, and resume functionality"""
    
    def setUp(self):
        """Set up async test fixtures"""
        super().setUp()
        
        # Create temporary directory for progress files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test site configuration
        self.site_config = SiteConfig(
            id="test-site-async",
            name="Test Site Async",
            description="Test site for async pause/stop/resume",
            base_url="https://example.com",
            start_urls=["https://example.com"],
            crawl_strategy=CrawlStrategy.BREADTH_FIRST,
            content_type=ContentType.GENERAL,
            limits=CrawlLimits(max_articles=5, max_depth=1)
        )
        
        # Create progress manager with temp directory
        self.progress_manager = CrawlProgressManager(progress_dir=self.temp_dir)
        
        # Create test crawler
        self.crawler = GenericWebCrawler(
            self.site_config, 
            self.progress_manager, 
            job_id="test-job-async"
        )
    
    def tearDown(self):
        """Clean up async test fixtures"""
        super().tearDown()
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_wait_while_paused_async(self):
        """Test the async pause waiting mechanism"""
        self.run_async(self._test_wait_while_paused_async())
    
    async def _test_wait_while_paused_async(self):
        """Actual async test for wait_while_paused"""
        # Test normal operation (not paused)
        start_time = self.loop.time()
        await self.crawler._wait_while_paused()
        end_time = self.loop.time()
        self.assertLess(end_time - start_time, 0.1)
        
        # Test pause and resume
        self.crawler.pause_crawl()
        
        # Create a task that will resume the crawler after a short delay
        async def resume_later():
            await asyncio.sleep(0.1)
            self.crawler.resume_crawl()
        
        resume_task = asyncio.create_task(resume_later())
        
        start_time = self.loop.time()
        await self.crawler._wait_while_paused()
        end_time = self.loop.time()
        
        await resume_task
        
        # Should have waited for the resume
        self.assertGreater(end_time - start_time, 0.05)
        self.assertFalse(self.crawler._should_pause)


if __name__ == '__main__':
    # Run both test classes
    unittest.main(verbosity=2) 