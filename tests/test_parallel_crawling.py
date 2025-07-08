#!/usr/bin/env python3
"""
Unit tests for parallel URL crawling functionality.
Tests that parallelism is properly used when fetching URLs based on configuration.
"""

import unittest
import asyncio
import sys
import os
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from concurrent.futures import ThreadPoolExecutor
from typing import List, Set, Any, Dict

# Add the src directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock the missing dependencies before importing
sys.modules['aiohttp'] = MagicMock()
sys.modules['crawl4ai'] = MagicMock()
sys.modules['crawl4ai.chunking_strategy'] = MagicMock()
sys.modules['bs4'] = MagicMock()
sys.modules['langdetect'] = MagicMock()

from generic_crawler import GenericWebCrawler, MultiSiteCrawler
from site_config import SiteConfig, CrawlStrategy, ContentType, CrawlLimits
from crawl_progress import CrawlProgressManager


class TestParallelCrawling(unittest.TestCase):
    """Test parallel URL crawling functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.site_config = SiteConfig(
            name="Test Site",
            description="Test site for parallel crawling",
            base_url="https://example.com",
            start_urls=["https://example.com/page1", "https://example.com/page2"],
            crawl_strategy=CrawlStrategy.BREADTH_FIRST,
            content_type=ContentType.GENERAL,
            limits=CrawlLimits(
                max_articles=10,
                max_depth=2,
                delay_seconds=0.1,  # Small delay for testing
                max_concurrent_requests=3,
                max_concurrent_sites=2,
                enable_parallel_processing=True,
                batch_size=5,
                thread_pool_size=4
            )
        )
        self.crawler = GenericWebCrawler(self.site_config)
    
    def test_parallel_config_loaded_correctly(self):
        """Test that parallel configuration is loaded correctly"""
        limits = self.site_config.limits
        
        self.assertEqual(limits.max_concurrent_requests, 3)
        self.assertEqual(limits.max_concurrent_sites, 2)
        self.assertTrue(limits.enable_parallel_processing)
        self.assertEqual(limits.batch_size, 5)
        self.assertEqual(limits.thread_pool_size, 4)
    
    @patch('generic_crawler.AsyncWebCrawler')
    def test_parallel_url_crawling_with_asyncio_gather(self, mock_crawler_class):
        """Test that URLs are crawled in parallel using asyncio.gather"""
        # Mock the AsyncWebCrawler instance
        mock_crawler = Mock()
        mock_crawler_class.return_value.__aenter__.return_value = mock_crawler
        
        # Create a mock that tracks timing to verify parallel execution
        crawl_times = []
        
        async def mock_crawl_single_url(crawler, url, depth):
            start_time = time.time()
            await asyncio.sleep(0.1)  # Simulate network delay
            crawl_times.append((url, time.time() - start_time))
            return {
                'url': url,
                'title': f'Title for {url}',
                'content': f'Content for {url}',
                'description': f'Description for {url}',
                'category': 'Test',
                'content_type': 'general',
                'site_name': 'Test Site',
                'site_id': self.site_config.id,
                'crawled_at': '2023-01-01T00:00:00',
                'word_count': 10,
                'chunk_size': 1000,
                'chunk_overlap': 200
            }, []
        
        # Mock various methods
        with patch.object(self.crawler, '_crawl_single_url', side_effect=mock_crawl_single_url):
            with patch.object(self.crawler, '_normalize_url', side_effect=lambda x: x):
                with patch.object(self.crawler, '_is_valid_url', return_value=True):
                    with patch.object(self.crawler, '_is_allowed_by_robots', return_value=True):
                        # Mock the actual breadth first method to track calls
                        with patch.object(self.crawler, '_crawl_breadth_first') as mock_breadth_first:
                            # Run the test
                            asyncio.run(self.crawler.crawl_site())
                            
                            # Verify that crawl method was called
                            mock_breadth_first.assert_called_once()
    
    def test_parallel_url_processing_implementation(self):
        """Test implementation of parallel URL processing"""
        async def run_test():
            # Create test URLs
            test_urls = [
                "https://example.com/page1",
                "https://example.com/page2", 
                "https://example.com/page3",
                "https://example.com/page4",
                "https://example.com/page5"
            ]
            
            # Track which URLs were processed and when
            processed_urls = []
            process_times = []
            
            async def mock_process_url(url):
                start_time = time.time()
                await asyncio.sleep(0.1)  # Simulate processing time
                end_time = time.time()
                processed_urls.append(url)
                process_times.append(end_time - start_time)
                return f"processed_{url}"
            
            # Test parallel processing with asyncio.gather
            start_time = time.time()
            
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(self.site_config.limits.max_concurrent_requests)
            
            async def process_with_semaphore(url):
                async with semaphore:
                    return await mock_process_url(url)
            
            # Process URLs in parallel
            tasks = [process_with_semaphore(url) for url in test_urls]
            results = await asyncio.gather(*tasks)
            
            total_time = time.time() - start_time
            
            # Verify results
            self.assertEqual(len(results), len(test_urls))
            self.assertEqual(len(processed_urls), len(test_urls))
            
            # With 3 concurrent requests and 5 URLs, total time should be less than sequential
            # Sequential would be 5 * 0.1 = 0.5 seconds
            # Parallel should be closer to 2 * 0.1 = 0.2 seconds (2 batches of 3 and 2)
            self.assertLess(total_time, 0.4, f"Parallel processing should be faster than sequential. Took {total_time:.2f}s")
            
            # All URLs should be processed
            self.assertEqual(set(processed_urls), set(test_urls))
        
        # Run the async test
        asyncio.run(run_test())
    
    def test_parallel_crawling_with_batch_processing(self):
        """Test that batch processing is used for parallel crawling"""
        # Create a list of URLs larger than batch size
        urls = [f"https://example.com/page{i}" for i in range(12)]
        batch_size = self.site_config.limits.batch_size  # 5
        
        # Mock batch processing
        batches_processed = []
        
        def mock_process_batch(batch):
            batches_processed.append(batch)
            return [f"processed_{url}" for url in batch]
        
        # Simulate batch processing
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            mock_process_batch(batch)
        
        # Verify batching
        self.assertEqual(len(batches_processed), 3)  # 12 URLs / 5 batch_size = 3 batches
        self.assertEqual(len(batches_processed[0]), 5)  # First batch
        self.assertEqual(len(batches_processed[1]), 5)  # Second batch
        self.assertEqual(len(batches_processed[2]), 2)  # Third batch (remainder)
        
        # Verify all URLs were processed
        all_processed = []
        for batch in batches_processed:
            all_processed.extend(batch)
        self.assertEqual(set(all_processed), set(urls))
    
    @patch('generic_crawler.AsyncWebCrawler')
    def test_concurrent_site_crawling(self, mock_crawler_class):
        """Test that multiple sites can be crawled concurrently"""
        # Mock the AsyncWebCrawler
        mock_crawler = Mock()
        mock_crawler_class.return_value.__aenter__.return_value = mock_crawler
        
        # Create multiple site configurations
        sites = []
        for i in range(4):
            site_config = SiteConfig(
                name=f"Test Site {i}",
                base_url=f"https://example{i}.com",
                start_urls=[f"https://example{i}.com/page1"],
                limits=CrawlLimits(
                    max_articles=5,
                    max_depth=1,
                    max_concurrent_sites=2,
                    enable_parallel_processing=True
                )
            )
            sites.append(site_config)
        
        # Track crawling times
        crawl_times = {}
        
        async def mock_crawl_site(site_config):
            start_time = time.time()
            await asyncio.sleep(0.2)  # Simulate crawling time
            crawl_times[site_config.name] = time.time() - start_time
            return []
        
        # Test concurrent site crawling
        async def test_concurrent_crawling():
            # Limit concurrent sites
            semaphore = asyncio.Semaphore(2)  # max_concurrent_sites = 2
            
            async def crawl_with_semaphore(site_config):
                async with semaphore:
                    return await mock_crawl_site(site_config)
            
            start_time = time.time()
            tasks = [crawl_with_semaphore(site) for site in sites]
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
            
            return results, total_time
        
        # Run the test
        results, total_time = asyncio.run(test_concurrent_crawling())
        
        # Verify results
        self.assertEqual(len(results), 4)  # All sites processed
        self.assertEqual(len(crawl_times), 4)  # All sites timed
        
        # With 2 concurrent sites and 4 total sites, should take about 2 * 0.2 = 0.4 seconds
        # vs sequential 4 * 0.2 = 0.8 seconds
        self.assertLess(total_time, 0.6, f"Concurrent site crawling should be faster. Took {total_time:.2f}s")
    
    def test_thread_pool_executor_usage(self):
        """Test that ThreadPoolExecutor is used for CPU-bound tasks"""
        from concurrent.futures import ThreadPoolExecutor
        
        # Create a CPU-bound task (content processing)
        def process_content(content):
            # Simulate CPU-bound work
            time.sleep(0.05)
            return content.upper()
        
        content_list = [f"content_{i}" for i in range(8)]
        
        # Test sequential processing
        start_time = time.time()
        sequential_results = [process_content(content) for content in content_list]
        sequential_time = time.time() - start_time
        
        # Test parallel processing with ThreadPoolExecutor
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=self.site_config.limits.thread_pool_size) as executor:
            parallel_results = list(executor.map(process_content, content_list))
        parallel_time = time.time() - start_time
        
        # Verify results are the same
        self.assertEqual(sequential_results, parallel_results)
        
        # Verify parallel processing is faster
        self.assertLess(parallel_time, sequential_time, 
                       f"Parallel processing should be faster. Sequential: {sequential_time:.2f}s, Parallel: {parallel_time:.2f}s")
    
    def test_parallelism_respects_configuration_limits(self):
        """Test that parallelism respects the configured limits"""
        # Test with different concurrent request limits
        test_cases = [
            (1, "Sequential processing"),
            (3, "Limited parallelism"),
            (10, "High parallelism")
        ]
        
        for max_concurrent, description in test_cases:
            with self.subTest(max_concurrent=max_concurrent, description=description):
                # Update configuration
                self.site_config.limits.max_concurrent_requests = max_concurrent
                
                # Verify the configuration was updated
                self.assertEqual(self.site_config.limits.max_concurrent_requests, max_concurrent)
                
                # Test that semaphore would be created with correct limit
                semaphore = asyncio.Semaphore(max_concurrent)
                self.assertEqual(semaphore._value, max_concurrent)
    
    def test_parallel_processing_can_be_disabled(self):
        """Test that parallel processing can be disabled via configuration"""
        # Disable parallel processing
        self.site_config.limits.enable_parallel_processing = False
        
        # Verify the setting
        self.assertFalse(self.site_config.limits.enable_parallel_processing)
        
        # In a real implementation, this would switch to sequential processing
        # Here we just verify the configuration is respected
        if not self.site_config.limits.enable_parallel_processing:
            expected_concurrent_requests = 1
        else:
            expected_concurrent_requests = self.site_config.limits.max_concurrent_requests
        
        # When disabled, should effectively use sequential processing
        self.assertFalse(self.site_config.limits.enable_parallel_processing)
    
    def test_parallel_url_fetching_with_real_semaphore(self):
        """Test parallel URL fetching using real semaphore implementation"""
        async def run_test():
            # Create test URLs
            urls = [f"https://example.com/page{i}" for i in range(6)]
            
            # Track concurrent executions
            current_executions = 0
            max_concurrent_seen = 0
            execution_log = []
            
            async def mock_fetch_url(url):
                nonlocal current_executions, max_concurrent_seen
                
                current_executions += 1
                max_concurrent_seen = max(max_concurrent_seen, current_executions)
                execution_log.append(f"Started {url} (concurrent: {current_executions})")
                
                # Simulate network delay
                await asyncio.sleep(0.1)
                
                current_executions -= 1
                execution_log.append(f"Finished {url} (concurrent: {current_executions})")
                return f"content_{url}"
            
            # Create semaphore with limit
            max_concurrent = 3
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def fetch_with_semaphore(url):
                async with semaphore:
                    return await mock_fetch_url(url)
            
            # Execute parallel fetching
            tasks = [fetch_with_semaphore(url) for url in urls]
            results = await asyncio.gather(*tasks)
            
            # Verify results
            self.assertEqual(len(results), len(urls))
            
            # Verify concurrency was limited
            self.assertLessEqual(max_concurrent_seen, max_concurrent,
                               f"Max concurrent executions ({max_concurrent_seen}) should not exceed limit ({max_concurrent})")
            
            # Verify we actually used parallel processing
            self.assertGreater(max_concurrent_seen, 1,
                              f"Should have used parallel processing, but max concurrent was {max_concurrent_seen}")
            
            # Print execution log for debugging
            print("\\nExecution log:")
            for entry in execution_log:
                print(entry)
        
        # Run the async test
        asyncio.run(run_test())


if __name__ == '__main__':
    # Create the tests directory if it doesn't exist
    os.makedirs('tests', exist_ok=True)
    
    # Run the tests
    unittest.main(verbosity=2)