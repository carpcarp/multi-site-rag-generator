#!/usr/bin/env python3
"""
Unit tests for URL normalization in the generic web crawler.
Tests that URLs have their query parameters removed as expected.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from urllib.parse import urljoin

# Add the src directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from generic_crawler import GenericWebCrawler
from site_config import SiteConfig, CrawlStrategy, ContentType, CrawlLimits


class TestURLNormalization(unittest.TestCase):
    """Test URL normalization functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.site_config = SiteConfig(
            name="Test Site",
            description="Test site for URL normalization",
            base_url="https://example.com",
            start_urls=["https://example.com"],
            crawl_strategy=CrawlStrategy.BREADTH_FIRST,
            content_type=ContentType.GENERAL,
            limits=CrawlLimits(max_articles=10, max_depth=2)
        )
        self.crawler = GenericWebCrawler(self.site_config)
    
    def test_normalize_url_removes_query_parameters(self):
        """Test that _normalize_url removes query parameters"""
        test_cases = [
            # (input_url, expected_output)
            ("https://example.com/page?param=value", "https://example.com/page"),
            ("https://example.com/page?param1=value1&param2=value2", "https://example.com/page"),
            ("https://example.com/page?hubs_content=test&hubs_content-cta=nav", "https://example.com/page"),
            ("https://example.com/page", "https://example.com/page"),  # No query params
            ("https://example.com/", "https://example.com"),  # Root with trailing slash
            ("https://example.com/page/", "https://example.com/page"),  # Trailing slash removed
            ("https://example.com/page#fragment", "https://example.com/page"),  # Fragment removed
            ("https://example.com/page?param=value#fragment", "https://example.com/page"),  # Both removed
        ]
        
        for input_url, expected in test_cases:
            with self.subTest(input_url=input_url):
                result = self.crawler._normalize_url(input_url)
                self.assertEqual(result, expected, 
                    f"Expected {expected}, got {result} for input {input_url}")
    
    def test_normalize_url_handles_hubspot_query_params(self):
        """Test that _normalize_url handles HubSpot-specific query parameters"""
        hubspot_urls = [
            "https://blog.hubspot.com/service/survey-questions?hubs_content=knowledge.hubspot.com/&hubs_content-cta=kb--header-nav-child-link",
            "https://www.hubspot.com/products/artificial-intelligence?hubs_content=www.hubspot.com/products/sales/email-tracking&hubs_content-cta=nav-software-ai",
            "https://www.hubspot.com/products/crm/starter?hubs_content=www.hubspot.com/products/artificial-intelligence&hubs_content-cta=nav-software-starter&hubs_post=blog.hubspot.com/&hubs_post-cta=footer-features-breeze",
            "https://link.chtbl.com/3iN6uLVq?hubs_content=blog.hubspot.com/customers&hubs_content-cta=null&hubs_post-cta=blognavcard-podcasts-myfirstmillion",
        ]
        
        expected_results = [
            "https://blog.hubspot.com/service/survey-questions",
            "https://www.hubspot.com/products/artificial-intelligence",
            "https://www.hubspot.com/products/crm/starter",
            "https://link.chtbl.com/3iN6uLVq",
        ]
        
        for input_url, expected in zip(hubspot_urls, expected_results):
            with self.subTest(input_url=input_url):
                result = self.crawler._normalize_url(input_url)
                self.assertEqual(result, expected,
                    f"Expected {expected}, got {result} for HubSpot URL {input_url}")
    
    def test_extract_links_returns_normalized_urls(self):
        """Test that _extract_links returns normalized URLs without query parameters"""
        # Mock HTML content with links containing query parameters
        html_content = """
        <html>
        <body>
            <a href="/page1?param=value">Link 1</a>
            <a href="/page2?hubs_content=test&hubs_content-cta=nav">Link 2</a>
            <a href="https://example.com/page3?utm_source=test">Link 3</a>
            <a href="/page4">Link 4</a>
        </body>
        </html>
        """
        
        base_url = "https://example.com"
        
        # Mock the _is_valid_url method to return True for all URLs
        with patch.object(self.crawler, '_is_valid_url', return_value=True):
            links = self.crawler._extract_links(html_content, base_url)
        
        # Expected normalized URLs
        expected_links = [
            "https://example.com/page1",
            "https://example.com/page2", 
            "https://example.com/page3",
            "https://example.com/page4"
        ]
        
        # Check that all returned links are normalized (no query parameters)
        for link in links:
            with self.subTest(link=link):
                self.assertNotIn('?', link, f"Link {link} should not contain query parameters")
                self.assertNotIn('#', link, f"Link {link} should not contain fragments")
        
        # Check that we got the expected normalized URLs
        self.assertEqual(set(links), set(expected_links),
            f"Expected {expected_links}, got {links}")
    
    def test_extract_links_deduplicates_normalized_urls(self):
        """Test that _extract_links properly deduplicates URLs based on normalized form"""
        # HTML with duplicate links that differ only by query parameters
        html_content = """
        <html>
        <body>
            <a href="/page1?param1=value1">Link 1</a>
            <a href="/page1?param2=value2">Link 1 with different params</a>
            <a href="/page1">Link 1 without params</a>
            <a href="/page2?utm_source=test">Link 2</a>
            <a href="/page2?utm_campaign=campaign">Link 2 with different params</a>
        </body>
        </html>
        """
        
        base_url = "https://example.com"
        
        # Mock the _is_valid_url method to return True for all URLs
        with patch.object(self.crawler, '_is_valid_url', return_value=True):
            links = self.crawler._extract_links(html_content, base_url)
        
        # Should only get 2 unique normalized URLs
        expected_links = [
            "https://example.com/page1",
            "https://example.com/page2"
        ]
        
        self.assertEqual(len(links), 2, f"Expected 2 unique links, got {len(links)}: {links}")
        self.assertEqual(set(links), set(expected_links),
            f"Expected {expected_links}, got {links}")
    
    def test_normalize_url_handles_edge_cases(self):
        """Test that _normalize_url handles edge cases gracefully"""
        edge_cases = [
            ("", ""),  # Empty string
            ("not-a-url", "not-a-url"),  # Invalid URL format
            ("https://", "https://"),  # Incomplete URL
            ("https://example.com:8080/path?query=value", "https://example.com:8080/path"),  # Port number
            ("https://example.com/path with spaces?query=value", "https://example.com/path with spaces"),  # Spaces in path
        ]
        
        for input_url, expected in edge_cases:
            with self.subTest(input_url=input_url):
                result = self.crawler._normalize_url(input_url)
                self.assertEqual(result, expected,
                    f"Expected {expected}, got {result} for edge case {input_url}")
    
    def test_url_normalization_integration(self):
        """Test that URL normalization works correctly in the crawling process"""
        # This tests the integration between _normalize_url and _extract_links
        # to ensure the bug is fixed
        
        # Create a mock crawler with some already crawled URLs
        self.crawler.crawled_urls = {"https://example.com/already-crawled"}
        
        html_content = """
        <html>
        <body>
            <a href="/new-page?param=value">New Page</a>
            <a href="/already-crawled?different=param">Already Crawled</a>
        </body>
        </html>
        """
        
        base_url = "https://example.com"
        
        # Mock the _is_valid_url method to return True for all URLs
        with patch.object(self.crawler, '_is_valid_url', return_value=True):
            links = self.crawler._extract_links(html_content, base_url)
        
        # Should only get the new page since the other one is already crawled
        # (even though it has different query parameters)
        expected_links = ["https://example.com/new-page"]
        
        self.assertEqual(links, expected_links,
            f"Expected {expected_links}, got {links}")
        
        # Verify that the returned URL is normalized
        self.assertNotIn('?', links[0], "Returned URL should not contain query parameters")


if __name__ == '__main__':
    # Create the tests directory if it doesn't exist
    os.makedirs('tests', exist_ok=True)
    
    # Run the tests
    unittest.main(verbosity=2) 