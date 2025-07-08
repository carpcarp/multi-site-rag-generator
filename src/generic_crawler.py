#!/usr/bin/env python3
"""
Generic Web Crawler for Multi-Site RAG System
"""

import asyncio
import json
import os
import re
import logging
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urljoin, urlparse, urlencode
from urllib.robotparser import RobotFileParser
from datetime import datetime
import aiohttp
import time
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler
from crawl4ai.chunking_strategy import RegexChunking

from site_config import SiteConfig, CrawlStrategy, ContentType, SiteConfigManager
from crawl_progress import CrawlProgressManager, SiteCrawlStatus

logger = logging.getLogger(__name__)


class GenericWebCrawler:
    """Generic web crawler that works with any website based on configuration"""
    
    def __init__(self, site_config: SiteConfig, progress_manager: Optional[CrawlProgressManager] = None, 
                 job_id: Optional[str] = None):
        self.config = site_config
        self.crawled_urls: Set[str] = set()
        self.articles: List[Dict[str, Any]] = []
        self.failed_urls: Set[str] = set()
        
        # Progress tracking
        self.progress_manager = progress_manager
        self.job_id = job_id
        
        # Control flags for pause/stop functionality
        self._should_stop = False
        self._should_pause = False
        
        # Robots.txt cache
        self.robots_cache: Dict[str, RobotFileParser] = {}
        
        # Session for HTTP requests
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def crawl_site(self) -> List[Dict[str, Any]]:
        """Main crawling method with progress tracking"""
        logger.info(f"Starting crawl of {self.config.name}")
        logger.info(f"Strategy: {self.config.crawl_strategy.value}")
        logger.info(f"Max articles: {self.config.limits.max_articles}")
        logger.info(f"Max depth: {self.config.limits.max_depth}")
        
        start_time = datetime.now()
        
        # Initialize progress tracking
        if self.progress_manager and self.job_id:
            self.progress_manager.update_site_progress(
                self.job_id, self.config.id, 
                status=SiteCrawlStatus.DISCOVERING,
                start_time=start_time
            )
        
        try:
            async with AsyncWebCrawler(
                headless=True,
                verbose=True,
                user_agent=f"GenericRAGBot/1.0 (+{self.config.base_url})"
            ) as crawler:
                
                if self.config.crawl_strategy == CrawlStrategy.SITEMAP:
                    await self._crawl_from_sitemap(crawler)
                elif self.config.crawl_strategy == CrawlStrategy.URL_LIST:
                    await self._crawl_url_list(crawler)
                elif self.config.crawl_strategy == CrawlStrategy.BREADTH_FIRST:
                    await self._crawl_breadth_first(crawler)
                else:  # DEPTH_FIRST
                    await self._crawl_depth_first(crawler)
        
        except Exception as e:
            logger.error(f"Error during crawling: {str(e)}")
            # Mark site as failed in progress tracking
            if self.progress_manager and self.job_id:
                self.progress_manager.fail_site(self.job_id, self.config.id, str(e))
            raise
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"Crawling completed in {duration:.2f} seconds")
        logger.info(f"Found {len(self.articles)} articles")
        logger.info(f"Failed URLs: {len(self.failed_urls)}")
        
        # Update site config with results
        self.config.total_articles = len(self.articles)
        self.config.last_crawled = end_time
        self.config.last_crawl_status = "completed" if self.articles else "no_content_found"
        
        # Complete progress tracking
        if self.progress_manager and self.job_id:
            self.progress_manager.complete_site(
                self.job_id, self.config.id, 
                articles_count=len(self.articles)
            )
        
        return self.articles
    
    def pause_crawl(self):
        """Signal the crawler to pause"""
        self._should_pause = True
        logger.info(f"Pause signal sent to crawler for {self.config.name}")
    
    def stop_crawl(self):
        """Signal the crawler to stop"""
        self._should_stop = True
        logger.info(f"Stop signal sent to crawler for {self.config.name}")
    
    def resume_crawl(self):
        """Resume the crawler from pause"""
        self._should_pause = False
        logger.info(f"Resume signal sent to crawler for {self.config.name}")
    
    def _check_control_signals(self) -> str:
        """Check if crawler should pause or stop. Returns action to take."""
        if self._should_stop:
            return "stop"
        elif self._should_pause:
            return "pause"
        return "continue"
    
    async def _wait_while_paused(self):
        """Wait while crawler is paused"""
        while self._should_pause and not self._should_stop:
            logger.debug(f"Crawler for {self.config.name} is paused, waiting...")
            await asyncio.sleep(1)  # Check every second
    
    async def _crawl_breadth_first(self, crawler: AsyncWebCrawler):
        """Breadth-first crawling strategy with progress tracking"""
        current_level = set(self.config.start_urls)
        
        # Track discovered URLs for progress
        if self.progress_manager and self.job_id:
            self.progress_manager.add_discovered_urls(self.job_id, self.config.id, list(current_level))
            self.progress_manager.update_site_progress(
                self.job_id, self.config.id, 
                status=SiteCrawlStatus.CRAWLING,
                current_depth=0
            )
        
        for depth in range(self.config.limits.max_depth + 1):
            # Check for control signals at the start of each depth level
            control_action = self._check_control_signals()
            if control_action == "stop":
                logger.info(f"Stopping crawl of {self.config.name} at depth {depth}")
                break
            elif control_action == "pause":
                logger.info(f"Pausing crawl of {self.config.name} at depth {depth}")
                if self.progress_manager and self.job_id:
                    self.progress_manager.update_site_progress(
                        self.job_id, self.config.id, status=SiteCrawlStatus.PAUSED
                    )
                await self._wait_while_paused()
                if self._should_stop:  # Check if stop was called while paused
                    break
                logger.info(f"Resuming crawl of {self.config.name} at depth {depth}")
            
            if not current_level or len(self.articles) >= self.config.limits.max_articles:
                break
            
            logger.info(f"Crawling depth {depth}: {len(current_level)} URLs")
            next_level = set()
            
            # Update current depth in progress
            if self.progress_manager and self.job_id:
                self.progress_manager.update_site_progress(
                    self.job_id, self.config.id, current_depth=depth
                )
            
            for url in current_level:
                # Check for control signals during URL processing
                control_action = self._check_control_signals()
                if control_action == "stop":
                    logger.info(f"Stopping crawl of {self.config.name} during URL processing")
                    break
                elif control_action == "pause":
                    logger.info(f"Pausing crawl of {self.config.name} during URL processing")
                    if self.progress_manager and self.job_id:
                        self.progress_manager.update_site_progress(
                            self.job_id, self.config.id, status=SiteCrawlStatus.PAUSED
                        )
                    await self._wait_while_paused()
                    if self._should_stop:  # Check if stop was called while paused
                        break
                    logger.info(f"Resuming crawl of {self.config.name} during URL processing")
                
                if len(self.articles) >= self.config.limits.max_articles:
                    break
                
                if self._normalize_url(url) in self.crawled_urls:
                    continue
                
                try:
                    article_data, links = await self._crawl_single_url(crawler, url, depth)
                    
                    if article_data:
                        self.articles.append(article_data)
                        logger.info(f"Added article: {article_data.get('title', 'Untitled')}")
                        
                        # Track successful crawl
                        if self.progress_manager and self.job_id:
                            self.progress_manager.mark_url_crawled(self.job_id, self.config.id, url, success=True)
                    
                    # Add new links for next level
                    if depth < self.config.limits.max_depth:
                        next_level.update(links)
                        # Track newly discovered URLs
                        if self.progress_manager and self.job_id and links:
                            self.progress_manager.add_discovered_urls(self.job_id, self.config.id, list(links))
                    
                    # Respect crawl delay
                    await asyncio.sleep(self.config.limits.delay_seconds)
                    
                except Exception as e:
                    logger.error(f"Error crawling {self._normalize_url(url)}: {str(e)}")
                    self.failed_urls.add(url)
                    
                    # Track failed crawl
                    if self.progress_manager and self.job_id:
                        self.progress_manager.mark_url_crawled(self.job_id, self.config.id, url, success=False)
            
            current_level = next_level
    
    async def _crawl_depth_first(self, crawler: AsyncWebCrawler):
        """Depth-first crawling strategy"""
        for start_url in self.config.start_urls:
            # Check for control signals
            control_action = self._check_control_signals()
            if control_action == "stop":
                logger.info(f"Stopping depth-first crawl of {self.config.name}")
                break
            elif control_action == "pause":
                logger.info(f"Pausing depth-first crawl of {self.config.name}")
                if self.progress_manager and self.job_id:
                    self.progress_manager.update_site_progress(
                        self.job_id, self.config.id, status=SiteCrawlStatus.PAUSED
                    )
                await self._wait_while_paused()
                if self._should_stop:
                    break
                logger.info(f"Resuming depth-first crawl of {self.config.name}")
            
            if len(self.articles) >= self.config.limits.max_articles:
                break
            await self._crawl_recursive(crawler, start_url, 0)
    
    async def _crawl_recursive(self, crawler: AsyncWebCrawler, url: str, depth: int):
        """Recursive crawling for depth-first strategy"""
        # Check for control signals
        control_action = self._check_control_signals()
        if control_action == "stop":
            return
        elif control_action == "pause":
            if self.progress_manager and self.job_id:
                self.progress_manager.update_site_progress(
                    self.job_id, self.config.id, status=SiteCrawlStatus.PAUSED
                )
            await self._wait_while_paused()
            if self._should_stop:
                return
        
        if (depth > self.config.limits.max_depth or 
            url in self.crawled_urls or 
            len(self.articles) >= self.config.limits.max_articles):
            return
        
        try:
            article_data, links = await self._crawl_single_url(crawler, url, depth)
            
            if article_data:
                self.articles.append(article_data)
                logger.info(f"Added article: {article_data.get('title', 'Untitled')}")
            
            # Crawl found links recursively
            for link in links:
                if len(self.articles) >= self.config.limits.max_articles:
                    break
                await asyncio.sleep(self.config.limits.delay_seconds)
                await self._crawl_recursive(crawler, link, depth + 1)
                
        except Exception as e:
            logger.error(f"Error crawling {self._normalize_url(url)}: {str(e)}")
            self.failed_urls.add(url)
    
    async def _crawl_from_sitemap(self, crawler: AsyncWebCrawler):
        """Crawl from sitemap.xml"""
        sitemap_urls = [
            urljoin(self.config.base_url, '/sitemap.xml'),
            urljoin(self.config.base_url, '/sitemap_index.xml'),
            urljoin(self.config.base_url, '/robots.txt')  # Check robots.txt for sitemap
        ]
        
        all_urls = set()
        
        for sitemap_url in sitemap_urls:
            try:
                urls = await self._extract_urls_from_sitemap(sitemap_url)
                all_urls.update(urls)
            except Exception as e:
                logger.debug(f"Could not access {sitemap_url}: {str(e)}")
        
        if not all_urls:
            logger.warning("No sitemap found, falling back to breadth-first crawling")
            await self._crawl_breadth_first(crawler)
            return
        
        logger.info(f"Found {len(all_urls)} URLs in sitemaps")
        
        # Crawl URLs from sitemap
        for url in list(all_urls)[:self.config.limits.max_articles]:
            try:
                article_data, _ = await self._crawl_single_url(crawler, url, 0)
                if article_data:
                    self.articles.append(article_data)
                await asyncio.sleep(self.config.limits.delay_seconds)
            except Exception as e:
                logger.error(f"Error crawling {self._normalize_url(url)}: {str(e)}")
    
    async def _crawl_url_list(self, crawler: AsyncWebCrawler):
        """Crawl from predefined URL list"""
        for url in self.config.start_urls[:self.config.limits.max_articles]:
            try:
                article_data, _ = await self._crawl_single_url(crawler, url, 0)
                if article_data:
                    self.articles.append(article_data)
                await asyncio.sleep(self.config.limits.delay_seconds)
            except Exception as e:
                logger.error(f"Error crawling {self._normalize_url(url)}: {str(e)}")
    
    async def _crawl_single_url(self, crawler: AsyncWebCrawler, url: str, depth: int) -> tuple:
        """Crawl a single URL and return article data and found links"""
        normalized_url = self._normalize_url(url)
        logger.info(f"Attempting to crawl URL: {normalized_url}")
        
        if not self._is_valid_url(url):
            logger.warning(f"URL validation failed for: {normalized_url}")
            return None, []

        if not await self._is_allowed_by_robots(url):
            logger.warning(f"URL blocked by robots.txt: {normalized_url}")
            return None, []

        logger.info(f"Crawling (depth {depth}): {normalized_url}")
        self.crawled_urls.add(normalized_url)
        
        try:
            # Use simplified crawling without LLM extraction to avoid API issues
            result = await crawler.arun(
                url=url,
                bypass_cache=False,
                delay_before_return_html=2.0,
                timeout=self.config.limits.timeout_seconds
            )
            
            if not result.success:
                logger.warning(f"Failed to crawl {normalized_url}: {result.error_message}")
                return None, []
            
            # Extract article content
            article_data = self._extract_article_content(result, url)
            
            # Extract links for further crawling
            links = self._extract_links(result.html, url) if result.html else []
            
            return article_data, links
            
        except Exception as e:
            logger.error(f"Error processing {normalized_url}: {str(e)}")
            return None, []
    
    def _is_english_content(self, url: str, soup: BeautifulSoup, content: str) -> bool:
        """Check if content is in English using multiple approaches"""
        target_language = self.config.language.lower()
        
        # Skip language filtering if not specifically set to English
        if target_language != "en":
            return True
            
        # 1. URL-based detection (fastest)
        if not self._is_english_url(url):
            logger.debug(f"Non-English URL detected: {url}")
            return False
            
        # 2. HTML lang attribute detection
        if not self._is_english_html_lang(soup):
            logger.debug(f"Non-English HTML lang attribute: {url}")
            return False
            
        # 3. Content-based detection (try advanced first, fall back to basic)
        if len(content.split()) >= 50:  # Use advanced detection for longer content
            if not self._is_english_content_advanced(content):
                logger.debug(f"Non-English content detected (advanced): {url}")
                return False
        else:
            # Use basic detection for shorter content
            if not self._is_english_content_basic(content):
                logger.debug(f"Non-English content detected (basic): {url}")
                return False
            
        return True
    
    def _is_english_url(self, url: str) -> bool:
        """Check URL for language indicators"""
        non_english_patterns = [
            r'/es[/-]',    # Spanish
            r'/fr[/-]',    # French  
            r'/de[/-]',    # German
            r'/it[/-]',    # Italian
            r'/pt[/-]',    # Portuguese
            r'/ja[/-]',    # Japanese
            r'/ko[/-]',    # Korean
            r'/zh[/-]',    # Chinese
            r'/ru[/-]',    # Russian
            r'/ar[/-]',    # Arabic
            r'/hi[/-]',    # Hindi
            r'/nl[/-]',    # Dutch
            r'/sv[/-]',    # Swedish
            r'/da[/-]',    # Danish
            r'/no[/-]',    # Norwegian
            r'/fi[/-]',    # Finnish
            r'/pl[/-]',    # Polish
            r'/th[/-]',    # Thai
            r'[?&]lang=(?!en)[a-z]{2}',     # Query param like ?lang=es
            r'[?&]language=(?!en)[a-z]{2}', # Query param like ?language=fr
            r'[?&]locale=(?!en)[a-z]{2}',   # Query param like ?locale=de
        ]
        
        for pattern in non_english_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return False
        return True
    
    def _is_english_html_lang(self, soup: BeautifulSoup) -> bool:
        """Check HTML lang attribute"""
        try:
            # Check <html lang="..."> attribute
            html_tag = soup.find('html')
            if html_tag and html_tag.get('lang'):
                lang = html_tag.get('lang').lower()
                # Accept English variants (en, en-us, en-gb, etc.)
                if not lang.startswith('en'):
                    return False
                    
            # Check meta language tags
            meta_lang = soup.find('meta', attrs={'http-equiv': 'Content-Language'})
            if meta_lang and meta_lang.get('content'):
                lang = meta_lang.get('content').lower()
                if not lang.startswith('en'):
                    return False
                    
            return True
            
        except Exception:
            # If we can't determine, assume English
            return True
    
    def _is_english_content_basic(self, content: str) -> bool:
        """Basic content-based English detection using common words"""
        if not content or len(content.split()) < 20:
            return True  # Too short to determine, assume English
            
        # Common English words that are rare in other languages
        english_indicators = [
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'this', 'that', 'these', 'those', 'you', 'your', 'we', 'our', 'they', 'their'
        ]
        
        # Count English indicator words
        words = content.lower().split()[:100]  # Check first 100 words
        english_count = sum(1 for word in words if word in english_indicators)
        
        # If less than 15% are common English words, likely not English
        english_ratio = english_count / len(words) if words else 0
        
        return english_ratio >= 0.15  # Threshold: at least 15% English indicators
    
    def _is_english_content_advanced(self, content: str) -> bool:
        """Advanced language detection using langdetect library"""
        try:
            from langdetect import detect
            if len(content.split()) < 20:
                return True  # Too short for accurate detection
            
            # Use first 500 words for detection (more reliable)
            sample_text = ' '.join(content.split()[:500])
            detected_lang = detect(sample_text)
            return detected_lang == 'en'
        except ImportError:
            # Fall back to basic detection if langdetect not available
            logger.debug("langdetect not available, using basic detection")
            return self._is_english_content_basic(content)
        except Exception as e:
            # If detection fails, assume English
            logger.debug(f"Language detection failed: {e}")
            return True
    
    def _extract_article_content(self, result, url: str) -> Optional[Dict[str, Any]]:
        """Extract article content using configured selectors"""
        try:
            if not result.html:
                return None
            
            soup = BeautifulSoup(result.html, 'html.parser')
            
            # Remove excluded elements
            for exclude_selector in self.config.selectors.exclude:
                for element in soup.select(exclude_selector):
                    element.decompose()
            
            # Extract title
            title = self._extract_with_selectors(soup, self.config.selectors.title)
            if not title:
                title = soup.find('title')
                title = title.get_text().strip() if title else "Untitled"
            
            # Extract main content - be more aggressive
            content = self._extract_with_selectors(soup, self.config.selectors.content)
            if not content:
                # Fallback to common content selectors
                content = self._extract_with_selectors(soup, 'article, .content, .post-content, main, .main, body')
            
            if not content:
                # Use any available text content
                content = result.cleaned_html or result.markdown or soup.get_text(separator=' ', strip=True)
            
            # Extract description
            description = self._extract_with_selectors(soup, self.config.selectors.description)
            if not description:
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                description = meta_desc.get('content', '') if meta_desc else ''
            
            # Simplified - no category filtering
            category = "General"
            
            # Clean and validate content
            content = self._clean_content(content)
            word_count = len(content.split()) if content else 0
            
            # Accept all content, even short content - let's be aggressive
            if word_count < 10:  # Only skip completely empty content
                logger.debug(f"Skipping empty content: {url} ({word_count} words)")
                return None
            
            # Check if content is in English (language filtering)
            if not self._is_english_content(url, soup, content):
                logger.debug(f"Skipping non-English content: {url}")
                return None
            
            return {
                "url": url,
                "title": title.strip() if title else "Untitled",
                "content": content,
                "description": description.strip() if description else "",
                "category": category or "General",
                "content_type": self.config.content_type.value,
                "site_name": self.config.name,
                "site_id": self.config.id,
                "crawled_at": datetime.now().isoformat(),
                "word_count": word_count,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap
            }
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            return None
    
    def _extract_with_selectors(self, soup: BeautifulSoup, selectors: Optional[str]) -> str:
        """Extract text using CSS selectors"""
        if not selectors:
            return ""
        
        # Handle multiple selectors separated by commas
        selector_list = [s.strip() for s in selectors.split(',')]
        
        for selector in selector_list:
            try:
                elements = soup.select(selector)
                if elements:
                    return ' '.join([elem.get_text(separator=' ', strip=True) for elem in elements])
            except Exception as e:
                logger.debug(f"Error with selector '{selector}': {str(e)}")
        
        return ""
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content"""
        if not content:
            return ""
        
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\n\s*\n', '\n\n', content)
        
        # Remove common noise
        content = re.sub(r'(Share|Tweet|Like|Follow us|Subscribe)', '', content, flags=re.IGNORECASE)
        
        return content.strip()
    
    def _infer_category_from_url(self, url: str) -> str:
        """Infer category from URL path"""
        try:
            path = urlparse(url).path.lower()
            
            # Check against configured categories
            for category in self.config.categories:
                if category.lower() in path:
                    return category
            
            # Common patterns
            if any(word in path for word in ['blog', 'news', 'article']):
                return "Blog"
            elif any(word in path for word in ['doc', 'guide', 'tutorial']):
                return "Documentation"
            elif any(word in path for word in ['faq', 'help', 'support']):
                return "Support"
            elif any(word in path for word in ['api', 'reference']):
                return "API"
            
            return "General"
        except Exception:
            return "General"
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL by removing query parameters and fragments for uniqueness checking"""
        try:
            # Handle edge cases
            if not url or not url.strip():
                return url
            
            parsed = urlparse(url)
            
            # If parsing fails completely, return original URL
            if not parsed.scheme and not parsed.netloc:
                return url
            
            # Remove query parameters and fragments, keep only scheme, netloc, and path
            normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            
            # Remove trailing slash for consistency, but only if there's a path
            if normalized.endswith('/') and parsed.path == '/':
                normalized = normalized[:-1]
            elif normalized.endswith('/') and len(parsed.path) > 1:
                normalized = normalized[:-1]
            
            return normalized
        except Exception:
            return url

    def _extract_links(self, html: str, base_url: str) -> List[str]:
        """Extract relevant links from HTML - comprehensive extraction from all link-containing elements"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            links = []
            seen_normalized = set()
            
            # Define all HTML elements and attributes that can contain links
            link_selectors = [
                # Standard anchor tags
                ('a', 'href'),
                # Area tags in image maps
                ('area', 'href'),
                # Link tags (stylesheets, etc.)
                ('link', 'href'),
                # Base tags
                ('base', 'href'),
                # Form actions
                ('form', 'action'),
                # Iframe sources
                ('iframe', 'src'),
                # Frame sources
                ('frame', 'src'),
                # Object data
                ('object', 'data'),
                # Embed sources
                ('embed', 'src'),
                # Image sources (for image links)
                ('img', 'src'),
                # Script sources (for potential navigation)
                ('script', 'src'),
                # Video sources
                ('video', 'src'),
                # Audio sources  
                ('audio', 'src'),
                # Source elements
                ('source', 'src'),
                # Track elements
                ('track', 'src'),
                # Meta refresh redirects
                ('meta', 'content'),
            ]
            
            for tag_name, attr_name in link_selectors:
                elements = soup.find_all(tag_name)
                
                for element in elements:
                    # Handle different attribute extraction scenarios
                    if attr_name == 'content' and tag_name == 'meta':
                        # Special handling for meta refresh
                        http_equiv = element.get('http-equiv', '').lower()
                        if http_equiv == 'refresh':
                            content = element.get('content', '')
                            # Extract URL from refresh content (format: "5;URL=http://example.com")
                            if 'url=' in content.lower():
                                url_part = content.split('url=', 1)[1].strip()
                                if url_part:
                                    self._process_extracted_url(url_part, base_url, links, seen_normalized)
                    else:
                        # Standard attribute extraction
                        url_value = element.get(attr_name)
                        if url_value:
                            self._process_extracted_url(url_value, base_url, links, seen_normalized)
                    
                    # Also check for onclick handlers and other JavaScript links
                    onclick = element.get('onclick', '')
                    if onclick:
                        # Extract URLs from onclick handlers (basic pattern matching)
                        url_patterns = re.findall(r'location\.href\s*=\s*[\'"]([^\'"]+)[\'"]', onclick)
                        url_patterns.extend(re.findall(r'window\.open\s*\(\s*[\'"]([^\'"]+)[\'"]', onclick))
                        url_patterns.extend(re.findall(r'window\.location\s*=\s*[\'"]([^\'"]+)[\'"]', onclick))
                        
                        for url_pattern in url_patterns:
                            self._process_extracted_url(url_pattern, base_url, links, seen_normalized)
            
            # Also extract URLs from text content using regex patterns
            # This catches URLs that might be in plain text or JavaScript
            text_content = soup.get_text()
            url_regex = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+|[^\s<>"\']+\.[a-z]{2,4}/[^\s<>"\']*'
            text_urls = re.findall(url_regex, text_content, re.IGNORECASE)
            
            for url in text_urls:
                if not url.startswith(('http://', 'https://')):
                    if url.startswith('www.'):
                        url = 'https://' + url
                    elif '.' in url and '/' in url:
                        url = 'https://' + url
                
                self._process_extracted_url(url, base_url, links, seen_normalized)
            
            logger.info(f"Extracted {len(links)} unique links from {base_url}")
            return links[:500]  # Increase limit significantly to capture more links
            
        except Exception as e:
            logger.error(f"Error extracting links: {str(e)}")
            return []
    
    def _process_extracted_url(self, url_value: str, base_url: str, links: List[str], seen_normalized: set):
        """Process and validate a single extracted URL"""
        try:
            # Clean the URL value
            url_value = url_value.strip()
            if not url_value:
                return
            
            # Create absolute URL
            full_url = urljoin(base_url, url_value)
            
            # Normalize URL for uniqueness checking
            normalized_url = self._normalize_url(full_url)
            
            # Check if URL should be included
            if (self._is_valid_url(full_url) and 
                normalized_url not in seen_normalized and 
                normalized_url not in {self._normalize_url(u) for u in self.crawled_urls}):
                links.append(normalized_url)
                seen_normalized.add(normalized_url)
                logger.debug(f"Added link: {normalized_url}")
                
        except Exception as e:
            logger.debug(f"Error processing URL '{url_value}': {str(e)}")
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL should be crawled based on patterns"""
        try:
            parsed = urlparse(url)
            logger.debug(f"Validating URL: {url}")
            
            # Basic validation first
            if not parsed.netloc:
                return False
            
            # Skip fragments and javascript early
            if any(skip in url.lower() for skip in [
                'javascript:', 'mailto:', 'tel:'
            ]):
                return False
            
            # Skip common non-content files
            if any(url.lower().endswith(ext) for ext in [
                '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.css', '.js',
                '.zip', '.rar', '.exe', '.dmg', '.mp3', '.mp4', '.avi', '.ico', '.xml'
            ]):
                return False
            
            # Check blocked patterns first (reject if blocked)
            if self.config.blocked_patterns:
                blocked = any(re.search(pattern, url, re.IGNORECASE) for pattern in self.config.blocked_patterns)
                if blocked:
                    logger.debug(f"URL {url} blocked by blocked patterns")
                    return False
            
            # Check allowed patterns - be more permissive
            if self.config.allowed_patterns:
                allowed = any(re.search(pattern, url, re.IGNORECASE) for pattern in self.config.allowed_patterns)
                logger.debug(f"Allowed patterns check for {url}: {allowed}, patterns: {self.config.allowed_patterns}")
                if not allowed:
                    logger.debug(f"URL {url} rejected by allowed patterns")
                    return False
            else:
                # If no allowed patterns specified, allow all URLs from the same domain
                base_domain = urlparse(self.config.base_url).netloc
                if parsed.netloc != base_domain and not parsed.netloc.endswith('.' + base_domain):
                    logger.debug(f"URL {url} rejected - different domain from base_url")
                    return False
            
            # Check for English content in URL (early filtering) - but be more lenient
            if self.config.language.lower() == "en" and not self._is_english_url(url):
                logger.debug(f"Skipping non-English URL: {url}")
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error validating URL {url}: {str(e)}")
            return False
    
    async def _is_allowed_by_robots(self, url: str) -> bool:
        """Check robots.txt permissions"""
        if not self.config.limits.respect_robots_txt:
            return True
        
        try:
            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            
            if robots_url not in self.robots_cache:
                rp = RobotFileParser()
                rp.set_url(robots_url)
                try:
                    rp.read()
                    self.robots_cache[robots_url] = rp
                except Exception:
                    # If robots.txt can't be read, assume allowed
                    return True
            
            rp = self.robots_cache[robots_url]
            return rp.can_fetch('*', url)
            
        except Exception:
            return True
    
    async def _extract_urls_from_sitemap(self, sitemap_url: str) -> Set[str]:
        """Extract URLs from sitemap.xml"""
        urls = set()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(sitemap_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Parse XML content
                        soup = BeautifulSoup(content, 'xml')
                        
                        # Extract URLs from sitemap
                        for loc in soup.find_all('loc'):
                            url = loc.get_text().strip()
                            if self._is_valid_url(url):
                                urls.add(url)
                        
                        # Handle sitemap index files
                        for sitemap in soup.find_all('sitemap'):
                            sitemap_loc = sitemap.find('loc')
                            if sitemap_loc:
                                sub_urls = await self._extract_urls_from_sitemap(sitemap_loc.get_text().strip())
                                urls.update(sub_urls)
        
        except Exception as e:
            logger.debug(f"Error reading sitemap {sitemap_url}: {str(e)}")
        
        return urls
    
    def save_articles(self, filename: str = None):
        """Save crawled articles to JSON file"""
        try:
            if not filename:
                filename = f"data/{self.config.name.lower().replace(' ', '_')}_articles.json"
            
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Sort articles alphabetically by URL for improved readability
            sorted_articles = sorted(self.articles, key=lambda x: x.get('url', ''))
            
            data = {
                'site_config': self.config.to_dict(),
                'articles': sorted_articles,
                'statistics': {
                    'total_articles': len(self.articles),
                    'crawled_urls': len(self.crawled_urls),
                    'failed_urls': len(self.failed_urls),
                    'crawl_completed_at': datetime.now().isoformat()
                }
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(self.articles)} articles to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving articles: {str(e)}")


class MultiSiteCrawler:
    """Orchestrates crawling of multiple sites with progress tracking"""
    
    def __init__(self, config_manager: SiteConfigManager = None, progress_manager: Optional[CrawlProgressManager] = None):
        self.config_manager = config_manager or SiteConfigManager()
        self.progress_manager = progress_manager
    
    async def crawl_all_active_sites(self) -> Dict[str, List[Dict[str, Any]]]:
        """Crawl all active sites"""
        active_sites = self.config_manager.get_active_sites()
        results = {}
        
        logger.info(f"Starting crawl of {len(active_sites)} active sites")
        
        for site_config in active_sites:
            try:
                logger.info(f"Crawling site: {site_config.name}")
                crawler = GenericWebCrawler(site_config)
                articles = await crawler.crawl_site()
                results[site_config.id] = articles
                
                # Save individual site results
                crawler.save_articles()
                
                # Update config with results
                self.config_manager.update_site(site_config.id, {
                    'total_articles': len(articles),
                    'last_crawled': datetime.now(),
                    'last_crawl_status': 'completed' if articles else 'no_content_found'
                })
                
            except Exception as e:
                logger.error(f"Error crawling {site_config.name}: {str(e)}")
                self.config_manager.update_site(site_config.id, {
                    'last_crawl_status': f'error: {str(e)}'
                })
                results[site_config.id] = []
        
        return results
    
    async def crawl_site_by_id(self, site_id: str, job_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Crawl a specific site by ID with optional progress tracking"""
        site_config = self.config_manager.get_site(site_id)
        if not site_config:
            raise ValueError(f"Site with ID {site_id} not found")
        
        crawler = GenericWebCrawler(site_config, self.progress_manager, job_id)
        articles = await crawler.crawl_site()
        
        # Save individual site results
        crawler.save_articles()
        
        # Update config
        self.config_manager.update_site(site_id, {
            'total_articles': len(articles),
            'last_crawled': datetime.now(),
            'last_crawl_status': 'completed' if articles else 'no_content_found'
        })
        
        # Save results
        crawler.save_articles()
        
        return articles


async def main():
    """Test the generic crawler"""
    # Initialize config manager
    config_manager = SiteConfigManager()
    
    # Get active sites
    active_sites = config_manager.get_active_sites()
    
    if not active_sites:
        logger.error("No active sites configured")
        return
    
    # Test crawling first active site
    site = active_sites[0]
    logger.info(f"Testing crawl of: {site.name}")
    
    crawler = GenericWebCrawler(site)
    articles = await crawler.crawl_site()
    
    logger.info(f"Crawled {len(articles)} articles")
    
    # Save results
    crawler.save_articles()


if __name__ == "__main__":
    import os
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Run the crawler
    asyncio.run(main()) 