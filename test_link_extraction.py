#!/usr/bin/env python3
"""
Test script for enhanced link extraction
"""

from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse

def test_extract_links():
    """Test the enhanced link extraction logic"""
    
    # Sample HTML with various link types
    test_html = """
    <html>
    <body>
        <!-- Standard anchor links -->
        <a href="/page1">Page 1</a>
        <a href="https://example.com/page2">Page 2</a>
        
        <!-- Image map areas -->
        <map>
            <area href="/area1" alt="Area 1">
            <area href="/area2" alt="Area 2">
        </map>
        
        <!-- Forms -->
        <form action="/submit">
            <input type="submit" value="Submit">
        </form>
        
        <!-- iframes -->
        <iframe src="/embedded-page"></iframe>
        
        <!-- Links in JavaScript -->
        <button onclick="location.href='/js-link1'">Click Me</button>
        <button onclick="window.open('/js-link2')">Open Window</button>
        
        <!-- Images -->
        <img src="/image1.jpg" alt="Image">
        
        <!-- Meta refresh -->
        <meta http-equiv="refresh" content="5;URL=/redirect-page">
        
        <!-- Plain text URLs -->
        <p>Visit https://example.com/text-url for more info</p>
        <p>Also check www.example.com/another-url</p>
        
        <!-- Link elements -->
        <link rel="stylesheet" href="/style.css">
        <link rel="alternate" href="/feed.xml">
    </body>
    </html>
    """
    
    base_url = "https://example.com"
    soup = BeautifulSoup(test_html, 'html.parser')
    links = []
    seen_normalized = set()
    
    # Define all HTML elements and attributes that can contain links
    link_selectors = [
        ('a', 'href'),
        ('area', 'href'),
        ('link', 'href'),
        ('base', 'href'),
        ('form', 'action'),
        ('iframe', 'src'),
        ('frame', 'src'),
        ('object', 'data'),
        ('embed', 'src'),
        ('img', 'src'),
        ('script', 'src'),
        ('video', 'src'),
        ('audio', 'src'),
        ('source', 'src'),
        ('track', 'src'),
        ('meta', 'content'),
    ]
    
    def normalize_url(url):
        """Simple URL normalization"""
        try:
            parsed = urlparse(url)
            return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        except:
            return url
    
    def process_extracted_url(url_value, base_url, links, seen_normalized):
        """Process and validate a single extracted URL"""
        try:
            url_value = url_value.strip()
            if not url_value:
                return
            
            full_url = urljoin(base_url, url_value)
            normalized_url = normalize_url(full_url)
            
            if normalized_url not in seen_normalized:
                links.append(normalized_url)
                seen_normalized.add(normalized_url)
                print(f"Added link: {normalized_url}")
                
        except Exception as e:
            print(f"Error processing URL '{url_value}': {str(e)}")
    
    # Extract links from various elements
    for tag_name, attr_name in link_selectors:
        elements = soup.find_all(tag_name)
        
        for element in elements:
            # Handle different attribute extraction scenarios
            if attr_name == 'content' and tag_name == 'meta':
                # Special handling for meta refresh
                http_equiv = element.get('http-equiv', '').lower()
                if http_equiv == 'refresh':
                    content = element.get('content', '')
                    if 'url=' in content.lower():
                        url_part = content.split('url=', 1)[1].strip()
                        if url_part:
                            process_extracted_url(url_part, base_url, links, seen_normalized)
            else:
                # Standard attribute extraction
                url_value = element.get(attr_name)
                if url_value:
                    process_extracted_url(url_value, base_url, links, seen_normalized)
            
            # Also check for onclick handlers
            onclick = element.get('onclick', '')
            if onclick:
                url_patterns = re.findall(r'location\.href\s*=\s*[\'"]([^\'"]+)[\'"]', onclick)
                url_patterns.extend(re.findall(r'window\.open\s*\(\s*[\'"]([^\'"]+)[\'"]', onclick))
                url_patterns.extend(re.findall(r'window\.location\s*=\s*[\'"]([^\'"]+)[\'"]', onclick))
                
                for url_pattern in url_patterns:
                    process_extracted_url(url_pattern, base_url, links, seen_normalized)
    
    # Extract URLs from text content using regex
    text_content = soup.get_text()
    url_regex = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+|[^\s<>"\']+\.[a-z]{2,4}/[^\s<>"\']*'
    text_urls = re.findall(url_regex, text_content, re.IGNORECASE)
    
    for url in text_urls:
        if not url.startswith(('http://', 'https://')):
            if url.startswith('www.'):
                url = 'https://' + url
            elif '.' in url and '/' in url:
                url = 'https://' + url
        
        process_extracted_url(url, base_url, links, seen_normalized)
    
    print(f"\nTotal unique links extracted: {len(links)}")
    print("\nAll links:")
    for link in sorted(links):
        print(f"  - {link}")
    
    return links

if __name__ == "__main__":
    test_extract_links()