#!/usr/bin/env python3
"""
Multi-Site RAG System - Main Entry Point
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional
import argparse
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from site_config import SiteConfig, SiteConfigManager, create_site_from_template
from generic_crawler import MultiSiteCrawler
from multi_site_vector_store import MultiSiteVectorStore, MultiSiteVectorStoreManager
from site_management_api import create_site_management_api
import uvicorn
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/multi_site_rag.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MultiSiteRAGSystem:
    """Main orchestrator for the multi-site RAG system"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = config_dir
        # SiteConfigManager expects a file path, not a directory
        config_file = f"{config_dir}/sites_config.json" if config_dir != "configs" else "data/sites_config.json"
        self.site_config_manager = SiteConfigManager(config_file)
        self.crawler = MultiSiteCrawler(self.site_config_manager)
        self.vector_store = MultiSiteVectorStore()
        self.vector_store_manager = MultiSiteVectorStoreManager(
            self.vector_store, 
            self.site_config_manager
        )
        
        # Ensure required directories exist
        self._ensure_directories()
        
        logger.info("Multi-Site RAG System initialized")
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        dirs = [
            self.config_dir,
            "data",
            "logs",
            "chroma_db",
            "backups"
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def add_site_from_template(self, 
                              template_name: str,
                              name: str,
                              base_url: str,
                              start_urls: List[str],
                              description: str = "") -> str:
        """Add a new site using a template"""
        try:
            site_config = create_site_from_template(
                template_name,
                name=name,
                description=description,
                base_url=base_url,
                start_urls=start_urls
            )
            
            site_id = self.site_config_manager.add_site(site_config)
            logger.info(f"Added site from template '{template_name}': {name}")
            return site_id
            
        except Exception as e:
            logger.error(f"Failed to add site from template: {str(e)}")
            raise
    
    def add_hubspot_site(self) -> str:
        """Add HubSpot Knowledge Base as a site"""
        return self.add_site_from_template(
            template_name="documentation",
            name="HubSpot Knowledge Base",
            base_url="https://knowledge.hubspot.com",
            start_urls=[
                "https://knowledge.hubspot.com/"
            ],
            description="HubSpot's comprehensive knowledge base with articles, guides, and documentation"
        )
    
    async def crawl_all_sites(self) -> dict:
        """Crawl all active sites"""
        logger.info("Starting crawl of all active sites")
        
        try:
            results = await self.crawler.crawl_all_active_sites()
            
            # Process results
            total_articles = sum(len(articles) for articles in results.values())
            logger.info(f"Crawl completed. Total articles: {total_articles}")
            
            return {
                'success': True,
                'total_articles': total_articles,
                'sites_crawled': len(results),
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error crawling sites: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def crawl_site(self, site_id: str) -> dict:
        """Crawl a specific site"""
        logger.info(f"Starting crawl of site: {site_id}")
        
        try:
            articles = await self.crawler.crawl_site_by_id(site_id)
            
            logger.info(f"Crawl completed. Articles found: {len(articles)}")
            
            return {
                'success': True,
                'site_id': site_id,
                'articles_count': len(articles),
                'articles': articles
            }
            
        except Exception as e:
            logger.error(f"Error crawling site {site_id}: {str(e)}")
            return {
                'success': False,
                'site_id': site_id,
                'error': str(e)
            }
    
    def sync_vector_store(self):
        """Sync all sites with the vector store"""
        logger.info("Syncing vector store with all sites")
        
        try:
            self.vector_store_manager.sync_all_sites()
            logger.info("Vector store sync completed")
            
        except Exception as e:
            logger.error(f"Error syncing vector store: {str(e)}")
            raise
    
    def search(self, 
               query: str, 
               site_ids: Optional[List[str]] = None,
               n_results: int = 10) -> List[dict]:
        """Search across sites"""
        logger.info(f"Searching for: {query}")
        
        try:
            results = self.vector_store.similarity_search(
                query=query,
                site_ids=site_ids,
                n_results=n_results
            )
            
            logger.info(f"Search completed. Results found: {len(results)}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            return []
    
    def get_system_stats(self) -> dict:
        """Get comprehensive system statistics"""
        try:
            # Get site stats
            sites = self.site_config_manager.get_all_sites()
            active_sites = self.site_config_manager.get_active_sites()
            
            # Get vector store stats
            vector_stats = self.vector_store_manager.get_comprehensive_stats()
            
            stats = {
                'timestamp': datetime.now().isoformat(),
                'sites': {
                    'total': len(sites),
                    'active': len(active_sites),
                    'by_type': {}
                },
                'vector_store': vector_stats,
                'system': {
                    'config_dir': self.config_dir,
                    'data_dir': 'data',
                    'vector_db_dir': 'chroma_db'
                }
            }
            
            # Count sites by type
            for site in sites:
                content_type = site.content_type.value
                stats['sites']['by_type'][content_type] = stats['sites']['by_type'].get(content_type, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {'error': str(e)}
    
    def list_sites(self) -> List[dict]:
        """List all configured sites"""
        sites = self.site_config_manager.get_all_sites()
        return [
            {
                'id': site.id,
                'name': site.name,
                'base_url': site.base_url,
                'content_type': site.content_type.value,
                'is_active': site.is_active,
                'total_articles': site.total_articles,
                'last_crawl_status': site.last_crawl_status,
                'last_crawled': site.last_crawled.isoformat() if site.last_crawled else None
            }
            for site in sites
        ]
    
    def backup_system(self, backup_dir: str = "backups"):
        """Backup the entire system"""
        backup_path = Path(backup_dir) / f"system_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Backup configurations
            self.site_config_manager.backup_configs(str(backup_path / "configs"))
            
            # Backup vector store data
            for site in self.site_config_manager.get_all_sites():
                self.vector_store.backup_site_data(
                    site.id,
                    str(backup_path / f"vector_store_{site.id}.json")
                )
            
            logger.info(f"System backup completed: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Error backing up system: {str(e)}")
            raise


def start_api_server(args):
    """Start the API server (non-async)"""
    # Load environment variables
    load_dotenv()
    
    print(f"Starting API server on {args.host}:{args.port}")
    app = create_site_management_api()
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)

async def main():
    """Main entry point with CLI interface"""
    parser = argparse.ArgumentParser(description="Multi-Site RAG System")
    parser.add_argument("command", choices=[
        "setup", "add-site", "crawl", "crawl-all", "sync", "search", 
        "stats", "list", "api", "backup"
    ], help="Command to execute")
    
    # Site management
    parser.add_argument("--template", choices=["documentation", "blog", "support"], 
                       help="Template to use for new site")
    parser.add_argument("--name", help="Site name")
    parser.add_argument("--url", help="Base URL")
    parser.add_argument("--start-urls", nargs="+", help="Start URLs")
    parser.add_argument("--description", help="Site description")
    parser.add_argument("--site-id", help="Site ID for operations")
    
    # Crawling
    parser.add_argument("--site-ids", nargs="+", help="Site IDs to crawl")
    
    # Search
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--results", type=int, default=10, help="Number of results")
    
    # API
    parser.add_argument("--host", default="0.0.0.0", help="API host")
    parser.add_argument("--port", type=int, default=8001, help="API port")
    parser.add_argument("--reload", action="store_true", help="Enable API reload")
    
    # General
    parser.add_argument("--config-dir", default="configs", help="Configuration directory")
    parser.add_argument("--backup-dir", default="backups", help="Backup directory")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Initialize system
    system = MultiSiteRAGSystem(args.config_dir)
    
    if args.command == "setup":
        # Initial setup - add HubSpot site
        print("Setting up Multi-Site RAG System...")
        print("Adding HubSpot Knowledge Base as default site...")
        
        hubspot_id = system.add_hubspot_site()
        print(f"HubSpot site added with ID: {hubspot_id}")
        
        print("\nTo start using the system:")
        print("1. python multi_site_main.py api  # Start the API server")
        print("2. Open multi_site_web_interface.html in your browser")
        print("3. Or use: python multi_site_main.py crawl-all  # Crawl all sites")
        
    elif args.command == "add-site":
        if not all([args.template, args.name, args.url, args.start_urls]):
            print("Error: --template, --name, --url, and --start-urls are required")
            return
        
        site_id = system.add_site_from_template(
            args.template,
            args.name,
            args.url,
            args.start_urls,
            args.description or ""
        )
        print(f"Site added with ID: {site_id}")
        
    elif args.command == "crawl":
        if not args.site_id:
            print("Error: --site-id is required")
            return
        
        result = await system.crawl_site(args.site_id)
        if result['success']:
            print(f"Crawl completed. Articles found: {result['articles_count']}")
        else:
            print(f"Crawl failed: {result['error']}")
    
    elif args.command == "crawl-all":
        result = await system.crawl_all_sites()
        if result['success']:
            print(f"Crawl completed. Total articles: {result['total_articles']}")
            print(f"Sites crawled: {result['sites_crawled']}")
        else:
            print(f"Crawl failed: {result['error']}")
    
    elif args.command == "sync":
        print("Syncing vector store...")
        system.sync_vector_store()
        print("Sync completed")
    
    elif args.command == "search":
        if not args.query:
            print("Error: --query is required")
            return
        
        results = system.search(
            args.query,
            site_ids=args.site_ids,
            n_results=args.results
        )
        
        print(f"Search results for '{args.query}':")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['metadata'].get('title', 'Untitled')}")
            print(f"   Site: {result['metadata'].get('site_name', 'Unknown')}")
            print(f"   URL: {result['metadata'].get('url', 'Unknown')}")
            print(f"   Score: {result['score']:.3f}")
            print(f"   Content: {result['content'][:200]}...")
    
    elif args.command == "stats":
        stats = system.get_system_stats()
        print("System Statistics:")
        print(f"Total Sites: {stats['sites']['total']}")
        print(f"Active Sites: {stats['sites']['active']}")
        print(f"Total Documents: {stats['vector_store']['total_documents']}")
        print(f"Sites by Type: {stats['sites']['by_type']}")
    
    elif args.command == "list":
        sites = system.list_sites()
        print("Configured Sites:")
        for site in sites:
            print(f"- {site['name']} ({site['id']})")
            print(f"  URL: {site['base_url']}")
            print(f"  Type: {site['content_type']}")
            print(f"  Status: {'Active' if site['is_active'] else 'Inactive'}")
            print(f"  Articles: {site['total_articles']}")
            print(f"  Last Crawl: {site['last_crawl_status']}")
            print()
    
    elif args.command == "api":
        # API server needs to run outside the async context
        return "api"
    
    elif args.command == "backup":
        backup_path = system.backup_system(args.backup_dir)
        print(f"System backup created: {backup_path}")


if __name__ == "__main__":
    # Handle API command separately due to asyncio event loop conflicts
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        # Parse args and start API server directly
        parser = argparse.ArgumentParser(description="Multi-Site RAG System")
        parser.add_argument("command")
        parser.add_argument("--host", default="0.0.0.0", help="API host")
        parser.add_argument("--port", type=int, default=8001, help="API port")
        parser.add_argument("--reload", action="store_true", help="Enable API reload")
        args = parser.parse_args()
        start_api_server(args)
    else:
        # Run other commands in async context
        result = asyncio.run(main())
        if result == "api":
            # This shouldn't happen with the new logic, but just in case
            print("API command should be handled directly") 