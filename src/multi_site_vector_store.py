#!/usr/bin/env python3
"""
Multi-Site Vector Store for Multi-Site RAG System
"""

import json
import logging
import os
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import hashlib
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np
from sentence_transformers import SentenceTransformer

from site_config import SiteConfig, SiteConfigManager

logger = logging.getLogger(__name__)


class MultiSiteVectorStore:
    """Vector store that manages content from multiple sites"""
    
    def __init__(self, 
                 persist_directory: str = "chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 collection_prefix: str = "site_"):
        """
        Initialize the multi-site vector store
        
        Args:
            persist_directory: Directory to persist the vector database
            embedding_model: Sentence transformer model for embeddings
            collection_prefix: Prefix for site-specific collections
        """
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        self.collection_prefix = collection_prefix
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        # Initialize sentence transformer for local embeddings
        self.sentence_transformer = SentenceTransformer(embedding_model)
        
        # Cache for collections
        self.collections: Dict[str, chromadb.Collection] = {}
        
        # Global collection for cross-site search
        self.global_collection_name = "global_multi_site"
        
        logger.info(f"Initialized MultiSiteVectorStore with embedding model: {embedding_model}")
    
    def get_site_collection_name(self, site_id: str) -> str:
        """Get the collection name for a specific site"""
        return f"{self.collection_prefix}{site_id}"
    
    def get_or_create_site_collection(self, site_config: SiteConfig) -> chromadb.Collection:
        """Get or create a collection for a specific site"""
        collection_name = self.get_site_collection_name(site_config.id)
        
        if collection_name not in self.collections:
            try:
                # Try to get existing collection
                collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info(f"Loaded existing collection for site: {site_config.name}")
            except Exception:
                # Create new collection
                metadata = {
                    "site_id": site_config.id,
                    "site_name": site_config.name,
                    "site_url": site_config.base_url,
                    "content_type": site_config.content_type.value,
                    "language": site_config.language,
                    "created_at": datetime.now().isoformat()
                }
                
                collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function,
                    metadata=metadata
                )
                logger.info(f"Created new collection for site: {site_config.name}")
            
            self.collections[collection_name] = collection
        
        return self.collections[collection_name]
    
    def get_or_create_global_collection(self) -> chromadb.Collection:
        """Get or create the global collection for cross-site search"""
        if self.global_collection_name not in self.collections:
            try:
                # Try to get existing collection
                collection = self.client.get_collection(
                    name=self.global_collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info("Loaded existing global collection")
            except Exception:
                # Create new collection
                metadata = {
                    "type": "global_multi_site",
                    "description": "Global collection for cross-site search",
                    "created_at": datetime.now().isoformat()
                }
                
                collection = self.client.create_collection(
                    name=self.global_collection_name,
                    embedding_function=self.embedding_function,
                    metadata=metadata
                )
                logger.info("Created new global collection")
            
            self.collections[self.global_collection_name] = collection
        
        return self.collections[self.global_collection_name]
    
    def add_site_content(self, site_config: SiteConfig, articles: List[Dict[str, Any]]):
        """Add content from a site to both site-specific and global collections"""
        if not articles:
            logger.warning(f"No articles to add for site: {site_config.name}")
            return
        
        logger.info(f"Adding {len(articles)} articles for site: {site_config.name}")
        
        # Get collections
        site_collection = self.get_or_create_site_collection(site_config)
        global_collection = self.get_or_create_global_collection()
        
        # Process articles in batches
        batch_size = 100
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            self._add_batch_to_collections(site_config, batch, site_collection, global_collection)
        
        logger.info(f"Successfully added {len(articles)} articles for site: {site_config.name}")
    
    def _add_batch_to_collections(self, 
                                  site_config: SiteConfig,
                                  articles: List[Dict[str, Any]],
                                  site_collection: chromadb.Collection,
                                  global_collection: chromadb.Collection):
        """Add a batch of articles to both collections"""
        
        documents = []
        metadatas = []
        ids = []
        
        for article in articles:
            # Create chunks from the article
            chunks = self._create_chunks(article, site_config)
            
            for chunk_idx, chunk in enumerate(chunks):
                # Create unique ID for this chunk
                chunk_id = self._create_chunk_id(article['url'], chunk_idx)
                ids.append(chunk_id)
                
                # Add chunk text
                documents.append(chunk['text'])
                
                # Create metadata
                metadata = {
                    "url": article['url'],
                    "title": article['title'],
                    "category": article.get('category', 'General'),
                    "site_id": site_config.id,
                    "site_name": site_config.name,
                    "site_url": site_config.base_url,
                    "content_type": article.get('content_type', site_config.content_type.value),
                    "language": site_config.language,
                    "crawled_at": article.get('crawled_at', datetime.now().isoformat()),
                    "chunk_index": chunk_idx,
                    "chunk_count": len(chunks),
                    "word_count": len(chunk['text'].split()),
                    "description": article.get('description', ''),
                    "source_type": "web_crawl"
                }
                
                # Add any additional metadata from the article
                for key, value in article.items():
                    if key not in metadata and isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                
                metadatas.append(metadata)
        
        try:
            # Add to site-specific collection
            site_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            # Add to global collection with site prefix in ID
            global_ids = [f"{site_config.id}_{id_}" for id_ in ids]
            global_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=global_ids
            )
            
            logger.debug(f"Added batch of {len(documents)} chunks to collections")
            
        except Exception as e:
            logger.error(f"Error adding batch to collections: {str(e)}")
            raise
    
    def _create_chunks(self, article: Dict[str, Any], site_config: SiteConfig) -> List[Dict[str, Any]]:
        """Create chunks from an article based on site configuration"""
        content = article.get('content', '')
        if not content:
            return []
        
        chunk_size = site_config.chunk_size
        chunk_overlap = site_config.chunk_overlap
        
        # Simple text chunking by words
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if chunk_text.strip():
                chunks.append({
                    'text': chunk_text,
                    'start_word': i,
                    'end_word': min(i + chunk_size, len(words))
                })
        
        return chunks
    
    def _create_chunk_id(self, url: str, chunk_index: int) -> str:
        """Create a unique ID for a chunk"""
        # Create a hash of the URL and chunk index
        content = f"{url}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def query_site(self, 
                   site_id: str,
                   query: str,
                   n_results: int = 10,
                   where: Optional[Dict[str, Any]] = None,
                   include_metadata: bool = True) -> List[Dict[str, Any]]:
        """Query a specific site's collection"""
        
        collection_name = self.get_site_collection_name(site_id)
        
        if collection_name not in self.collections:
            # Try to load the collection
            try:
                collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                self.collections[collection_name] = collection
            except Exception:
                logger.warning(f"No collection found for site: {site_id}")
                return []
        
        collection = self.collections[collection_name]
        
        try:
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                include=['documents', 'metadatas', 'distances'] if include_metadata else ['documents', 'distances']
            )
            
            return self._format_query_results(results)
            
        except Exception as e:
            logger.error(f"Error querying site {site_id}: {str(e)}")
            return []
    
    def query_multiple_sites(self,
                            site_ids: List[str],
                            query: str,
                            n_results: int = 10,
                            where: Optional[Dict[str, Any]] = None,
                            include_metadata: bool = True) -> List[Dict[str, Any]]:
        """Query multiple sites and merge results"""
        
        all_results = []
        
        for site_id in site_ids:
            site_results = self.query_site(
                site_id=site_id,
                query=query,
                n_results=n_results,
                where=where,
                include_metadata=include_metadata
            )
            all_results.extend(site_results)
        
        # Sort by score (distance) and return top results
        all_results.sort(key=lambda x: x.get('score', float('inf')))
        return all_results[:n_results]
    
    def query_global(self,
                     query: str,
                     n_results: int = 10,
                     where: Optional[Dict[str, Any]] = None,
                     site_filter: Optional[List[str]] = None,
                     include_metadata: bool = True) -> List[Dict[str, Any]]:
        """Query across all sites using the global collection"""
        
        global_collection = self.get_or_create_global_collection()
        
        # Add site filter to where clause if specified
        if site_filter:
            if where is None:
                where = {}
            where["site_id"] = {"$in": site_filter}
        
        try:
            results = global_collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                include=['documents', 'metadatas', 'distances'] if include_metadata else ['documents', 'distances']
            )
            
            return self._format_query_results(results)
            
        except Exception as e:
            logger.error(f"Error querying global collection: {str(e)}")
            return []
    
    def _format_query_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format ChromaDB query results into a more usable format"""
        formatted = []
        
        if not results['documents'] or not results['documents'][0]:
            return formatted
        
        documents = results['documents'][0]
        metadatas = results.get('metadatas', [None])[0] or [{}] * len(documents)
        distances = results.get('distances', [None])[0] or [0.0] * len(documents)
        
        for i, document in enumerate(documents):
            result = {
                'content': document,
                'score': float(distances[i]) if distances[i] is not None else 0.0,
                'metadata': metadatas[i] if i < len(metadatas) else {}
            }
            formatted.append(result)
        
        return formatted
    
    def get_site_stats(self, site_id: str) -> Dict[str, Any]:
        """Get statistics for a specific site"""
        collection_name = self.get_site_collection_name(site_id)
        
        try:
            if collection_name not in self.collections:
                collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                self.collections[collection_name] = collection
            else:
                collection = self.collections[collection_name]
            
            count = collection.count()
            metadata = collection.metadata or {}
            
            return {
                'site_id': site_id,
                'collection_name': collection_name,
                'document_count': count,
                'collection_metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error getting stats for site {site_id}: {str(e)}")
            return {
                'site_id': site_id,
                'collection_name': collection_name,
                'document_count': 0,
                'error': str(e)
            }
    
    def get_all_site_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all sites"""
        collections = self.client.list_collections()
        site_stats = []
        
        for collection_info in collections:
            collection_name = collection_info.name
            
            # Skip global collection
            if collection_name == self.global_collection_name:
                continue
            
            # Extract site ID from collection name
            if collection_name.startswith(self.collection_prefix):
                site_id = collection_name[len(self.collection_prefix):]
                stats = self.get_site_stats(site_id)
                site_stats.append(stats)
        
        return site_stats
    
    def delete_site_content(self, site_id: str):
        """Delete all content for a specific site"""
        collection_name = self.get_site_collection_name(site_id)
        
        try:
            # Delete site-specific collection
            self.client.delete_collection(name=collection_name)
            if collection_name in self.collections:
                del self.collections[collection_name]
            
            # Remove from global collection
            global_collection = self.get_or_create_global_collection()
            
            # Get all documents for this site
            results = global_collection.get(
                where={"site_id": site_id},
                include=['ids']
            )
            
            if results['ids']:
                global_collection.delete(ids=results['ids'])
            
            logger.info(f"Deleted all content for site: {site_id}")
            
        except Exception as e:
            logger.error(f"Error deleting site content for {site_id}: {str(e)}")
            raise
    
    def update_site_content(self, site_config: SiteConfig, articles: List[Dict[str, Any]]):
        """Update content for a site (delete old, add new)"""
        logger.info(f"Updating content for site: {site_config.name}")
        
        # Delete existing content
        self.delete_site_content(site_config.id)
        
        # Add new content
        self.add_site_content(site_config, articles)
        
        logger.info(f"Successfully updated content for site: {site_config.name}")
    
    def similarity_search(self, 
                         query: str,
                         site_ids: Optional[List[str]] = None,
                         n_results: int = 10,
                         score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Perform similarity search across specified sites or all sites"""
        
        if site_ids:
            # Query specific sites
            results = self.query_multiple_sites(
                site_ids=site_ids,
                query=query,
                n_results=n_results
            )
        else:
            # Query all sites using global collection
            results = self.query_global(
                query=query,
                n_results=n_results
            )
        
        # Filter by score threshold (lower distance = higher similarity)
        filtered_results = [
            result for result in results 
            if result.get('score', float('inf')) <= (1.0 - score_threshold)
        ]
        
        return filtered_results
    
    def get_similar_content(self,
                           content: str,
                           site_id: Optional[str] = None,
                           n_results: int = 5,
                           exclude_url: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find similar content to the given content"""
        
        where_clause = {}
        if exclude_url:
            where_clause["url"] = {"$ne": exclude_url}
        
        if site_id:
            results = self.query_site(
                site_id=site_id,
                query=content,
                n_results=n_results,
                where=where_clause
            )
        else:
            results = self.query_global(
                query=content,
                n_results=n_results,
                where=where_clause
            )
        
        return results
    
    def get_categories(self, site_id: Optional[str] = None) -> List[str]:
        """Get all categories available in the store"""
        
        if site_id:
            collection_name = self.get_site_collection_name(site_id)
            if collection_name in self.collections:
                collection = self.collections[collection_name]
            else:
                try:
                    collection = self.client.get_collection(
                        name=collection_name,
                        embedding_function=self.embedding_function
                    )
                except Exception:
                    return []
        else:
            collection = self.get_or_create_global_collection()
        
        try:
            # Get all documents to extract categories
            results = collection.get(
                include=['metadatas']
            )
            
            categories = set()
            for metadata in results.get('metadatas', []):
                if metadata and 'category' in metadata:
                    categories.add(metadata['category'])
            
            return sorted(list(categories))
            
        except Exception as e:
            logger.error(f"Error getting categories: {str(e)}")
            return []
    
    def backup_site_data(self, site_id: str, backup_path: str):
        """Backup site data to a JSON file"""
        collection_name = self.get_site_collection_name(site_id)
        
        try:
            if collection_name not in self.collections:
                collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
            else:
                collection = self.collections[collection_name]
            
            # Get all data
            results = collection.get(
                include=['documents', 'metadatas', 'ids']
            )
            
            backup_data = {
                'site_id': site_id,
                'collection_name': collection_name,
                'backup_date': datetime.now().isoformat(),
                'document_count': len(results.get('documents', [])),
                'data': results
            }
            
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Backed up site {site_id} to {backup_path}")
            
        except Exception as e:
            logger.error(f"Error backing up site {site_id}: {str(e)}")
            raise
    
    def restore_site_data(self, backup_path: str):
        """Restore site data from a backup file"""
        try:
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            site_id = backup_data['site_id']
            collection_name = backup_data['collection_name']
            data = backup_data['data']
            
            # Delete existing collection if it exists
            try:
                self.client.delete_collection(name=collection_name)
            except Exception:
                pass
            
            # Create new collection
            collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            
            # Add data
            if data.get('documents'):
                collection.add(
                    documents=data['documents'],
                    metadatas=data.get('metadatas', []),
                    ids=data['ids']
                )
            
            logger.info(f"Restored site {site_id} from {backup_path}")
            
        except Exception as e:
            logger.error(f"Error restoring from {backup_path}: {str(e)}")
            raise


class MultiSiteVectorStoreManager:
    """Manager for multi-site vector store operations"""
    
    def __init__(self, 
                 vector_store: MultiSiteVectorStore = None,
                 site_config_manager: SiteConfigManager = None):
        self.vector_store = vector_store or MultiSiteVectorStore()
        self.site_config_manager = site_config_manager or SiteConfigManager()
    
    def sync_all_sites(self):
        """Sync all active sites with the vector store"""
        active_sites = self.site_config_manager.get_active_sites()
        
        for site_config in active_sites:
            try:
                self.sync_site(site_config.id)
            except Exception as e:
                logger.error(f"Error syncing site {site_config.name}: {str(e)}")
    
    def sync_site(self, site_id: str):
        """Sync a specific site with the vector store"""
        site_config = self.site_config_manager.get_site(site_id)
        if not site_config:
            raise ValueError(f"Site {site_id} not found")
        
        # Load articles for this site
        articles_file = f"data/{site_config.name.lower().replace(' ', '_')}_articles.json"
        
        if os.path.exists(articles_file):
            try:
                with open(articles_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                articles = data.get('articles', [])
                if articles:
                    # Update vector store
                    self.vector_store.update_site_content(site_config, articles)
                    logger.info(f"Synced {len(articles)} articles for site: {site_config.name}")
                else:
                    logger.warning(f"No articles found for site: {site_config.name}")
                    
            except Exception as e:
                logger.error(f"Error loading articles for site {site_config.name}: {str(e)}")
                raise
        else:
            logger.warning(f"No articles file found for site: {site_config.name}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all sites"""
        site_stats = self.vector_store.get_all_site_stats()
        
        total_documents = sum(stats.get('document_count', 0) for stats in site_stats)
        
        return {
            'total_sites': len(site_stats),
            'total_documents': total_documents,
            'sites': site_stats,
            'timestamp': datetime.now().isoformat()
        }


def main():
    """Test the multi-site vector store"""
    # Initialize components
    vector_store = MultiSiteVectorStore()
    site_config_manager = SiteConfigManager()
    
    # Get active sites
    active_sites = site_config_manager.get_active_sites()
    
    if not active_sites:
        logger.error("No active sites configured")
        return
    
    # Test with first site
    site = active_sites[0]
    logger.info(f"Testing with site: {site.name}")
    
    # Create test articles
    test_articles = [
        {
            'url': f'{site.base_url}/test1',
            'title': 'Test Article 1',
            'content': 'This is a test article about machine learning and AI.',
            'category': 'Technology',
            'crawled_at': datetime.now().isoformat()
        },
        {
            'url': f'{site.base_url}/test2',
            'title': 'Test Article 2',
            'content': 'This is another test article about web development.',
            'category': 'Programming',
            'crawled_at': datetime.now().isoformat()
        }
    ]
    
    # Add content
    vector_store.add_site_content(site, test_articles)
    
    # Test queries
    results = vector_store.query_site(site.id, "machine learning", n_results=5)
    logger.info(f"Query results: {len(results)}")
    
    for result in results:
        logger.info(f"Score: {result['score']:.3f}, Content: {result['content'][:100]}...")
    
    # Test global query
    global_results = vector_store.query_global("web development", n_results=5)
    logger.info(f"Global query results: {len(global_results)}")
    
    # Get stats
    stats = vector_store.get_site_stats(site.id)
    logger.info(f"Site stats: {stats}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the test
    main() 