import re
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import spacy

logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """Enhanced chunk metadata"""
    section_type: str  # header, paragraph, list, code, etc.
    topic_cluster: int
    semantic_density: float
    importance_score: float
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = None

@dataclass
class SemanticChunk:
    """Semantically-aware document chunk"""
    id: str
    content: str
    title: str
    url: str
    category: str
    tags: List[str]
    chunk_index: int
    word_count: int
    character_count: int
    sentences: List[str]
    embedding: Optional[np.ndarray]
    metadata: ChunkMetadata
    original_metadata: Dict[str, Any]

class AdvancedDocumentChunker:
    """Advanced document chunking with semantic awareness"""
    
    def __init__(self, 
                 target_chunk_size: int = 800,
                 min_chunk_size: int = 200,
                 max_chunk_size: int = 1200,
                 overlap_percentage: float = 0.15,
                 use_semantic_clustering: bool = True,
                 embedding_model_name: str = "all-MiniLM-L6-v2"):
        
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_percentage = overlap_percentage
        self.use_semantic_clustering = use_semantic_clustering
        
        # Initialize models
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Try to load spaCy model for better sentence splitting
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Using basic sentence splitting.")
            self.nlp = None
        
        # Content type patterns
        self.content_patterns = {
            'header': re.compile(r'^#{1,6}\s+.+$', re.MULTILINE),
            'list_item': re.compile(r'^\s*[-*+]\s+.+$', re.MULTILINE),
            'numbered_list': re.compile(r'^\s*\d+\.\s+.+$', re.MULTILINE),
            'code_block': re.compile(r'```[\s\S]*?```', re.MULTILINE),
            'inline_code': re.compile(r'`[^`]+`'),
            'bold': re.compile(r'\*\*[^*]+\*\*'),
            'italic': re.compile(r'\*[^*]+\*'),
            'link': re.compile(r'\[([^\]]+)\]\([^)]+\)'),
        }
    
    def chunk_documents(self, articles: List[Dict[str, Any]]) -> List[SemanticChunk]:
        """Create advanced chunks from articles"""
        all_chunks = []
        
        for article in articles:
            if self._is_valid_article(article):
                article_chunks = self._create_semantic_chunks(article)
                all_chunks.extend(article_chunks)
        
        # Post-process chunks for semantic clustering
        if self.use_semantic_clustering and all_chunks:
            all_chunks = self._apply_semantic_clustering(all_chunks)
        
        logger.info(f"Created {len(all_chunks)} semantic chunks from {len(articles)} articles")
        return all_chunks
    
    def _is_valid_article(self, article: Dict[str, Any]) -> bool:
        """Validate article for processing"""
        content = article.get('content', '')
        title = article.get('title', '')
        
        if not content or not title or len(content.split()) < 30:
            return False
        
        return True
    
    def _create_semantic_chunks(self, article: Dict[str, Any]) -> List[SemanticChunk]:
        """Create semantically-aware chunks from an article"""
        content = self._clean_content(article.get('content', ''))
        title = article.get('title', '')
        url = article.get('url', '')
        category = article.get('category', '')
        tags = article.get('tags', [])
        
        # Parse content structure
        content_blocks = self._parse_content_structure(content)
        
        # Create base chunks
        base_chunks = self._create_base_chunks(content_blocks)
        
        # Merge small chunks and split large ones
        optimized_chunks = self._optimize_chunk_sizes(base_chunks)
        
        # Create SemanticChunk objects
        semantic_chunks = []
        for i, chunk_data in enumerate(optimized_chunks):
            chunk = self._create_semantic_chunk(
                chunk_data, title, url, category, tags, i, article
            )
            semantic_chunks.append(chunk)
        
        # Create hierarchical relationships
        self._establish_chunk_relationships(semantic_chunks)
        
        return semantic_chunks
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content"""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Normalize line breaks
        content = re.sub(r'\n\s*\n', '\n\n', content)
        
        # Remove HTML entities
        content = re.sub(r'&[a-zA-Z]+;', ' ', content)
        
        return content.strip()
    
    def _parse_content_structure(self, content: str) -> List[Dict[str, Any]]:
        """Parse content into structured blocks"""
        blocks = []
        
        # Split by double newlines (paragraphs)
        paragraphs = content.split('\n\n')
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            block_type = self._identify_content_type(paragraph)
            
            blocks.append({
                'content': paragraph,
                'type': block_type,
                'sentences': self._split_into_sentences(paragraph),
                'word_count': len(paragraph.split()),
                'importance': self._calculate_content_importance(paragraph, block_type)
            })
        
        return blocks
    
    def _identify_content_type(self, text: str) -> str:
        """Identify the type of content block"""
        if self.content_patterns['header'].match(text):
            return 'header'
        elif self.content_patterns['list_item'].match(text):
            return 'list'
        elif self.content_patterns['numbered_list'].match(text):
            return 'numbered_list'
        elif self.content_patterns['code_block'].search(text):
            return 'code'
        elif len(text.split()) < 10:
            return 'fragment'
        else:
            return 'paragraph'
    
    def _calculate_content_importance(self, text: str, content_type: str) -> float:
        """Calculate importance score for content"""
        score = 1.0
        
        # Type-based scoring
        type_scores = {
            'header': 1.5,
            'paragraph': 1.0,
            'list': 1.2,
            'numbered_list': 1.3,
            'code': 1.4,
            'fragment': 0.5
        }
        score *= type_scores.get(content_type, 1.0)
        
        # Length-based scoring (prefer medium-length content)
        word_count = len(text.split())
        if 50 <= word_count <= 200:
            score *= 1.2
        elif word_count < 20:
            score *= 0.7
        
        # Keyword-based scoring (HubSpot-specific terms)
        hubspot_keywords = [
            'contact', 'deal', 'pipeline', 'workflow', 'automation',
            'integration', 'report', 'dashboard', 'property', 'campaign'
        ]
        
        keyword_count = sum(1 for keyword in hubspot_keywords if keyword in text.lower())
        score += keyword_count * 0.1
        
        return min(score, 2.0)  # Cap at 2.0
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy or regex"""
        if self.nlp:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents]
        else:
            # Fallback to regex-based splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _create_base_chunks(self, content_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create base chunks from content blocks"""
        chunks = []
        current_chunk = {
            'blocks': [],
            'word_count': 0,
            'sentences': [],
            'types': set(),
            'importance': 0.0
        }
        
        for block in content_blocks:
            # Check if adding this block would exceed target size
            if (current_chunk['word_count'] + block['word_count'] > self.target_chunk_size 
                and current_chunk['blocks']):
                
                # Finalize current chunk
                chunks.append(self._finalize_chunk(current_chunk))
                
                # Start new chunk with overlap
                current_chunk = self._create_overlapping_chunk(current_chunk, block)
            else:
                # Add block to current chunk
                current_chunk['blocks'].append(block)
                current_chunk['word_count'] += block['word_count']
                current_chunk['sentences'].extend(block['sentences'])
                current_chunk['types'].add(block['type'])
                current_chunk['importance'] += block['importance']
        
        # Add final chunk
        if current_chunk['blocks']:
            chunks.append(self._finalize_chunk(current_chunk))
        
        return chunks
    
    def _create_overlapping_chunk(self, previous_chunk: Dict[str, Any], new_block: Dict[str, Any]) -> Dict[str, Any]:
        """Create new chunk with overlap from previous chunk"""
        overlap_words = int(self.target_chunk_size * self.overlap_percentage)
        
        # Get sentences from end of previous chunk for overlap
        overlap_sentences = []
        word_count = 0
        
        for sentence in reversed(previous_chunk['sentences']):
            sentence_words = len(sentence.split())
            if word_count + sentence_words <= overlap_words:
                overlap_sentences.insert(0, sentence)
                word_count += sentence_words
            else:
                break
        
        # Create new chunk
        overlap_block = {
            'content': ' '.join(overlap_sentences),
            'type': 'overlap',
            'sentences': overlap_sentences,
            'word_count': word_count,
            'importance': 0.5
        }
        
        return {
            'blocks': [overlap_block, new_block],
            'word_count': word_count + new_block['word_count'],
            'sentences': overlap_sentences + new_block['sentences'],
            'types': {'overlap', new_block['type']},
            'importance': overlap_block['importance'] + new_block['importance']
        }
    
    def _finalize_chunk(self, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize chunk by combining blocks"""
        content = '\n\n'.join(block['content'] for block in chunk_data['blocks'])
        
        return {
            'content': content,
            'word_count': chunk_data['word_count'],
            'sentences': chunk_data['sentences'],
            'types': list(chunk_data['types']),
            'importance': chunk_data['importance'] / len(chunk_data['blocks']),
            'blocks': chunk_data['blocks']
        }
    
    def _optimize_chunk_sizes(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize chunk sizes by merging small chunks and splitting large ones"""
        optimized = []
        i = 0
        
        while i < len(chunks):
            chunk = chunks[i]
            
            if chunk['word_count'] < self.min_chunk_size:
                # Try to merge with next chunk
                if i + 1 < len(chunks) and chunks[i + 1]['word_count'] + chunk['word_count'] <= self.max_chunk_size:
                    merged = self._merge_chunks(chunk, chunks[i + 1])
                    optimized.append(merged)
                    i += 2
                else:
                    optimized.append(chunk)
                    i += 1
            elif chunk['word_count'] > self.max_chunk_size:
                # Split large chunk
                split_chunks = self._split_large_chunk(chunk)
                optimized.extend(split_chunks)
                i += 1
            else:
                optimized.append(chunk)
                i += 1
        
        return optimized
    
    def _merge_chunks(self, chunk1: Dict[str, Any], chunk2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two chunks"""
        return {
            'content': chunk1['content'] + '\n\n' + chunk2['content'],
            'word_count': chunk1['word_count'] + chunk2['word_count'],
            'sentences': chunk1['sentences'] + chunk2['sentences'],
            'types': list(set(chunk1['types'] + chunk2['types'])),
            'importance': (chunk1['importance'] + chunk2['importance']) / 2,
            'blocks': chunk1['blocks'] + chunk2['blocks']
        }
    
    def _split_large_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a large chunk into smaller ones"""
        sentences = chunk['sentences']
        if len(sentences) <= 1:
            return [chunk]  # Can't split further
        
        # Split sentences into roughly equal parts
        mid_point = len(sentences) // 2
        
        first_sentences = sentences[:mid_point]
        second_sentences = sentences[mid_point:]
        
        first_content = ' '.join(first_sentences)
        second_content = ' '.join(second_sentences)
        
        return [
            {
                'content': first_content,
                'word_count': len(first_content.split()),
                'sentences': first_sentences,
                'types': chunk['types'],
                'importance': chunk['importance'],
                'blocks': chunk['blocks'][:len(chunk['blocks'])//2]
            },
            {
                'content': second_content,
                'word_count': len(second_content.split()),
                'sentences': second_sentences,
                'types': chunk['types'],
                'importance': chunk['importance'],
                'blocks': chunk['blocks'][len(chunk['blocks'])//2:]
            }
        ]
    
    def _create_semantic_chunk(self, chunk_data: Dict[str, Any], title: str, url: str, 
                              category: str, tags: List[str], index: int, 
                              article: Dict[str, Any]) -> SemanticChunk:
        """Create SemanticChunk object"""
        import hashlib
        
        # Generate unique ID
        chunk_id = hashlib.md5(f"{url}_{index}".encode()).hexdigest()[:12]
        
        # Calculate semantic density
        semantic_density = self._calculate_semantic_density(chunk_data['content'])
        
        # Create metadata
        primary_type = max(chunk_data['types'], key=lambda t: chunk_data['types'].count(t)) if chunk_data['types'] else 'paragraph'
        
        metadata = ChunkMetadata(
            section_type=primary_type,
            topic_cluster=0,  # Will be set later
            semantic_density=semantic_density,
            importance_score=chunk_data['importance']
        )
        
        # Generate embedding
        embedding = self.embedding_model.encode(chunk_data['content'])
        
        return SemanticChunk(
            id=chunk_id,
            content=chunk_data['content'],
            title=title,
            url=url,
            category=category,
            tags=tags,
            chunk_index=index,
            word_count=chunk_data['word_count'],
            character_count=len(chunk_data['content']),
            sentences=chunk_data['sentences'],
            embedding=embedding,
            metadata=metadata,
            original_metadata=article.get('metadata', {})
        )
    
    def _calculate_semantic_density(self, content: str) -> float:
        """Calculate semantic density of content"""
        words = content.split()
        if len(words) < 10:
            return 0.5
        
        # Simple heuristic: ratio of unique words to total words
        unique_words = len(set(word.lower() for word in words))
        density = unique_words / len(words)
        
        return min(density, 1.0)
    
    def _apply_semantic_clustering(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Apply semantic clustering to group related chunks"""
        if len(chunks) < 5:
            return chunks
        
        try:
            # Get embeddings
            embeddings = np.array([chunk.embedding for chunk in chunks])
            
            # Determine optimal number of clusters
            n_clusters = min(max(len(chunks) // 10, 3), 20)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Assign cluster labels to chunks
            for chunk, label in zip(chunks, cluster_labels):
                chunk.metadata.topic_cluster = int(label)
            
            logger.info(f"Applied semantic clustering: {n_clusters} clusters for {len(chunks)} chunks")
            
        except Exception as e:
            logger.warning(f"Semantic clustering failed: {str(e)}")
        
        return chunks
    
    def _establish_chunk_relationships(self, chunks: List[SemanticChunk]):
        """Establish parent-child relationships between chunks"""
        # Simple implementation: consecutive chunks from same article
        for i in range(len(chunks) - 1):
            current = chunks[i]
            next_chunk = chunks[i + 1]
            
            # If chunks are from same article and consecutive
            if (current.url == next_chunk.url and 
                next_chunk.chunk_index == current.chunk_index + 1):
                
                if current.metadata.child_chunk_ids is None:
                    current.metadata.child_chunk_ids = []
                current.metadata.child_chunk_ids.append(next_chunk.id)
                next_chunk.metadata.parent_chunk_id = current.id
    
    def get_chunking_stats(self, chunks: List[SemanticChunk]) -> Dict[str, Any]:
        """Get statistics about the chunking process"""
        if not chunks:
            return {}
        
        word_counts = [chunk.word_count for chunk in chunks]
        importance_scores = [chunk.metadata.importance_score for chunk in chunks]
        semantic_densities = [chunk.metadata.semantic_density for chunk in chunks]
        
        # Count by section type
        section_types = {}
        for chunk in chunks:
            section_type = chunk.metadata.section_type
            section_types[section_type] = section_types.get(section_type, 0) + 1
        
        # Count by topic cluster
        topic_clusters = {}
        for chunk in chunks:
            cluster = chunk.metadata.topic_cluster
            topic_clusters[cluster] = topic_clusters.get(cluster, 0) + 1
        
        return {
            'total_chunks': len(chunks),
            'word_count_stats': {
                'mean': np.mean(word_counts),
                'median': np.median(word_counts),
                'min': np.min(word_counts),
                'max': np.max(word_counts),
                'std': np.std(word_counts)
            },
            'importance_stats': {
                'mean': np.mean(importance_scores),
                'median': np.median(importance_scores),
                'min': np.min(importance_scores),
                'max': np.max(importance_scores)
            },
            'semantic_density_stats': {
                'mean': np.mean(semantic_densities),
                'median': np.median(semantic_densities),
                'min': np.min(semantic_densities),
                'max': np.max(semantic_densities)
            },
            'section_types': section_types,
            'topic_clusters': topic_clusters,
            'chunks_with_parents': sum(1 for c in chunks if c.metadata.parent_chunk_id),
            'chunks_with_children': sum(1 for c in chunks if c.metadata.child_chunk_ids)
        }