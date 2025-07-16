#!/usr/bin/env python3
"""
RAG System with XLM-RoBERTa Embeddings
=====================================

A comprehensive Retrieval-Augmented Generation system using XLM-RoBERTa
for multilingual text embeddings with local storage and query processing.

Features:
- XLM-RoBERTa based embeddings for multilingual support
- Local vector storage with metadata
- Semantic similarity search
- Query processing with context retrieval
- Embedding visualization and analysis
- Batch processing support

Dependencies:
pip install transformers torch sentence-transformers faiss-cpu pandas numpy
pip install scikit-learn matplotlib seaborn plotly umap-learn
pip install gradio streamlit (optional for web interface)
"""

import os
import sys
import json
import pickle
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Core libraries
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Deep learning and embeddings
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# Vector search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available. Install with: pip install faiss-cpu")

# Visualization
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Install with: pip install plotly")

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Data class for document chunks"""
    chunk_id: str
    text: str
    source_file: str
    chunk_index: int
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        if self.embedding is not None:
            data['embedding'] = self.embedding.tolist()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentChunk':
        """Create from dictionary"""
        embedding = None
        if 'embedding' in data and data['embedding'] is not None:
            embedding = np.array(data['embedding'])

        return cls(
            chunk_id=data['chunk_id'],
            text=data['text'],
            source_file=data['source_file'],
            chunk_index=data['chunk_index'],
            metadata=data.get('metadata', {}),
            embedding=embedding
        )

@dataclass
class QueryResult:
    """Data class for query results"""
    query: str
    results: List[Dict[str, Any]]
    processing_time: float
    embedding_model: str
    search_method: str


def _load_embedding_model(self):
    """Load the embedding model"""
    try:
        logger.info(f"Loading embedding model: {self.model_name}")

        # Use SentenceTransformer for better multilingual support
        self.model = SentenceTransformer(self.model_name, device=self.device)
        
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")

    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        # Fallback to basic XLM-RoBERTa
        try:
            logger.info("Falling back to basic XLM-RoBERTa model")
            self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
            self.model = AutoModel.from_pretrained("xlm-roberta-base").to(self.device)
            self.embedding_dim = 768  # XLM-RoBERTa base dimension
            self.use_sentence_transformer = False
        except Exception as e2:
            logger.error(f"Failed to load fallback model: {e2}")
            raise e2
    else:
        self.use_sentence_transformer = True


    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        try:
            if self.use_sentence_transformer:
                # Use SentenceTransformer
                embedding = self.model.encode(text, convert_to_numpy=True)
            else:
                # Use basic transformer with mean pooling
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                      padding=True, max_length=512).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Mean pooling
                    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()

            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(self.embedding_dim)

    def _generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        try:
            if self.use_sentence_transformer:
                embeddings = self.model.encode(texts, convert_to_numpy=True,
                                             show_progress_bar=True, batch_size=32)
            else:
                embeddings = []
                for text in texts:
                    emb = self._generate_embedding(text)
                    embeddings.append(emb)
                embeddings = np.array(embeddings)

            return embeddings

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            # Return zero vectors as fallback
            return np.zeros((len(texts), self.embedding_dim))

    def add_documents_from_texts(self, text_files: List[Union[str, Path]]) -> Dict[str, Any]:
        """
        Add documents from extracted text files

        Args:
            text_files: List of paths to text files

        Returns:
            Dictionary with processing statistics
        """
        start_time = datetime.now()

        stats = {
            'files_processed': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'errors': []
        }

        all_new_chunks = []

        for text_file in text_files:
            try:
                text_file = Path(text_file)

                if not text_file.exists():
                    stats['errors'].append(f"File not found: {text_file}")
                    continue

                # Read text file
                with open(text_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract the actual extracted text (skip the header)
                if "=== EXTRACTED TEXT ===" in content:
                    text_content = content.split("=== EXTRACTED TEXT ===")[1].strip()
                else:
                    text_content = content

                # Create chunks
                chunks = self._chunk_text(text_content, str(text_file))
                all_new_chunks.extend(chunks)

                stats['files_processed'] += 1
                stats['chunks_created'] += len(chunks)

                logger.info(f"Processed {text_file.name}: {len(chunks)} chunks created")

            except Exception as e:
                error_msg = f"Error processing {text_file}: {e}"
                stats['errors'].append(error_msg)
                logger.error(error_msg)

        # Generate embeddings in batches
        if all_new_chunks:
            logger.info(f"Generating embeddings for {len(all_new_chunks)} chunks...")

            texts = [chunk.text for chunk in all_new_chunks]
            embeddings = self._generate_embeddings_batch(texts)

            # Assign embeddings to chunks
            for chunk, embedding in zip(all_new_chunks, embeddings):
                chunk.embedding = embedding

            # Add to storage
            self.documents.extend(all_new_chunks)

            # Update embeddings matrix
            if self.document_embeddings is None:
                self.document_embeddings = embeddings
            else:
                self.document_embeddings = np.vstack([self.document_embeddings, embeddings])

            stats['embeddings_generated'] = len(embeddings)

            # Save to disk
            self._save_documents()
            self._save_embeddings()
            self._update_metadata(stats)

            # Rebuild FAISS index
            if FAISS_AVAILABLE:
                self._build_faiss_index()

        processing_time = (datetime.now() - start_time).total_seconds()
        stats['processing_time'] = processing_time

        logger.info(f"Document processing completed in {processing_time:.2f} seconds")
        logger.info(f"Stats: {stats}")

        return stats

    def add_document_from_text(self, text: str, source_file: str = "direct_input") -> Dict[str, Any]:
        """
        Add a single document from text string

        Args:
            text: Text content to add
            source_file: Source identifier

        Returns:
            Processing statistics
        """
        start_time = datetime.now()

        # Create chunks
        chunks = self._chunk_text(text, source_file)

        if not chunks:
            return {
                'chunks_created': 0,
                'embeddings_generated': 0,
                'processing_time': 0,
                'error': 'No valid chunks created from text'
            }

        # Generate embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = self._generate_embeddings_batch(texts)

        # Assign embeddings
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        # Add to storage
        self.documents.extend(chunks)

        if self.document_embeddings is None:
            self.document_embeddings = embeddings
        else:
            self.document_embeddings = np.vstack([self.document_embeddings, embeddings])

        # Save to disk
        self._save_documents()
        self._save_embeddings()

        # Rebuild FAISS index
        if FAISS_AVAILABLE:
            self._build_faiss_index()

        processing_time = (datetime.now() - start_time).total_seconds()

        stats = {
            'chunks_created': len(chunks),
            'embeddings_generated': len(embeddings),
            'processing_time': processing_time
        }

        self._update_metadata(stats)

        return stats

    def query(self,
              query_text: str,
              top_k: int = 5,
              search_method: str = "cosine",
              min_similarity: float = 0.0) -> QueryResult:
        """
        Query the document collection

        Args:
            query_text: Query string
            top_k: Number of top results to return
            search_method: Search method ('cosine', 'faiss', 'semantic')
            min_similarity: Minimum similarity threshold

        Returns:
            QueryResult object
        """
        start_time = datetime.now()

        if not self.documents:
            return QueryResult(
                query=query_text,
                results=[],
                processing_time=0,
                embedding_model=self.model_name,
                search_method=search_method
            )

        # Generate query embedding
        query_embedding = self._generate_embedding(query_text)

        if search_method == "faiss" and FAISS_AVAILABLE and self.faiss_index:
            results = self._search_faiss(query_embedding, top_k, min_similarity)
        elif search_method == "semantic":
            results = self._search_semantic(query_text, query_embedding, top_k, min_similarity)
        else:  # cosine similarity
            results = self._search_cosine(query_embedding, top_k, min_similarity)

        processing_time = (datetime.now() - start_time).total_seconds()

        return QueryResult(
            query=query_text,
            results=results,
            processing_time=processing_time,
            embedding_model=self.model_name,
            search_method=search_method
        )

    def _search_cosine(self, query_embedding: np.ndarray, top_k: int, min_similarity: float) -> List[Dict[str, Any]]:
        """Search using cosine similarity"""
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.document_embeddings)[0]

        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            similarity = float(similarities[idx])

            if similarity >= min_similarity:
                doc = self.documents[idx]
                results.append({
                    'chunk_id': doc.chunk_id,
                    'text': doc.text,
                    'source_file': doc.source_file,
                    'similarity': similarity,
                    'metadata': doc.metadata
                })

        return results

    def _search_faiss(self, query_embedding: np.ndarray, top_k: int, min_similarity: float) -> List[Dict[str, Any]]:
        """Search using FAISS index"""
        if not self.faiss_index:
            return self._search_cosine(query_embedding, top_k, min_similarity)

        # Search FAISS index
        query_vector = query_embedding.reshape(1, -1).astype('float32')
        similarities, indices = self.faiss_index.search(query_vector, top_k)

        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx != -1 and similarity >= min_similarity:
                doc = self.documents[idx]
                results.append({
                    'chunk_id': doc.chunk_id,
                    'text': doc.text,
                    'source_file': doc.source_file,
                    'similarity': float(similarity),
                    'metadata': doc.metadata
                })

        return results

    def _search_semantic(self, query_text: str, query_embedding: np.ndarray, top_k: int, min_similarity: float) -> List[Dict[str, Any]]:
        """Enhanced semantic search with keyword matching"""
        # Start with cosine similarity results
        cosine_results = self._search_cosine(query_embedding, top_k * 2, 0.0)  # Get more candidates

        # Add keyword-based scoring
        query_words = set(query_text.lower().split())

        enhanced_results = []
        for result in cosine_results:
            text_words = set(result['text'].lower().split())
            keyword_overlap = len(query_words.intersection(text_words)) / len(query_words)

            # Combine semantic and keyword scores
            combined_score = 0.7 * result['similarity'] + 0.3 * keyword_overlap

            if combined_score >= min_similarity:
                result['combined_score'] = combined_score
                result['keyword_overlap'] = keyword_overlap
                enhanced_results.append(result)

        # Sort by combined score and return top_k
        enhanced_results.sort(key=lambda x: x['combined_score'], reverse=True)
        return enhanced_results[:top_k]

    def _build_faiss_index(self):
        """Build FAISS index for fast similarity search"""
        if not FAISS_AVAILABLE or self.document_embeddings is None:
            return

        try:
            # Create FAISS index
            index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity

            # Normalize embeddings for cosine similarity
            embeddings_normalized = self.document_embeddings.astype('float32')
            faiss.normalize_L2(embeddings_normalized)

            # Add to index
            index.add(embeddings_normalized)

            self.faiss_index = index

            # Save index
            faiss.write_index(index, str(self.index_file))

            logger.info(f"FAISS index built with {index.ntotal} vectors")

        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            self.faiss_index = None

    def _load_documents(self) -> List[DocumentChunk]:
        """Load documents from storage"""
        if not self.documents_file.exists():
            return []

        try:
            with open(self.documents_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            documents = [DocumentChunk.from_dict(doc_data) for doc_data in data]
            logger.info(f"Loaded {len(documents)} documents from storage")
            return documents

        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return []

    def _load_embeddings(self) -> Optional[np.ndarray]:
        """Load embeddings from storage"""
        if not self.embeddings_file.exists():
            return None

        try:
            with open(self.embeddings_file, 'rb') as f:
                embeddings = pickle.load(f)

            logger.info(f"Loaded embeddings matrix: {embeddings.shape}")
            return embeddings

        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return None

    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from storage"""
        if not self.metadata_file.exists():
            return {
                'created_at': datetime.now().isoformat(),
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim,
                'total_documents': 0,
                'total_chunks': 0,
                'processing_history': []
            }

        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            return metadata

        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return {}

    def _save_documents(self):
        """Save documents to storage"""
        try:
            data = [doc.to_dict() for doc in self.documents]

            with open(self.documents_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(self.documents)} documents to storage")

        except Exception as e:
            logger.error(f"Error saving documents: {e}")

    def _save_embeddings(self):
        """Save embeddings to storage"""
        try:
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.document_embeddings, f)

            logger.info(f"Saved embeddings matrix: {self.document_embeddings.shape}")

        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")

    def _update_metadata(self, stats: Dict[str, Any]):
        """Update metadata with processing statistics"""
        self.metadata.update({
            'last_updated': datetime.now().isoformat(),
            'total_documents': len(set(doc.source_file for doc in self.documents)),
            'total_chunks': len(self.documents),
            'embedding_dim': self.embedding_dim,
            'model_name': self.model_name
        })

        self.metadata['processing_history'].append({
            'timestamp': datetime.now().isoformat(),
            'stats': stats
        })

        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        # Basic stats (always present)
        stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'embedding_dimension': self.embedding_dim,
            'model_name': self.model_name,
            'storage_directory': str(self.storage_dir),
            'faiss_available': FAISS_AVAILABLE,
            'faiss_index_built': self.faiss_index is not None,
            'document_sources': {},
            'text_statistics': {
                'total_words': 0,
                'total_characters': 0,
                'avg_words_per_chunk': 0,
                'avg_chars_per_chunk': 0
            }
        }

        if not self.documents:
            stats['status'] = 'No documents in system'
            return stats

        # Update with actual data
        stats['total_chunks'] = len(self.documents)
        stats['total_documents'] = len(set(doc.source_file for doc in self.documents))

        # Document sources
        sources = {}
        for doc in self.documents:
            source = doc.source_file
            if source not in sources:
                sources[source] = 0
            sources[source] += 1

        stats['document_sources'] = sources

        # Text statistics
        word_counts = [doc.metadata.get('word_count', 0) for doc in self.documents]
        char_counts = [doc.metadata.get('char_count', 0) for doc in self.documents]

        stats['text_statistics'] = {
            'total_words': sum(word_counts),
            'total_characters': sum(char_counts),
            'avg_words_per_chunk': np.mean(word_counts) if word_counts else 0,
            'avg_chars_per_chunk': np.mean(char_counts) if char_counts else 0
        }

        stats['status'] = 'Ready'
        return stats

    def visualize_embeddings(self,
                           method: str = "tsne",
                           save_path: Optional[str] = None,
                           sample_size: int = 1000) -> Optional[str]:
        """
        Visualize embeddings in 2D space

        Args:
            method: Visualization method ('tsne', 'pca', 'umap')
            save_path: Path to save the visualization
            sample_size: Maximum number of points to visualize

        Returns:
            Path to saved visualization or None
        """
        if self.document_embeddings is None or len(self.documents) == 0:
            logger.error("No embeddings available for visualization")
            return None

        # Sample data if too large
        if len(self.documents) > sample_size:
            indices = np.random.choice(len(self.documents), sample_size, replace=False)
            embeddings_sample = self.document_embeddings[indices]
            docs_sample = [self.documents[i] for i in indices]
        else:
            embeddings_sample = self.document_embeddings
            docs_sample = self.documents

        # Apply dimensionality reduction
        if method == "pca":
            reducer = PCA(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings_sample)
            title = f"PCA Visualization of Document Embeddings ({len(docs_sample)} documents)"
        elif method == "umap" and UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings_sample)
            title = f"UMAP Visualization of Document Embeddings ({len(docs_sample)} documents)"
        else:  # tsne
            if len(embeddings_sample) > 500:
                # Use PCA first for large datasets
                pca = PCA(n_components=50, random_state=42)
                embeddings_reduced = pca.fit_transform(embeddings_sample)
            else:
                embeddings_reduced = embeddings_sample

            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_sample)//4))
            embeddings_2d = reducer.fit_transform(embeddings_reduced)
            title = f"t-SNE Visualization of Document Embeddings ({len(docs_sample)} documents)"

        # Create visualization
        if PLOTLY_AVAILABLE:
            # Interactive plot with Plotly
            sources = [doc.source_file for doc in docs_sample]
            hover_texts = [f"Source: {doc.source_file}<br>Chunk: {doc.chunk_index}<br>Text: {doc.text[:100]}..."
                          for doc in docs_sample]

            fig = px.scatter(
                x=embeddings_2d[:, 0],
                y=embeddings_2d[:, 1],
                color=sources,
                title=title,
                hover_name=hover_texts,
                labels={'x': 'Dimension 1', 'y': 'Dimension 2', 'color': 'Source File'}
            )

            fig.update_traces(marker=dict(size=5, opacity=0.7))
            fig.update_layout(height=600, width=800)

            if save_path:
                save_path = Path(save_path)
                if save_path.suffix.lower() == '.html':
                    fig.write_html(save_path)
                else:
                    fig.write_image(save_path)
            else:
                save_path = self.storage_dir / f"embeddings_visualization_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                fig.write_html(save_path)
        else:
            # Static plot with Matplotlib
            plt.figure(figsize=(12, 8))

            # Color by source file
            unique_sources = list(set(doc.source_file for doc in docs_sample))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_sources)))
            source_colors = {source: colors[i] for i, source in enumerate(unique_sources)}

            for source in unique_sources:
                mask = [doc.source_file == source for doc in docs_sample]
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                           label=source, alpha=0.7, s=20)

            plt.title(title)
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            if save_path:
                save_path = Path(save_path)
            else:
                save_path = self.storage_dir / f"embeddings_visualization_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        logger.info(f"Embedding visualization saved to: {save_path}")
        return str(save_path)

    def similarity_analysis(self,
                          query_text: str,
                          top_k: int = 10) -> Dict[str, Any]:
        """
        Analyze similarity patterns for a query

        Args:
            query_text: Query text to analyze
            top_k: Number of top results to analyze

        Returns:
            Analysis results
        """
        query_result = self.query(query_text, top_k=top_k, search_method="semantic")

        if not query_result.results:
            return {'error': 'No results found for query'}

        analysis = {
            'query': query_text,
            'top_results': query_result.results,
            'similarity_distribution': {},
            'source_distribution': {},
            'content_analysis': {}
        }

        # Similarity distribution
        similarities = [r['similarity'] for r in query_result.results]
        analysis['similarity_distribution'] = {
            'mean': float(np.mean(similarities)),
            'std': float(np.std(similarities)),
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities)),
            'median': float(np.median(similarities))
        }

        # Source distribution
        sources = [r['source_file'] for r in query_result.results]
        source_counts = {source: sources.count(source) for source in set(sources)}
        analysis['source_distribution'] = source_counts

        # Content analysis
        all_text = ' '.join([r['text'] for r in query_result.results])
        words = all_text.lower().split()
        word_counts = {word: words.count(word) for word in set(words)}
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        analysis['content_analysis'] = {
            'total_words': len(words),
            'unique_words': len(set(words)),
            'top_words': top_words
        }

        return analysis

    def export_embeddings(self,
                         export_path: Optional[str] = None,
                         format: str = "csv") -> str:
        """
        Export embeddings and metadata to external format

        Args:
            export_path: Path to export file
            format: Export format ('csv', 'json', 'parquet')

        Returns:
            Path to exported file
        """
        if not self.documents:
            raise ValueError("No documents available for export")

        # Prepare data
        export_data = []
        for i, doc in enumerate(self.documents):
            row = {
                'chunk_id': doc.chunk_id,
                'source_file': doc.source_file,
                'chunk_index': doc.chunk_index,
                'text': doc.text,
                'word_count': doc.metadata.get('word_count', 0),
                'char_count': doc.metadata.get('char_count', 0),
                'created_at': doc.metadata.get('created_at', '')
            }

            # Add embedding dimensions
            if doc.embedding is not None:
                for j, val in enumerate(doc.embedding):
                    row[f'embedding_{j}'] = float(val)

            export_data.append(row)

        # Export based on format
        if export_path is None:
            export_path = self.storage_dir / f"embeddings_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"

        export_path = Path(export_path)

        if format == "csv":
            df = pd.DataFrame(export_data)
            df.to_csv(export_path, index=False)
        elif format == "json":
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
        elif format == "parquet":
            df = pd.DataFrame(export_data)
            df.to_parquet(export_path, index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Embeddings exported to: {export_path}")
        return str(export_path)

    def clear_storage(self):
        """Clear all stored data"""
        files_to_remove = [
            self.documents_file,
            self.embeddings_file,
            self.metadata_file,
            self.index_file
        ]

        for file_path in files_to_remove:
            if file_path.exists():
                file_path.unlink()

        # Reset in-memory data
        self.documents = []
        self.document_embeddings = None
        self.metadata = {}
        self.faiss_index = None

        logger.info("All storage cleared")

def main():
    """Main function with interactive user input"""
    print("🤖 RAG System with XLM-RoBERTa Embeddings")
    print("=" * 60)

    # Get user preferences
    print("\n⚙️ System Configuration:")

    # Storage directory
    storage_dir = input("Storage directory (default: embeddings_storage): ").strip()
    if not storage_dir:
        storage_dir = "embeddings_storage"

    # Model selection
    print("\nAvailable models:")
    print("1. sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (default, fast)")
    print("2. sentence-transformers/distiluse-base-multilingual-cased")
    print("3. sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (best quality)")
    print("4. Custom model (enter manually)")

    model_choice = input("Choose model (1-4, default: 1): ").strip()

    model_map = {
        "1": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "2": "sentence-transformers/distiluse-base-multilingual-cased",
        "3": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    }

    if model_choice == "4":
        model_name = input("Enter custom model name: ").strip()
        if not model_name:
            model_name = model_map["1"]
    else:
        model_name = model_map.get(model_choice, model_map["1"])

    print(f"\n🔄 Initializing RAG system with model: {model_name}")

    # Initialize system
    rag_system = RAGEmbeddingSystem(
        model_name=model_name,
        storage_dir=storage_dir
    )

    # Get current stats safely
    current_stats = rag_system.get_statistics()
    if 'error' in current_stats:
        chunk_count = 0
    else:
        chunk_count = current_stats.get('total_chunks', 0)

    print(f"📊 Current system stats: {chunk_count} chunks in database")

    # Quick start option for new users
    if chunk_count == 0:
        print("\n🚀 Quick Start: No documents found in the system")
        print("Would you like to add some text files now?")
        print("1. Yes, add text files")
        print("2. No, go to main menu")

        quick_choice = input("Choose (1/2, default: 1): ").strip()

        if quick_choice != "2":
            print("\n📄 Quick Add Text Files:")
            print("💡 If you just used the text extraction system, look for files in 'extracted_texts' folder")

            # Check for common extraction folders
            common_folders = ["extracted_texts", "output", "extracts"]
            found_extracts = False

            for folder_name in common_folders:
                folder_path = Path(folder_name)
                if folder_path.exists():
                    txt_files = list(folder_path.glob("*_extracted_*.txt"))
                    if txt_files:
                        print(f"\n🔍 Found {len(txt_files)} extracted files in '{folder_name}' folder!")

                        # Show first few files
                        print("Recent extractions:")
                        for i, file_path in enumerate(txt_files[-5:], 1):  # Show last 5
                            print(f"  {i}. {file_path.name}")

                        use_found = input(f"\nUse these extracted files? (y/n, default: y): ").strip().lower()
                        if use_found != 'n':
                            print(f"\n🔄 Processing {len(txt_files)} extracted files...")
                            stats = rag_system.add_documents_from_texts([str(f) for f in txt_files])

                            print("\n✅ Quick start completed!")
                            print(f"📄 Files processed: {stats['files_processed']}")
                            print(f"📝 Chunks created: {stats['chunks_created']}")
                            print(f"🧠 Embeddings generated: {stats['embeddings_generated']}")

                            found_extracts = True
                            break

            if not found_extracts:
                # Manual file entry for quick start
                print("\nEnter text file paths (empty line to finish):")
                quick_files = []

                while True:
                    file_path = input("Text file path: ").strip()
                    if not file_path:
                        break

                    if Path(file_path).exists():
                        quick_files.append(file_path)
                        print(f"  ✅ Added: {Path(file_path).name}")
                    else:
                        print(f"  ❌ File not found: {file_path}")

                if quick_files:
                    print(f"\n🔄 Processing {len(quick_files)} files...")
                    stats = rag_system.add_documents_from_texts(quick_files)

                    print("\n✅ Quick start completed!")
                    print(f"📄 Files processed: {stats['files_processed']}")
                    print(f"📝 Chunks created: {stats['chunks_created']}")
                    print(f"🧠 Embeddings generated: {stats['embeddings_generated']}")
                else:
                    print("No files added. You can add them later from the main menu.")

    try:
        while True:
            # Main menu
            print("\n🎯 Main Menu:")
            print("1. Add text files to system")
            print("2. Add single text content")
            print("3. Query the system")
            print("4. Show system statistics")
            print("5. Visualize embeddings")
            print("6. Export embeddings")
            print("7. Clear all data")
            print("8. Exit")

            choice = input("\nChoose option (1-8): ").strip()

            if choice == "1":
                # Add text files
                print("\n📄 Add Text Files to RAG System:")
                print("Choose input method:")
                print("1. Enter file paths manually")
                print("2. Select from extraction output folder")
                print("3. Process entire folder of text files")
                print("4. Load from file list")

                method = input("Choose method (1-4, default: 1): ").strip()

                text_files = []

                if method == "2":
                    # Look for extraction output folder
                    default_extract_dir = "extracted_texts"
                    extract_dir = input(f"Extraction output folder (default: {default_extract_dir}): ").strip()
                    if not extract_dir:
                        extract_dir = default_extract_dir

                    extract_path = Path(extract_dir)
                    if extract_path.exists():
                        # Find all extracted text files
                        txt_files = list(extract_path.glob("*_extracted_*.txt"))

                        if txt_files:
                            print(f"\n📋 Found {len(txt_files)} extracted text files:")
                            for i, file_path in enumerate(txt_files, 1):
                                print(f"  {i}. {file_path.name}")

                            selection = input("\nProcess all files? (y/n, default: y): ").strip().lower()
                            if selection != 'n':
                                text_files = [str(f) for f in txt_files]
                            else:
                                print("\nSelect files to process (enter numbers separated by commas, e.g., 1,3,5):")
                                indices = input("File numbers: ").strip()
                                try:
                                    selected_indices = [int(i.strip()) - 1 for i in indices.split(',')]
                                    text_files = [str(txt_files[i]) for i in selected_indices if 0 <= i < len(txt_files)]
                                except:
                                    print("❌ Invalid selection, processing all files")
                                    text_files = [str(f) for f in txt_files]
                        else:
                            print(f"❌ No extracted text files found in {extract_dir}")
                    else:
                        print(f"❌ Extraction folder not found: {extract_dir}")

                elif method == "3":
                    # Process entire folder
                    folder_path = input("\n📂 Enter folder path containing text files: ").strip()
                    if folder_path:
                        folder = Path(folder_path)
                        if folder.exists():
                            # Look for text files
                            patterns = ["*.txt", "*.text"]
                            all_txt_files = []

                            for pattern in patterns:
                                all_txt_files.extend(folder.glob(pattern))

                            if all_txt_files:
                                print(f"\n📋 Found {len(all_txt_files)} text files:")
                                for i, file_path in enumerate(all_txt_files[:10], 1):  # Show first 10
                                    print(f"  {i}. {file_path.name}")

                                if len(all_txt_files) > 10:
                                    print(f"  ... and {len(all_txt_files) - 10} more files")

                                confirm = input(f"\nProcess all {len(all_txt_files)} text files? (y/n): ").strip().lower()
                                if confirm == 'y':
                                    text_files = [str(f) for f in all_txt_files]
                                else:
                                    print("❌ Processing cancelled")
                            else:
                                print(f"❌ No text files found in {folder_path}")
                        else:
                            print(f"❌ Folder not found: {folder_path}")

                elif method == "4":
                    # Load from file list
                    list_file = input("\n📄 Enter path to file containing text file paths (one per line): ").strip()
                    if list_file:
                        try:
                            with open(list_file, 'r', encoding='utf-8') as f:
                                file_paths = [line.strip() for line in f if line.strip() and not line.startswith('#')]

                            # Validate files exist
                            valid_files = []
                            for file_path in file_paths:
                                if Path(file_path).exists():
                                    valid_files.append(file_path)
                                else:
                                    print(f"⚠️ File not found: {file_path}")

                            if valid_files:
                                print(f"\n📋 Found {len(valid_files)} valid text files from list")
                                text_files = valid_files
                            else:
                                print("❌ No valid files found in the list")

                        except Exception as e:
                            print(f"❌ Error reading file list: {e}")

                else:
                    # Method 1: Manual entry (default)
                    print("\n📄 Enter text file paths:")
                    print("💡 Tips:")
                    print("  - Enter full paths to text files")
                    print("  - You can use extracted files from the text extraction system")
                    print("  - Press Enter on empty line to finish")
                    print()

                    while True:
                        file_path = input("Text file path: ").strip()
                        if not file_path:
                            break

                        # Validate file exists
                        if Path(file_path).exists():
                            text_files.append(file_path)
                            print(f"  ✅ Added: {Path(file_path).name}")
                        else:
                            print(f"  ❌ File not found: {file_path}")
                            retry = input("  Try again? (y/n): ").strip().lower()
                            if retry != 'y':
                                break

                # Process the collected files
                if text_files:
                    print(f"\n🔄 Processing {len(text_files)} files...")

                    # Show file preview
                    if len(text_files) <= 5:
                        print("Files to process:")
                        for i, file_path in enumerate(text_files, 1):
                            print(f"  {i}. {Path(file_path).name}")
                    else:
                        print(f"Files to process: {len(text_files)} files")
                        show_list = input("Show full file list? (y/n): ").strip().lower()
                        if show_list == 'y':
                            for i, file_path in enumerate(text_files, 1):
                                print(f"  {i}. {Path(file_path).name}")

                    # Confirm processing
                    confirm = input(f"\nProceed with processing {len(text_files)} files? (y/n, default: y): ").strip().lower()
                    if confirm != 'n':
                        stats = rag_system.add_documents_from_texts(text_files)

                        print("\n✅ Processing completed!")
                        print(f"📄 Files processed: {stats['files_processed']}")
                        print(f"📝 Chunks created: {stats['chunks_created']}")
                        print(f"🧠 Embeddings generated: {stats['embeddings_generated']}")
                        print(f"⏱️ Processing time: {stats['processing_time']:.2f} seconds")

                        if stats['errors']:
                            print(f"⚠️ Errors encountered: {len(stats['errors'])}")
                            show_errors = input("Show error details? (y/n): ").strip().lower()
                            if show_errors == 'y':
                                for error in stats['errors']:
                                    print(f"  - {error}")
                    else:
                        print("❌ Processing cancelled")
                else:
                    print("❌ No text files provided!")

            elif choice == "2":
                # Add single text
                print("\n📝 Add Single Text Content:")
                print("💡 Options:")
                print("1. Type/paste text directly")
                print("2. Read from a text file")

                text_method = input("Choose method (1/2, default: 1): ").strip()

                text_content = ""

                if text_method == "2":
                    # Read from file
                    file_path = input("Enter path to text file: ").strip()
                    if file_path:
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                text_content = f.read().strip()
                            print(f"✅ Read {len(text_content)} characters from {Path(file_path).name}")
                        except Exception as e:
                            print(f"❌ Error reading file: {e}")
                            continue
                else:
                    # Direct text input
                    print("\nEnter your text content:")
                    print("💡 Tip: You can paste large text blocks here")
                    text_content = input("Text: ").strip()

                if text_content:
                    # Get source name
                    default_name = f"direct_input_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    source_name = input(f"\nSource name (default: {default_name}): ").strip()
                    if not source_name:
                        source_name = default_name

                    # Show preview
                    preview = text_content[:200] + "..." if len(text_content) > 200 else text_content
                    print(f"\n📖 Text Preview:\n{'-'*40}\n{preview}\n{'-'*40}")
                    print(f"📊 Total length: {len(text_content)} characters, ~{len(text_content.split())} words")

                    confirm = input("\nProceed with processing? (y/n, default: y): ").strip().lower()
                    if confirm != 'n':
                        print("\n🔄 Processing text...")
                        stats = rag_system.add_document_from_text(text_content, source_name)

                        print("✅ Text added successfully!")
                        print(f"📝 Chunks created: {stats['chunks_created']}")
                        print(f"🧠 Embeddings generated: {stats['embeddings_generated']}")
                        print(f"⏱️ Processing time: {stats['processing_time']:.2f} seconds")
                    else:
                        print("❌ Processing cancelled")
                else:
                    print("❌ No text content provided!")

            elif choice == "3":
                # Query system
                if not rag_system.documents:
                    print("❌ No documents in system! Add some documents first.")
                    continue

                print("\n🔍 Query System:")
                query_text = input("Enter your query: ").strip()
                if not query_text:
                    print("❌ No query provided!")
                    continue

                # Query parameters
                top_k = input("Number of results (default: 5): ").strip()
                try:
                    top_k = int(top_k) if top_k else 5
                except ValueError:
                    top_k = 5

                print("Search methods:")
                print("1. Semantic (default)")
                print("2. Cosine similarity")
                print("3. FAISS (if available)")

                method_choice = input("Choose search method (1-3, default: 1): ").strip()
                method_map = {"1": "semantic", "2": "cosine", "3": "faiss"}
                search_method = method_map.get(method_choice, "semantic")

                print(f"\n🔄 Searching with {search_method} method...")
                result = rag_system.query(query_text, top_k=top_k, search_method=search_method)

                print(f"\n🎯 Query Results ({len(result.results)}):")
                print(f"⏱️ Processing time: {result.processing_time:.3f}s")
                print(f"🔍 Search method: {result.search_method}")
                print("-" * 80)

                if result.results:
                    for i, res in enumerate(result.results, 1):
                        print(f"\n{i}. 📊 Similarity: {res['similarity']:.3f}")
                        print(f"   📄 Source: {Path(res['source_file']).name}")
                        print(f"   📝 Text: {res['text'][:200]}...")

                        # Show more if requested
                        if i <= 3:  # Show full text for top 3 results
                            show_full = input(f"   👀 Show full text for result {i}? (y/n): ").strip().lower()
                            if show_full == 'y':
                                print(f"   📖 Full text:\n{'-'*60}\n{res['text']}\n{'-'*60}")
                else:
                    print("❌ No relevant results found!")

                # Analysis option
                analyze = input("\n📈 Perform similarity analysis? (y/n): ").strip().lower()
                if analyze == 'y':
                    analysis = rag_system.similarity_analysis(query_text, top_k=top_k)
                    print(f"\n📊 Similarity Analysis:")
                    print(f"   Mean similarity: {analysis['similarity_distribution']['mean']:.3f}")
                    print(f"   Max similarity: {analysis['similarity_distribution']['max']:.3f}")
                    print(f"   Source distribution: {analysis['source_distribution']}")

            elif choice == "4":
                # Show statistics
                print("\n📊 System Statistics:")
                stats = rag_system.get_statistics()

                print(f"  📈 Status: {stats.get('status', 'Unknown')}")
                print(f"  📄 Total documents: {stats['total_documents']}")
                print(f"  📝 Total chunks: {stats['total_chunks']}")
                print(f"  🧠 Embedding dimension: {stats['embedding_dimension']}")
                print(f"  🤖 Model: {stats['model_name']}")
                print(f"  💾 Storage: {stats['storage_directory']}")
                print(f"  🔍 FAISS available: {stats['faiss_available']}")
                print(f"  📈 FAISS index built: {stats['faiss_index_built']}")

                if stats['document_sources']:
                    print(f"  📂 Document sources:")
                    for source, count in stats['document_sources'].items():
                        print(f"    - {Path(source).name}: {count} chunks")

                text_stats = stats['text_statistics']
                print(f"  📝 Text statistics:")
                print(f"    - Total words: {text_stats['total_words']:,}")
                print(f"    - Total characters: {text_stats['total_characters']:,}")
                print(f"    - Avg words per chunk: {text_stats['avg_words_per_chunk']:.1f}")
                print(f"    - Avg chars per chunk: {text_stats['avg_chars_per_chunk']:.1f}")

            elif choice == "5":
                # Visualize embeddings
                if not rag_system.documents:
                    print("❌ No documents in system! Add some documents first.")
                    continue

                print("\n🎨 Visualize Embeddings:")
                print("Visualization methods:")
                print("1. t-SNE (default)")
                print("2. PCA")
                print("3. UMAP (if available)")

                viz_choice = input("Choose method (1-3, default: 1): ").strip()
                viz_map = {"1": "tsne", "2": "pca", "3": "umap"}
                viz_method = viz_map.get(viz_choice, "tsne")

                sample_size = input("Sample size (default: 1000): ").strip()
                try:
                    sample_size = int(sample_size) if sample_size else 1000
                except ValueError:
                    sample_size = 1000

                print(f"\n🔄 Creating {viz_method} visualization...")
                viz_path = rag_system.visualize_embeddings(
                    method=viz_method,
                    sample_size=sample_size
                )

                if viz_path:
                    print(f"✅ Visualization saved to: {viz_path}")
                else:
                    print("❌ Visualization failed!")

            elif choice == "6":
                # Export embeddings
                if not rag_system.documents:
                    print("❌ No documents in system! Add some documents first.")
                    continue

                print("\n💾 Export Embeddings:")
                print("Export formats:")
                print("1. CSV (default)")
                print("2. JSON")
                print("3. Parquet")

                format_choice = input("Choose format (1-3, default: 1): ").strip()
                format_map = {"1": "csv", "2": "json", "3": "parquet"}
                export_format = format_map.get(format_choice, "csv")

                export_path = input(f"Export file path (leave empty for auto-generated): ").strip()
                if not export_path:
                    export_path = None

                print(f"\n🔄 Exporting in {export_format} format...")
                try:
                    export_path = rag_system.export_embeddings(
                        export_path=export_path,
                        format=export_format
                    )
                    print(f"✅ Export completed: {export_path}")
                except Exception as e:
                    print(f"❌ Export failed: {e}")

            elif choice == "7":
                # Clear data
                confirm = input("\n⚠️ Clear all data? This cannot be undone! (yes/no): ").strip().lower()
                if confirm == "yes":
                    rag_system.clear_storage()
                    print("✅ All data cleared!")
                else:
                    print("❌ Clear operation cancelled.")

            elif choice == "8":
                # Exit
                print("\n👋 Goodbye!")
                break

            else:
                print("❌ Invalid choice! Please select 1-8.")

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
