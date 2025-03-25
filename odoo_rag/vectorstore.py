import os
import json
from typing import Dict, List, Optional, Any
import logging
import chromadb
from chromadb.utils import embedding_functions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OdooVectorStore:
    """Vector store for Odoo code and documentation using ChromaDB directly"""
    
    def __init__(self, persist_directory: str = "chroma_db", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the vector store
        
        Args:
            persist_directory: Directory to persist the vector store
            model_name: Name of the embedding model to use
        """
        self.persist_directory = persist_directory
        self.model_name = model_name
        
        # Initialize embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        
        # Initialize storage
        self._ensure_directory_exists(persist_directory)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="odoo_code",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
    
    def _ensure_directory_exists(self, directory: str) -> None:
        """Ensure that the specified directory exists"""
        os.makedirs(directory, exist_ok=True)
    
    def add_chunks(self, chunks: List[Dict]) -> None:
        """
        Add chunks to the vector store
        
        Args:
            chunks: List of chunks to add
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            # Generate a unique ID
            chunk_id = f"chunk_{self.collection.count() + i}"
            
            # Extract content and metadata
            content = chunk['content']
            metadata = chunk['metadata']
            
            ids.append(chunk_id)
            documents.append(content)
            metadatas.append(metadata)
        
        # Add documents to ChromaDB
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(chunks)} chunks to vector store")
    
    def search(self, query: str, filter: Optional[Dict] = None, k: int = 5) -> List[Dict]:
        """
        Search for similar documents
        
        Args:
            query: Query string
            filter: Optional filter to apply to the search
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        # Construct filter query if needed
        where_clause = filter if filter else None
        
        # Query ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where_clause
        )
        
        # Process results
        docs = []
        if len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                docs.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "id": results['ids'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else None
                })
        
        return docs
    
    def search_by_module(self, query: str, module_name: str, k: int = 5) -> List[Dict]:
        """
        Search within a specific module
        
        Args:
            query: Query string
            module_name: Name of the module to search within
            k: Number of results to return
            
        Returns:
            List of similar documents from the specified module
        """
        return self.search(
            query=query,
            filter={"module": module_name},
            k=k
        )
    
    def search_by_type(self, query: str, content_type: str, k: int = 5) -> List[Dict]:
        """
        Search for a specific type of content
        
        Args:
            query: Query string
            content_type: Type of content to search for (model, view, manifest)
            k: Number of results to return
            
        Returns:
            List of similar documents of the specified type
        """
        return self.search(
            query=query,
            filter={"type": content_type},
            k=k
        )
    
    def search_by_model(self, query: str, model_name: str, k: int = 5) -> List[Dict]:
        """
        Search for content related to a specific model
        
        Args:
            query: Query string
            model_name: Technical name of the model (e.g. 'res.partner')
            k: Number of results to return
            
        Returns:
            List of similar documents related to the specified model
        """
        # First try to find exact matches
        exact_matches = self.search(
            query=query,
            filter={"model_name": model_name},
            k=k
        )
        
        # If we don't have enough exact matches, broaden the search
        if len(exact_matches) < k:
            # Look for views that reference this model
            view_matches = self.search(
                query=query,
                filter={"type": "view", "model": model_name},
                k=k - len(exact_matches)
            )
            exact_matches.extend(view_matches)
        
        return exact_matches
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            "total_documents": self.collection.count()
        } 