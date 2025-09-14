from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self):
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    async def add_documents(self, document_id: str, text_chunks: List[str], filename: str, metadata: Optional[Dict] = None):
        """
        Add document chunks to the vector store
        """
        try:
            # Create embeddings for text chunks
            embeddings = self.embedding_model.encode(text_chunks).tolist()
            
            # Create unique IDs for each chunk
            chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(text_chunks))]
            
            # Create metadata for each chunk
            base_metadata = {
                "document_id": document_id,
                "filename": filename,
            }
            
            # Add additional metadata if provided
            if metadata:
                base_metadata.update(metadata)
            
            metadatas = [
                {
                    **base_metadata,
                    "chunk_index": i,
                    "text": chunk
                }
                for i, chunk in enumerate(text_chunks)
            ]
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=text_chunks,
                metadatas=metadatas,
                ids=chunk_ids
            )
            
        except Exception as e:
            raise Exception(f"Error adding documents to vector store: {str(e)}")
    
    async def search_similar(self, query: str, n_results: int = 5, document_ids: Optional[List[str]] = None) -> Dict:
        """
        Search for similar documents based on query
        """
        try:
            # Create embedding for query
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Prepare where clause for filtering by document_ids
            where_clause = None
            if document_ids:
                where_clause = {"document_id": {"$in": document_ids}}
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            return results
        
        except Exception as e:
            raise Exception(f"Error searching vector store: {str(e)}")
    
    async def delete_document(self, document_id: str):
        """
        Delete all chunks of a document from the vector store
        """
        try:
            # Get all chunk IDs for this document
            results = self.collection.get(
                where={"document_id": document_id},
                include=["metadatas"]
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
        
        except Exception as e:
            raise Exception(f"Error deleting document from vector store: {str(e)}")
    
    async def list_documents(self) -> List[Dict]:
        """
        List all documents in the vector store
        """
        try:
            # Get all documents
            results = self.collection.get(include=["metadatas"])
            
            # Extract unique documents
            documents = {}
            for metadata in results['metadatas']:
                doc_id = metadata['document_id']
                if doc_id not in documents:
                    documents[doc_id] = {
                        "document_id": doc_id,
                        "filename": metadata['filename'],
                        "chunks_count": 1,
                        "processing_method": metadata.get("processing_method", "standard"),
                        "total_pages": metadata.get("total_pages")
                    }
                else:
                    documents[doc_id]["chunks_count"] += 1
            
            return list(documents.values())
        
        except Exception as e:
            raise Exception(f"Error listing documents: {str(e)}")
    
    def reset_collection(self):
        """
        Reset the entire collection (for testing purposes)
        """
        try:
            self.client.reset()
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            raise Exception(f"Error resetting collection: {str(e)}")
