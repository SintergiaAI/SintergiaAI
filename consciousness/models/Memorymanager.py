import pandas as pd
from fastapi import HTTPException
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
import os
from pydantic import BaseModel
from datetime import datetime
import numpy as np
from collections import Counter
from app.controllers.logger_controller import logger
from app.models.EmbeddingCache import EmbeddingCache

class MemoryEntry(BaseModel):
    text: str
    metadata: Dict[str, Any] = {}
    source: str  # e.g., "human_interaction", "social_media", "system"
    timestamp: Optional[datetime] = None

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    diversity_threshold: float = 0.7  # Controla qué tan diversos queremos los resultados
    
class DiversityMetrics(BaseModel):
    source_distribution: Dict[str, float]
    time_distribution: Dict[str, float]
    entropy_score: float

class MemoryManager:
    def __init__(self, 
                 index_name: str = "sintergia-memory", 
                 dimension: int = 1536,):
        self.index_name = index_name
        self.dimension = dimension
        self.embedding_cache = EmbeddingCache.get_instance()  # Inicializar el cache
        self.pc = None
        self.index = None

        try:
            self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY","pcsk_5Be6QE_HHgb4j6zkxVfL9hGmnLG3dxAnzC7TcuTG7YPrMkMFu8efcZyvJMLzZ97SqduDnh"))
            self._initialize_index()
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")

    def _initialize_index(self):
        """Initialize Pinecone index with better error handling"""
        try:
            existing_indexes = self.pc.list_indexes()
            
            if self.index_name not in existing_indexes:
                try:
                    self.pc.create_index(
                        name=self.index_name,
                        dimension=self.dimension,
                        metric="cosine",
                        spec=ServerlessSpec()
                    )
                except Exception as e:
                    logger.error(f"Failed to create index: {str(e)}")
                    return False
            
            self.index = self.pc.Index(self.index_name)
            logger.info("Successfully initialized Pinecone index")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone index: {str(e)}")
            return False

    async def add_to_memory(self, entry: MemoryEntry) -> bool:
        try:
            if not entry.timestamp:
                entry.timestamp = datetime.now()
            
            cached_embedding = self.embedding_cache.get(entry.text, "multilingual-e5-large")

            if cached_embedding:
                embedding_values = cached_embedding
            else:
                # Si no está en cache, generar nuevo embedding
                embedding = self.pc.inference.embed(
                    model="multilingual-e5-large",
                    inputs=[entry.text],
                    parameters={"input_type": "passage", "truncate": "END"}
                )
                embedding_values = embedding[0].values
                # Guardar en cache
                self.embedding_cache.set(entry.text, "multilingual-e5-large", embedding_values)
            
            # Upsert en Pinecone
            self.index.upsert(
                vectors=[{
                    "id": str(hash(f"{entry.text}{entry.timestamp}")),
                    "values": embedding_values,
                    "metadata": {
                        "text": entry.text,
                        "source": entry.source,
                        "timestamp": entry.timestamp.isoformat(),
                        **entry.metadata
                    }
                }]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add to memory: {str(e)}")
            return False

    async def query_memory(self, query_request: QueryRequest) -> List[dict]:
        """Query memory with fallback to cache-only if Pinecone fails"""
        try:
            if not self.index:  # Si no hay conexión a Pinecone
                logger.warning("Pinecone not available, using cache only")
                return []

            cached_embedding = self.embedding_cache.get(
                query_request.query, 
                "multilingual-e5-large"
            )
            
            if cached_embedding:
                query_vector = cached_embedding
            else:
                if not self.pc:
                    return []
                    
                query_embedding = self.pc.inference.embed(
                    model="multilingual-e5-large",
                    inputs=[query_request.query],
                    parameters={"input_type": "query"}
                )
                query_vector = query_embedding[0].values
                self.embedding_cache.set(
                    query_request.query, 
                    "multilingual-e5-large", 
                    query_vector
                )
            
            response = self.index.query(
                vector=query_vector,
                top_k=query_request.top_k,
                include_values=True,
                include_metadata=True
            )
            
            return response.matches if response else []
            
        except Exception as e:
            logger.error(f"Failed to query memory: {str(e)}")
            return []
