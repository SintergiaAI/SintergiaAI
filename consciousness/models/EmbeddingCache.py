from typing import Dict, Optional
import hashlib
from datetime import datetime, timedelta
import json

class EmbeddingCache:
    _instance = None
    
    def __init__(self):
        self._cache: Dict[str, dict] = {}
        self._last_cleanup = datetime.now()
        self.max_cache_age = timedelta(hours=24)  # Cache por 24 horas
        self.max_cache_size = 10000  # Máximo número de embeddings en cache
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _generate_key(self, text: str, model: str) -> str:
        """Genera una key única para el texto y modelo"""
        content = f"{text}:{model}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, text: str, model: str) -> Optional[list]:
        """Obtiene embedding del cache"""
        key = self._generate_key(text, model)
        if key in self._cache:
            entry = self._cache[key]
            if datetime.now() - entry['timestamp'] < self.max_cache_age:
                return entry['embedding']
            del self._cache[key]
        return None
    
    def set(self, text: str, model: str, embedding: list):
        """Guarda embedding en cache"""
        # Limpieza automática si el cache está lleno
        if len(self._cache) >= self.max_cache_size:
            self._cleanup()
            
        key = self._generate_key(text, model)
        self._cache[key] = {
            'embedding': embedding,
            'timestamp': datetime.now()
        }
    
    def _cleanup(self):
        """Limpia entradas antiguas del cache"""
        current_time = datetime.now()
        self._cache = {
            k: v for k, v in self._cache.items()
            if current_time - v['timestamp'] < self.max_cache_age
        }