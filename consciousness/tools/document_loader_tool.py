 fastapi import FastAPI
from pinecone import Pinecone
from typing import List, Dict
import re

app = FastAPI()
pc = Pinecone(api_key="YOUR_API_KEY")

class DocumentIndexer:
    def __init__(self):
        self.index_name = "docs"
        self._initialize_index()
        self.index = pc.Index(self.index_name)

    def _initialize_index(self):
        if not pc.has_index(self.index_name):
            pc.create_index(
                name=self.index_name,
                dimension=1024,
                metric="cosine"
            )

    def _split_document(self, content: str) -> List[Dict[str, str]]:
        """
        """
        sections = []
        current_section = {"title": "", "content": []}
        
        lines = content.split('\n')
        for line in lines:
            if line.startswith('## '):
                # Si tenemos una sección anterior, la guardamos
                if current_section["title"]:
                    sections.append(current_section)
                current_section = {
                    "title": line.replace('## ', '').strip(),
                    "content": []
                }
            elif line.startswith('- '):
                current_section["content"].append(line.replace('- ', '').strip())
            else:
                if line.strip():
                    current_section["content"].append(line.strip())
        
        if current_section["title"]:
            sections.append(current_section)
            
        return sections

    def index_document(self, content: str, doc_type: str):
        """
        """
        sections = self._split_document(content)
        vectors = []
        
        for i, section in enumerate(sections):
            section_text = f"""
            Documento: {doc_type}
            Sección: {section['title']}
            Contenido: {' '.join(section['content'])}
            """
            
            embedding = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[section_text],
                parameters={"input_type": "passage"}
            )
            
            vector = {
                "id": f"{doc_type}-section-{i}",
                "values": embedding[0].values,
                "metadata": {
                    "doc_type": doc_type,
                    "section_title": section['title'],
                    "content": section['content']
                }
            }
            vectors.append(vector)
        
        # Upsert vectors
        self.index.upsert(vectors=vectors)
        return len(vectors)