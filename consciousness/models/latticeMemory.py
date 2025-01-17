from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel
from datetime import datetime
import networkx as nx
from collections import defaultdict

class LatticeNode(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any] = {}
    source: str
    timestamp: datetime
    node_type: str  # e.g.
    connections: Set[str] = set() 
    embedding: Optional[List[float]] = None
    
class LatticeEdge(BaseModel):
    source_id: str
    target_id: str
    relation_type: str 
    weight: float = 1.0
    metadata: Dict[str, Any] = {}

class LatticeMemory:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.graph = nx.MultiDiGraph()  # Allows multiple edges between nodes
        self.node_index = {}  # Quick lookup for nodes
        self.relation_types = set()  # Track all relation types
        
    async def add_node(self, entry: MemoryEntry, node_type: str = "concept") -> str:
        """Add a new node to the lattice"""
        node_id = str(hash(f"{entry.text}{entry.timestamp}"))
        
        # Create embedding using existing memory manager
        cached_embedding = self.memory_manager.embedding_cache.get(
            entry.text, 
            "multilingual-e5-large"
        )
        
        if not cached_embedding:
            embedding = self.memory_manager.pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[entry.text],
                parameters={"input_type": "passage", "truncate": "END"}
            )
            cached_embedding = embedding[0].values
            self.memory_manager.embedding_cache.set(
                entry.text,
                "multilingual-e5-large",
                cached_embedding
            )
        
        node = LatticeNode(
            id=node_id,
            text=entry.text,
            metadata=entry.metadata,
            source=entry.source,
            timestamp=entry.timestamp or datetime.now(),
            node_type=node_type,
            embedding=cached_embedding
        )
        
        self.graph.add_node(node_id, **node.dict())
        self.node_index[node_id] = node
        
        # Add to vector store for similarity search
        await self.memory_manager.add_to_memory(entry)
        
        return node_id
        
    def add_edge(self, edge: LatticeEdge):
        """Add a new edge between nodes"""
        if edge.source_id not in self.graph or edge.target_id not in self.graph:
            raise ValueError("Both source and target nodes must exist")
            
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            relation_type=edge.relation_type,
            weight=edge.weight,
            metadata=edge.metadata
        )
        
        # Update connections in nodes
        source_node = self.node_index[edge.source_id]
        target_node = self.node_index[edge.target_id]
        source_node.connections.add(edge.target_id)
        target_node.connections.add(edge.source_id)
        
        self.relation_types.add(edge.relation_type)
        
    async def find_similar_nodes(self, query: str, top_k: int = 5) -> List[LatticeNode]:
        """Find similar nodes using vector similarity"""
        query_request = QueryRequest(query=query, top_k=top_k)
        matches = await self.memory_manager.query_memory(query_request)
        
        similar_nodes = []
        for match in matches:
            node_id = str(hash(f"{match.metadata['text']}{match.metadata['timestamp']}"))
            if node_id in self.node_index:
                similar_nodes.append(self.node_index[node_id])
                
        return similar_nodes
        
    def get_node_neighborhood(self, node_id: str, depth: int = 1) -> nx.MultiDiGraph:
        """Get subgraph of nodes connected to given node up to specified depth"""
        if node_id not in self.graph:
            raise ValueError("Node not found")
            
        neighborhood = nx.ego_graph(self.graph, node_id, radius=depth)
        return neighborhood
        
    def find_paths(self, source_id: str, target_id: str, cutoff: Optional[int] = None) -> List[List[str]]:
        """Find all paths between two nodes, optionally limited by length"""
        try:
            paths = list(nx.all_simple_paths(
                self.graph,
                source=source_id,
                target=target_id,
                cutoff=cutoff
            ))
            return paths
        except nx.NetworkXNoPath:
            return []
            
    def get_node_centrality(self, node_id: str) -> float:
        """Calculate centrality score for a node"""
        if node_id not in self.graph:
            raise ValueError("Node not found")
            
        centrality = nx.eigenvector_centrality_numpy(self.graph)
        return centrality[node_id]
        
    def get_community_structure(self) -> Dict[str, List[str]]:
        """Detect communities in the lattice"""
        communities = nx.community.louvain_communities(self.graph.to_undirected())
        
        community_dict = {}
        for i, community in enumerate(communities):
            community_dict[f"community_{i}"] = list(community)
            
        return community_dict
        
    def prune_weak_connections(self, weight_threshold: float = 0.3):
        """Remove edges with weight below threshold"""
        edges_to_remove = [
            (u, v, k) for u, v, k, d in self.graph.edges(data=True, keys=True)
            if d['weight'] < weight_threshold
        ]
        
        for u, v, k in edges_to_remove:
            self.graph.remove_edge(u, v, k)
            
    def merge_nodes(self, node_ids: List[str], merged_text: str):
        """Merge multiple nodes into a single node"""
        if not all(node_id in self.graph for node_id in node_ids):
            raise ValueError("All nodes must exist in the graph")
            
        # Create new merged node
        merged_metadata = {}
        merged_connections = set()
        sources = set()
        timestamps = []
        
        for node_id in node_ids:
            node = self.node_index[node_id]
            merged_metadata.update(node.metadata)
            merged_connections.update(node.connections)
            sources.add(node.source)
            timestamps.append(node.timestamp)
            
        merged_entry = MemoryEntry(
            text=merged_text,
            metadata=merged_metadata,
            source=", ".join(sources),
            timestamp=max(timestamps)
        )
        
        # Add merged node
        merged_id = self.add_node(merged_entry)
        
        # Transfer connections
        for conn in merged_connections:
            if conn not in node_ids:  # Don't connect to nodes being merged
                self.add_edge(LatticeEdge(
                    source_id=merged_id,
                    target_id=conn,
                    relation_type="merged_connection",
                    weight=1.0
                ))
                
        # Remove original nodes
        for node_id in node_ids:
            self.graph.remove_node(node_id)
            del self.node_index[node_id]
            
        return merged_id