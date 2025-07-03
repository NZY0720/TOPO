#!/usr/bin/env python3
"""
Graph utilities for Power Grid Topology Reconstruction
"""

import numpy as np
import torch
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from typing import Tuple, List, Optional, Union, Dict, Any
import warnings
warnings.filterwarnings('ignore')


def create_candidate_graph(coordinates: np.ndarray, k: int = 5, 
                         method: str = 'knn') -> Tuple[np.ndarray, np.ndarray]:
    """
    基于节点坐标创建候选图
    
    Args:
        coordinates: 节点坐标 [N, 2]
        k: 邻居数量
        method: 构建方法 ('knn', 'radius', 'delaunay')
        
    Returns:
        edge_index: 边索引 [2, E]
        edge_distances: 边距离 [E]
    """
    n_nodes = len(coordinates)
    
    if method == 'knn':
        return _create_knn_graph(coordinates, k)
    elif method == 'radius':
        return _create_radius_graph(coordinates, k)  # k作为半径
    elif method == 'delaunay':
        return _create_delaunay_graph(coordinates)
    elif method == 'hybrid':
        return _create_hybrid_graph(coordinates, k)
    else:
        raise ValueError(f"Unsupported method: {method}")


def _create_knn_graph(coordinates: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """创建k-近邻图"""
    n_nodes = len(coordinates)
    
    # 使用sklearn的NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(coordinates)
    distances, indices = nbrs.kneighbors(coordinates)
    
    edges = []
    edge_distances = []
    
    for i in range(n_nodes):
        for j in range(1, len(indices[i])):  # 跳过自己
            neighbor = indices[i][j]
            if i < neighbor:  # 避免重复边
                edges.append([i, neighbor])
                edge_distances.append(distances[i][j])
    
    edge_index = np.array(edges, dtype=np.int64).T
    edge_distances = np.array(edge_distances, dtype=np.float32)
    
    return edge_index, edge_distances


def _create_radius_graph(coordinates: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray]:
    """创建半径图"""
    n_nodes = len(coordinates)
    
    # 计算所有节点间距离
    distances = squareform(pdist(coordinates))
    
    edges = []
    edge_distances = []
    
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if distances[i, j] <= radius:
                edges.append([i, j])
                edge_distances.append(distances[i, j])
    
    if edges:
        edge_index = np.array(edges, dtype=np.int64).T
        edge_distances = np.array(edge_distances, dtype=np.float32)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_distances = np.empty(0, dtype=np.float32)
    
    return edge_index, edge_distances


def _create_delaunay_graph(coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """创建Delaunay三角剖分图"""
    try:
        from scipy.spatial import Delaunay
    except ImportError:
        raise ImportError("需要安装scipy来使用Delaunay三角剖分")
    
    if len(coordinates) < 3:
        return _create_knn_graph(coordinates, min(2, len(coordinates)-1))
    
    tri = Delaunay(coordinates)
    
    edges = set()
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i+1, len(simplex)):
                edge = tuple(sorted([simplex[i], simplex[j]]))
                edges.add(edge)
    
    edges = list(edges)
    edge_distances = []
    
    for i, j in edges:
        dist = np.linalg.norm(coordinates[i] - coordinates[j])
        edge_distances.append(dist)
    
    if edges:
        edge_index = np.array(edges, dtype=np.int64).T
        edge_distances = np.array(edge_distances, dtype=np.float32)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_distances = np.empty(0, dtype=np.float32)
    
    return edge_index, edge_distances


def _create_hybrid_graph(coordinates: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """创建混合图（k-NN + Delaunay）"""
    # 获取k-NN边
    knn_edges, knn_distances = _create_knn_graph(coordinates, k)
    
    # 获取Delaunay边
    try:
        delaunay_edges, delaunay_distances = _create_delaunay_graph(coordinates)
    except:
        delaunay_edges = np.empty((2, 0), dtype=np.int64)
        delaunay_distances = np.empty(0, dtype=np.float32)
    
    # 合并边
    all_edges = set()
    all_distances = {}
    
    # 添加k-NN边
    for i in range(knn_edges.shape[1]):
        edge = tuple(sorted([knn_edges[0, i], knn_edges[1, i]]))
        all_edges.add(edge)
        all_distances[edge] = knn_distances[i]
    
    # 添加Delaunay边
    for i in range(delaunay_edges.shape[1]):
        edge = tuple(sorted([delaunay_edges[0, i], delaunay_edges[1, i]]))
        if edge not in all_edges:
            all_edges.add(edge)
            all_distances[edge] = delaunay_distances[i]
    
    if all_edges:
        edges = list(all_edges)
        edge_index = np.array(edges, dtype=np.int64).T
        edge_distances = np.array([all_distances[edge] for edge in edges], dtype=np.float32)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_distances = np.empty(0, dtype=np.float32)
    
    return edge_index, edge_distances


def ensure_connected_graph(edge_index: np.ndarray, num_nodes: int, 
                         coordinates: Optional[np.ndarray] = None) -> np.ndarray:
    """确保图是连通的"""
    if edge_index.shape[1] == 0:
        # 如果没有边，创建一个简单的路径图
        if num_nodes <= 1:
            return edge_index
        
        path_edges = []
        for i in range(num_nodes - 1):
            path_edges.append([i, i + 1])
        
        return np.array(path_edges, dtype=np.int64).T
    
    # 检查连通性
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    edge_list = edge_index.T.tolist()
    G.add_edges_from(edge_list)
    
    if nx.is_connected(G):
        return edge_index
    
    # 如果不连通，添加边使其连通
    components = list(nx.connected_components(G))
    additional_edges = []
    
    for i in range(len(components) - 1):
        comp1 = components[i]
        comp2 = components[i + 1]
        
        # 选择距离最近的两个节点相连
        if coordinates is not None:
            min_dist = float('inf')
            best_edge = None
            
            for node1 in comp1:
                for node2 in comp2:
                    dist = np.linalg.norm(coordinates[node1] - coordinates[node2])
                    if dist < min_dist:
                        min_dist = dist
                        best_edge = [node1, node2]
            
            if best_edge:
                additional_edges.append(best_edge)
        else:
            # 随机选择节点连接
            node1 = list(comp1)[0]
            node2 = list(comp2)[0]
            additional_edges.append([node1, node2])
    
    if additional_edges:
        additional_edges = np.array(additional_edges, dtype=np.int64).T
        edge_index = np.concatenate([edge_index, additional_edges], axis=1)
    
    return edge_index


def ensure_radial_topology(edge_index: torch.Tensor, edge_weights: torch.Tensor,
                         num_nodes: int, root_node: int = 0) -> torch.Tensor:
    """确保拓扑为径向（树结构）"""
    if edge_index.size(1) == 0:
        return edge_index
    
    # 转换为NetworkX图
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    # 添加带权重的边
    edges_with_weights = []
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        weight = 1.0 / (edge_weights[i].item() + 1e-6)  # 权重越高，概率越大
        edges_with_weights.append((src, dst, weight))
    
    G.add_weighted_edges_from(edges_with_weights)
    
    # 计算最小生成树
    if G.number_of_edges() > 0:
        try:
            mst = nx.minimum_spanning_tree(G, weight='weight')
            radial_edges = torch.tensor(list(mst.edges()), dtype=torch.long).t()
        except:
            # 如果MST失败，使用简单的BFS生成树
            radial_edges = _bfs_spanning_tree(edge_index, num_nodes, root_node)
    else:
        radial_edges = torch.empty((2, 0), dtype=torch.long)
    
    return radial_edges


def _bfs_spanning_tree(edge_index: torch.Tensor, num_nodes: int, 
                      root_node: int = 0) -> torch.Tensor:
    """使用BFS生成生成树"""
    if edge_index.size(1) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    
    # 构建邻接列表
    adj_list = [[] for _ in range(num_nodes)]
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        adj_list[src].append(dst)
        adj_list[dst].append(src)
    
    # BFS生成生成树
    visited = [False] * num_nodes
    tree_edges = []
    queue = [root_node]
    visited[root_node] = True
    
    while queue:
        current = queue.pop(0)
        
        for neighbor in adj_list[current]:
            if not visited[neighbor]:
                visited[neighbor] = True
                tree_edges.append([current, neighbor])
                queue.append(neighbor)
    
    if tree_edges:
        return torch.tensor(tree_edges, dtype=torch.long).t()
    else:
        return torch.empty((2, 0), dtype=torch.long)


def compute_graph_metrics(edge_index: torch.Tensor, num_nodes: int) -> Dict[str, Any]:
    """计算图的基本指标"""
    if edge_index.size(1) == 0:
        return {
            'num_nodes': num_nodes,
            'num_edges': 0,
            'density': 0.0,
            'is_connected': False,
            'num_components': num_nodes,
            'average_degree': 0.0,
            'is_tree': num_nodes <= 1
        }
    
    # 转换为NetworkX图
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    edge_list = edge_index.t().numpy().tolist()
    G.add_edges_from(edge_list)
    
    metrics = {
        'num_nodes': num_nodes,
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'is_connected': nx.is_connected(G),
        'num_components': nx.number_connected_components(G),
        'average_degree': sum(dict(G.degree()).values()) / num_nodes if num_nodes > 0 else 0,
        'is_tree': nx.is_tree(G)
    }
    
    # 如果连通，计算更多指标
    if nx.is_connected(G) and G.number_of_edges() > 0:
        metrics.update({
            'diameter': nx.diameter(G),
            'average_clustering': nx.average_clustering(G),
            'average_shortest_path_length': nx.average_shortest_path_length(G)
        })
    
    return metrics


def adjacency_to_edge_index(adj_matrix: np.ndarray) -> torch.Tensor:
    """将邻接矩阵转换为edge_index格式"""
    edges = np.nonzero(np.triu(adj_matrix))  # 只取上三角部分（无向图）
    edge_index = torch.tensor(np.stack(edges), dtype=torch.long)
    return edge_index


def edge_index_to_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """将edge_index转换为邻接矩阵"""
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    
    if edge_index.size(1) > 0:
        src, dst = edge_index[0], edge_index[1]
        adj[src, dst] = 1.0
        adj[dst, src] = 1.0  # 无向图
    
    return adj


def find_shortest_paths(edge_index: torch.Tensor, num_nodes: int, 
                       source: int, targets: Optional[List[int]] = None) -> Dict[int, List[int]]:
    """找到从源节点到目标节点的最短路径"""
    if edge_index.size(1) == 0:
        return {}
    
    # 构建NetworkX图
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    edge_list = edge_index.t().numpy().tolist()
    G.add_edges_from(edge_list)
    
    paths = {}
    
    if targets is None:
        targets = list(range(num_nodes))
    
    for target in targets:
        if target != source:
            try:
                path = nx.shortest_path(G, source, target)
                paths[target] = path
            except nx.NetworkXNoPath:
                paths[target] = []
    
    return paths


def compute_node_centralities(edge_index: torch.Tensor, num_nodes: int) -> Dict[str, np.ndarray]:
    """计算节点中心性指标"""
    if edge_index.size(1) == 0:
        return {
            'degree_centrality': np.zeros(num_nodes),
            'betweenness_centrality': np.zeros(num_nodes),
            'closeness_centrality': np.zeros(num_nodes),
            'eigenvector_centrality': np.zeros(num_nodes)
        }
    
    # 构建NetworkX图
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    edge_list = edge_index.t().numpy().tolist()
    G.add_edges_from(edge_list)
    
    centralities = {}
    
    # 度中心性
    degree_cent = nx.degree_centrality(G)
    centralities['degree_centrality'] = np.array([degree_cent.get(i, 0) for i in range(num_nodes)])
    
    # 介数中心性
    if nx.is_connected(G):
        betweenness_cent = nx.betweenness_centrality(G)
        centralities['betweenness_centrality'] = np.array([betweenness_cent.get(i, 0) for i in range(num_nodes)])
        
        # 接近中心性
        closeness_cent = nx.closeness_centrality(G)
        centralities['closeness_centrality'] = np.array([closeness_cent.get(i, 0) for i in range(num_nodes)])
        
        # 特征向量中心性
        try:
            eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000)
            centralities['eigenvector_centrality'] = np.array([eigenvector_cent.get(i, 0) for i in range(num_nodes)])
        except:
            centralities['eigenvector_centrality'] = np.zeros(num_nodes)
    else:
        centralities['betweenness_centrality'] = np.zeros(num_nodes)
        centralities['closeness_centrality'] = np.zeros(num_nodes)
        centralities['eigenvector_centrality'] = np.zeros(num_nodes)
    
    return centralities


def validate_edge_index(edge_index: torch.Tensor, num_nodes: int) -> bool:
    """验证edge_index的有效性"""
    if edge_index.size(0) != 2:
        return False
    
    if edge_index.size(1) == 0:
        return True
    
    # 检查节点索引是否在有效范围内
    if torch.any(edge_index < 0) or torch.any(edge_index >= num_nodes):
        return False
    
    # 检查是否有自环
    if torch.any(edge_index[0] == edge_index[1]):
        return False
    
    return True


def remove_duplicate_edges(edge_index: torch.Tensor) -> torch.Tensor:
    """移除重复边"""
    if edge_index.size(1) == 0:
        return edge_index
    
    # 对于无向图，将边标准化为 (min_node, max_node) 形式
    normalized_edges = torch.stack([
        torch.min(edge_index, dim=0)[0],
        torch.max(edge_index, dim=0)[0]
    ])
    
    # 找到唯一边
    unique_edges, indices = torch.unique(normalized_edges.t(), dim=0, return_inverse=True)
    
    return unique_edges.t().contiguous()


def graph_laplacian(edge_index: torch.Tensor, num_nodes: int, 
                   normalized: bool = True) -> torch.Tensor:
    """计算图拉普拉斯矩阵"""
    # 创建邻接矩阵
    adj = edge_index_to_adjacency(edge_index, num_nodes)
    
    # 计算度矩阵
    degree = torch.sum(adj, dim=1)
    
    if normalized:
        # 标准化拉普拉斯矩阵
        degree_inv_sqrt = torch.pow(degree + 1e-6, -0.5)
        degree_inv_sqrt[degree == 0] = 0
        
        laplacian = torch.eye(num_nodes) - torch.outer(degree_inv_sqrt, degree_inv_sqrt) * adj
    else:
        # 组合拉普拉斯矩阵
        degree_matrix = torch.diag(degree)
        laplacian = degree_matrix - adj
    
    return laplacian