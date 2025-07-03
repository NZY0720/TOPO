#!/usr/bin/env python3
"""
Evaluation metrics for power grid topology reconstruction
"""

import torch
import numpy as np
import networkx as nx
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class TopologyMetrics:
    """拓扑重建评估指标"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置指标"""
        self.predictions = []
        self.targets = []
        
    def update(self, predicted_edges: torch.Tensor, true_edges: torch.Tensor, 
               num_nodes: int):
        """更新指标"""
        # 转换为邻接矩阵
        pred_adj = self._edges_to_adjacency(predicted_edges, num_nodes)
        true_adj = self._edges_to_adjacency(true_edges, num_nodes)
        
        self.predictions.extend(pred_adj.flatten().tolist())
        self.targets.extend(true_adj.flatten().tolist())
    
    def compute(self, predicted_edges: torch.Tensor, true_edges: torch.Tensor,
                num_nodes: int) -> Dict[str, float]:
        """计算所有拓扑指标"""
        # 转换为邻接矩阵
        pred_adj = self._edges_to_adjacency(predicted_edges, num_nodes)
        true_adj = self._edges_to_adjacency(true_edges, num_nodes)
        
        # 展平为向量
        pred_flat = pred_adj.flatten().numpy()
        true_flat = true_adj.flatten().numpy()
        
        # 基础分类指标
        metrics = {}
        
        if len(np.unique(true_flat)) > 1:  # 确保有正负样本
            metrics['f1_score'] = f1_score(true_flat, pred_flat, zero_division=0)
            metrics['precision'] = precision_score(true_flat, pred_flat, zero_division=0)
            metrics['recall'] = recall_score(true_flat, pred_flat, zero_division=0)
            metrics['accuracy'] = accuracy_score(true_flat, pred_flat)
        else:
            metrics['f1_score'] = 0.0
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
            metrics['accuracy'] = 0.0
        
        # 图结构指标
        metrics.update(self._compute_graph_metrics(predicted_edges, true_edges, num_nodes))
        
        # 拓扑特异性指标
        metrics.update(self._compute_topology_specific_metrics(pred_adj, true_adj))
        
        return metrics
    
    def _edges_to_adjacency(self, edges: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """将边列表转换为邻接矩阵"""
        adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
        
        if edges.size(1) > 0:
            for i in range(edges.size(1)):
                src, dst = edges[0, i].item(), edges[1, i].item()
                if 0 <= src < num_nodes and 0 <= dst < num_nodes:
                    adj[src, dst] = 1.0
                    adj[dst, src] = 1.0  # 无向图
        
        return adj
    
    def _compute_graph_metrics(self, predicted_edges: torch.Tensor, 
                              true_edges: torch.Tensor, num_nodes: int) -> Dict[str, float]:
        """计算图结构指标"""
        metrics = {}
        
        # 边数对比
        metrics['num_predicted_edges'] = predicted_edges.size(1)
        metrics['num_true_edges'] = true_edges.size(1)
        metrics['edge_count_error'] = abs(predicted_edges.size(1) - true_edges.size(1))
        
        # 连通性分析
        if predicted_edges.size(1) > 0:
            pred_graph = self._edges_to_networkx(predicted_edges, num_nodes)
            metrics['predicted_components'] = nx.number_connected_components(pred_graph)
            metrics['predicted_is_connected'] = nx.is_connected(pred_graph)
            
            # 如果是连通的，计算图的性质
            if nx.is_connected(pred_graph):
                metrics['predicted_diameter'] = nx.diameter(pred_graph)
                metrics['predicted_avg_clustering'] = nx.average_clustering(pred_graph)
                metrics['predicted_avg_path_length'] = nx.average_shortest_path_length(pred_graph)
            else:
                metrics['predicted_diameter'] = float('inf')
                metrics['predicted_avg_clustering'] = 0.0
                metrics['predicted_avg_path_length'] = float('inf')
        else:
            metrics['predicted_components'] = num_nodes
            metrics['predicted_is_connected'] = False
            metrics['predicted_diameter'] = float('inf')
            metrics['predicted_avg_clustering'] = 0.0
            metrics['predicted_avg_path_length'] = float('inf')
        
        # 真实图的性质
        if true_edges.size(1) > 0:
            true_graph = self._edges_to_networkx(true_edges, num_nodes)
            metrics['true_components'] = nx.number_connected_components(true_graph)
            metrics['true_is_connected'] = nx.is_connected(true_graph)
            
            if nx.is_connected(true_graph):
                metrics['true_diameter'] = nx.diameter(true_graph)
                metrics['true_avg_clustering'] = nx.average_clustering(true_graph)
                metrics['true_avg_path_length'] = nx.average_shortest_path_length(true_graph)
        
        return metrics
    
    def _edges_to_networkx(self, edges: torch.Tensor, num_nodes: int) -> nx.Graph:
        """将边列表转换为NetworkX图"""
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        
        if edges.size(1) > 0:
            edge_list = edges.t().numpy().tolist()
            G.add_edges_from(edge_list)
        
        return G
    
    def _compute_topology_specific_metrics(self, pred_adj: torch.Tensor, 
                                         true_adj: torch.Tensor) -> Dict[str, float]:
        """计算拓扑特异性指标"""
        metrics = {}
        
        # 度分布比较
        pred_degrees = pred_adj.sum(dim=1).numpy()
        true_degrees = true_adj.sum(dim=1).numpy()
        
        metrics['degree_mse'] = mean_squared_error(true_degrees, pred_degrees)
        metrics['degree_mae'] = mean_absolute_error(true_degrees, pred_degrees)
        
        if len(np.unique(true_degrees)) > 1:
            metrics['degree_correlation'] = np.corrcoef(true_degrees, pred_degrees)[0, 1]
            if np.isnan(metrics['degree_correlation']):
                metrics['degree_correlation'] = 0.0
        else:
            metrics['degree_correlation'] = 0.0
        
        # 径向性指标（对于配电网）
        metrics['predicted_is_tree'] = self._is_tree(pred_adj)
        metrics['true_is_tree'] = self._is_tree(true_adj)
        
        # 稀疏性
        total_possible_edges = true_adj.size(0) * (true_adj.size(0) - 1) / 2
        metrics['predicted_sparsity'] = pred_adj.sum().item() / (2 * total_possible_edges)
        metrics['true_sparsity'] = true_adj.sum().item() / (2 * total_possible_edges)
        
        return metrics
    
    def _is_tree(self, adj: torch.Tensor) -> bool:
        """检查是否为树结构"""
        num_nodes = adj.size(0)
        num_edges = adj.sum().item() / 2  # 无向图
        
        # 树的必要条件：边数 = 节点数 - 1
        if abs(num_edges - (num_nodes - 1)) > 0.5:
            return False
        
        # 检查连通性
        edges = torch.nonzero(torch.triu(adj), as_tuple=False)
        if edges.size(0) == 0:
            return num_nodes <= 1
        
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edges.numpy().tolist())
        
        return nx.is_connected(G) and nx.is_tree(G)


class ParameterMetrics:
    """线路参数评估指标"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置指标"""
        self.predictions = []
        self.targets = []
    
    def update(self, predicted_params: torch.Tensor, true_params: torch.Tensor):
        """更新指标"""
        min_len = min(predicted_params.size(0), true_params.size(0))
        if min_len > 0:
            self.predictions.extend(predicted_params[:min_len].numpy().tolist())
            self.targets.extend(true_params[:min_len].numpy().tolist())
    
    def compute(self, predicted_params: torch.Tensor, 
                true_params: torch.Tensor) -> Dict[str, float]:
        """计算参数预测指标"""
        min_len = min(predicted_params.size(0), true_params.size(0))
        
        if min_len == 0:
            return {
                'param_mae_r': 0.0,
                'param_mae_x': 0.0,
                'param_mse_r': 0.0,
                'param_mse_x': 0.0,
                'param_r2_r': 0.0,
                'param_r2_x': 0.0,
                'param_mape_r': 0.0,
                'param_mape_x': 0.0
            }
        
        pred = predicted_params[:min_len].numpy()
        true = true_params[:min_len].numpy()
        
        metrics = {}
        
        # 分别计算R和X的指标
        for i, param_name in enumerate(['r', 'x']):
            pred_param = pred[:, i]
            true_param = true[:, i]
            
            # 基础回归指标
            metrics[f'param_mae_{param_name}'] = mean_absolute_error(true_param, pred_param)
            metrics[f'param_mse_{param_name}'] = mean_squared_error(true_param, pred_param)
            
            # R²分数
            if len(np.unique(true_param)) > 1:
                metrics[f'param_r2_{param_name}'] = r2_score(true_param, pred_param)
            else:
                metrics[f'param_r2_{param_name}'] = 0.0
            
            # 平均绝对百分比误差
            mape = self._compute_mape(true_param, pred_param)
            metrics[f'param_mape_{param_name}'] = mape
        
        # 综合指标
        metrics['param_mae_avg'] = (metrics['param_mae_r'] + metrics['param_mae_x']) / 2
        metrics['param_mse_avg'] = (metrics['param_mse_r'] + metrics['param_mse_x']) / 2
        metrics['param_r2_avg'] = (metrics['param_r2_r'] + metrics['param_r2_x']) / 2
        
        # 参数比值指标（X/R比）
        if np.all(pred[:, 0] > 1e-6) and np.all(true[:, 0] > 1e-6):
            pred_ratio = pred[:, 1] / pred[:, 0]
            true_ratio = true[:, 1] / true[:, 0]
            metrics['xr_ratio_mae'] = mean_absolute_error(true_ratio, pred_ratio)
        else:
            metrics['xr_ratio_mae'] = 0.0
        
        return metrics
    
    def _compute_mape(self, y_true: np.ndarray, y_pred: np.ndarray, 
                     epsilon: float = 1e-6) -> float:
        """计算平均绝对百分比误差"""
        y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
        return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100


class PhysicsMetrics:
    """物理约束评估指标"""
    
    def __init__(self):
        pass
    
    def compute_kcl_violations(self, node_features: torch.Tensor, 
                              edge_index: torch.Tensor,
                              edge_probs: torch.Tensor, 
                              edge_params: torch.Tensor) -> Dict[str, float]:
        """计算基尔霍夫电流定律违反程度"""
        n_nodes = node_features.size(0)
        violations = []
        
        for node in range(n_nodes):
            current_sum = 0.0
            
            # 计算节点的电流平衡
            src_mask = edge_index[0] == node
            dst_mask = edge_index[1] == node
            
            if src_mask.any():
                for idx in torch.where(src_mask)[0]:
                    current = self._compute_current(
                        node_features[node], 
                        node_features[edge_index[1, idx]],
                        edge_params[idx]
                    )
                    current_sum -= current * edge_probs[idx].item()
            
            if dst_mask.any():
                for idx in torch.where(dst_mask)[0]:
                    current = self._compute_current(
                        node_features[edge_index[0, idx]],
                        node_features[node],
                        edge_params[idx]
                    )
                    current_sum += current * edge_probs[idx].item()
            
            violations.append(abs(current_sum))
        
        return {
            'kcl_max_violation': max(violations) if violations else 0.0,
            'kcl_avg_violation': np.mean(violations) if violations else 0.0,
            'kcl_num_violations': sum(1 for v in violations if v > 0.1)
        }
    
    def compute_voltage_violations(self, node_features: torch.Tensor) -> Dict[str, float]:
        """计算电压约束违反"""
        # 计算电压幅值
        voltage_magnitudes = torch.sqrt(
            node_features[:, 0]**2 + node_features[:, 1]**2
        ).numpy()
        
        # 标幺值约束范围
        lower_bound = 0.95
        upper_bound = 1.05
        
        lower_violations = np.maximum(0, lower_bound - voltage_magnitudes)
        upper_violations = np.maximum(0, voltage_magnitudes - upper_bound)
        
        return {
            'voltage_lower_violations': np.sum(lower_violations > 0),
            'voltage_upper_violations': np.sum(upper_violations > 0),
            'voltage_max_violation': max(np.max(lower_violations), np.max(upper_violations)),
            'voltage_avg_magnitude': np.mean(voltage_magnitudes)
        }
    
    def _compute_current(self, v_from: torch.Tensor, v_to: torch.Tensor,
                        params: torch.Tensor) -> float:
        """计算电流幅值"""
        v_from_complex = complex(v_from[0].item(), v_from[1].item())
        v_to_complex = complex(v_to[0].item(), v_to[1].item())
        
        impedance = complex(params[0].item(), params[1].item())
        
        if abs(impedance) < 1e-6:
            return 0.0
        
        current_complex = (v_from_complex - v_to_complex) / impedance
        return abs(current_complex)


class ComprehensiveMetrics:
    """综合评估指标"""
    
    def __init__(self):
        self.topology_metrics = TopologyMetrics()
        self.parameter_metrics = ParameterMetrics()
        self.physics_metrics = PhysicsMetrics()
    
    def compute_all(self, predictions: Dict[str, torch.Tensor],
                   targets: Dict[str, torch.Tensor],
                   data) -> Dict[str, Union[float, Dict[str, float]]]:
        """计算所有评估指标"""
        all_metrics = {}
        
        # 拓扑指标
        if 'edge_index' in predictions and 'true_edge_index' in targets:
            topo_metrics = self.topology_metrics.compute(
                predictions['edge_index'],
                targets['true_edge_index'],
                data.num_nodes
            )
            all_metrics['topology'] = topo_metrics
        
        # 参数指标
        if ('edge_params' in predictions and 'true_edge_params' in targets and
            targets['true_edge_params'].size(0) > 0):
            param_metrics = self.parameter_metrics.compute(
                predictions['edge_params'],
                targets['true_edge_params']
            )
            all_metrics['parameters'] = param_metrics
        
        # 物理指标
        if 'edge_probs' in predictions:
            physics_metrics = {}
            
            # KCL违反
            kcl_metrics = self.physics_metrics.compute_kcl_violations(
                data.x, data.edge_index, 
                predictions['edge_probs'], predictions['edge_params']
            )
            physics_metrics.update(kcl_metrics)
            
            # 电压违反
            voltage_metrics = self.physics_metrics.compute_voltage_violations(data.x)
            physics_metrics.update(voltage_metrics)
            
            all_metrics['physics'] = physics_metrics
        
        # 计算综合得分
        all_metrics['composite_score'] = self._compute_composite_score(all_metrics)
        
        return all_metrics
    
    def _compute_composite_score(self, metrics: Dict[str, Union[float, Dict[str, float]]]) -> float:
        """计算综合评估得分"""
        score = 0.0
        weight_sum = 0.0
        
        # 拓扑得分 (权重: 0.5)
        if 'topology' in metrics:
            topo_score = metrics['topology'].get('f1_score', 0.0)
            score += 0.5 * topo_score
            weight_sum += 0.5
        
        # 参数得分 (权重: 0.3)
        if 'parameters' in metrics:
            param_r2 = metrics['parameters'].get('param_r2_avg', 0.0)
            param_score = max(0.0, param_r2)  # R²可能为负
            score += 0.3 * param_score
            weight_sum += 0.3
        
        # 物理得分 (权重: 0.2)
        if 'physics' in metrics:
            # 基于违反数量的惩罚得分
            max_kcl = metrics['physics'].get('kcl_max_violation', 0.0)
            max_voltage = metrics['physics'].get('voltage_max_violation', 0.0)
            
            physics_score = 1.0 / (1.0 + max_kcl + max_voltage)
            score += 0.2 * physics_score
            weight_sum += 0.2
        
        return score / weight_sum if weight_sum > 0 else 0.0