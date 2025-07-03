#!/usr/bin/env python3
"""
Physics-informed loss functions for power grid topology reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Dict, Tuple, Optional
import numpy as np

from ..config.base_config import PhysicsConfig


class KirchhoffCurrentLaw(nn.Module):
    """基尔霍夫电流定律约束"""
    
    def __init__(self, tolerance: float = 0.1):
        super().__init__()
        self.tolerance = tolerance
        
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_probs: torch.Tensor, edge_params: torch.Tensor) -> torch.Tensor:
        """
        计算KCL违反程度
        
        Args:
            node_features: 节点特征 [N, F]
            edge_index: 边索引 [2, E]  
            edge_probs: 边存在概率 [E]
            edge_params: 边参数 [E, 2] (R, X)
            
        Returns:
            KCL违反损失
        """
        device = node_features.device
        n_nodes = node_features.size(0)
        
        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=device)
        
        kcl_violations = []
        
        for node in range(n_nodes):
            current_sum = torch.tensor(0.0, device=device)
            
            # 找到与该节点相连的边
            src_mask = edge_index[0] == node
            dst_mask = edge_index[1] == node
            
            # 处理作为源节点的边（电流流出）
            if src_mask.any():
                src_indices = torch.where(src_mask)[0]
                for idx in src_indices:
                    current = self._compute_current(
                        node_features[node], 
                        node_features[edge_index[1, idx]],
                        edge_params[idx]
                    )
                    current_sum -= current * edge_probs[idx]
            
            # 处理作为目标节点的边（电流流入）
            if dst_mask.any():
                dst_indices = torch.where(dst_mask)[0]
                for idx in dst_indices:
                    current = self._compute_current(
                        node_features[edge_index[0, idx]],
                        node_features[node], 
                        edge_params[idx]
                    )
                    current_sum += current * edge_probs[idx]
            
            # KCL违反程度（电流不平衡的平方）
            kcl_violations.append(current_sum ** 2)
        
        return torch.stack(kcl_violations).mean()
    
    def _compute_current(self, v_from: torch.Tensor, v_to: torch.Tensor, 
                        params: torch.Tensor) -> torch.Tensor:
        """计算边上的电流"""
        # 简化的电流计算：I = (V_from - V_to) / Z
        # 这里假设电压为复数形式：V = V_real + j*V_imag
        v_from_complex = torch.complex(v_from[0], v_from[1])
        v_to_complex = torch.complex(v_to[0], v_to[1])
        
        # 阻抗：Z = R + jX
        impedance = torch.complex(params[0], params[1])
        
        # 电流计算
        current_complex = (v_from_complex - v_to_complex) / (impedance + 1e-6)
        
        # 返回电流幅值
        return torch.abs(current_complex)


class KirchhoffVoltageLaw(nn.Module):
    """基尔霍夫电压定律约束"""
    
    def __init__(self, tolerance: float = 0.05):
        super().__init__()
        self.tolerance = tolerance
        
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_probs: torch.Tensor, edge_params: torch.Tensor) -> torch.Tensor:
        """
        计算KVL违反程度（简化版本）
        
        对于径向网络，KVL自动满足，这里实现阻抗一致性约束
        """
        device = edge_params.device
        
        if edge_params.size(0) == 0:
            return torch.tensor(0.0, device=device)
        
        # 方法1：参数分布的一致性
        param_consistency = self._parameter_consistency_loss(edge_params, edge_probs)
        
        # 方法2：电压降一致性
        voltage_consistency = self._voltage_drop_consistency(
            node_features, edge_index, edge_probs, edge_params
        )
        
        return param_consistency + voltage_consistency
    
    def _parameter_consistency_loss(self, edge_params: torch.Tensor, 
                                  edge_probs: torch.Tensor) -> torch.Tensor:
        """参数一致性损失"""
        if edge_params.size(0) <= 1:
            return torch.tensor(0.0, device=edge_params.device)
        
        # 加权参数统计
        weights = edge_probs.unsqueeze(1)
        weighted_params = edge_params * weights
        
        # 计算加权方差
        mean_params = weighted_params.sum(dim=0) / (weights.sum() + 1e-6)
        param_variance = ((edge_params - mean_params) ** 2 * weights).sum(dim=0)
        
        return param_variance.mean()
    
    def _voltage_drop_consistency(self, node_features: torch.Tensor, 
                                edge_index: torch.Tensor,
                                edge_probs: torch.Tensor, 
                                edge_params: torch.Tensor) -> torch.Tensor:
        """电压降一致性约束"""
        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=node_features.device)
        
        voltage_drops = []
        
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i], edge_index[1, i]
            
            # 计算电压降
            v_src = torch.complex(node_features[src, 0], node_features[src, 1])
            v_dst = torch.complex(node_features[dst, 0], node_features[dst, 1])
            voltage_drop = torch.abs(v_src - v_dst)
            
            voltage_drops.append(voltage_drop * edge_probs[i])
        
        if voltage_drops:
            voltage_drops = torch.stack(voltage_drops)
            # 电压降的方差应该相对较小
            return torch.var(voltage_drops)
        else:
            return torch.tensor(0.0, device=node_features.device)


class RadialityConstraint(nn.Module):
    """径向性约束（树结构）"""
    
    def __init__(self, lambda_radial: float = 1.0):
        super().__init__()
        self.lambda_radial = lambda_radial
        
    def forward(self, edge_probs: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        径向性约束：确保网络形成树结构
        对于n个节点的树，应该有n-1条边
        """
        if edge_probs.size(0) == 0:
            return torch.tensor(0.0, device=edge_probs.device)
        
        # 期望边数
        expected_edges = num_nodes - 1
        actual_edges = edge_probs.sum()
        
        # 边数约束
        edge_count_loss = (actual_edges - expected_edges) ** 2
        
        # 稀疏性约束（鼓励稀疏连接）
        sparsity_loss = edge_probs.mean()
        
        return self.lambda_radial * edge_count_loss + 0.1 * sparsity_loss


class PowerFlowConsistency(nn.Module):
    """潮流一致性约束"""
    
    def __init__(self, tolerance: float = 0.1):
        super().__init__()
        self.tolerance = tolerance
        
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_probs: torch.Tensor, edge_params: torch.Tensor) -> torch.Tensor:
        """
        简化的潮流一致性检查
        """
        device = node_features.device
        
        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=device)
        
        # 功率平衡检查
        power_balance_loss = self._power_balance_constraint(
            node_features, edge_index, edge_probs, edge_params
        )
        
        # 电压约束
        voltage_magnitude_loss = self._voltage_magnitude_constraint(node_features)
        
        return power_balance_loss + voltage_magnitude_loss
    
    def _power_balance_constraint(self, node_features: torch.Tensor,
                                edge_index: torch.Tensor, edge_probs: torch.Tensor,
                                edge_params: torch.Tensor) -> torch.Tensor:
        """功率平衡约束"""
        n_nodes = node_features.size(0)
        power_imbalances = []
        
        for node in range(n_nodes):
            # 节点负载功率（从特征中获取）
            p_load = node_features[node, 2]  # 有功功率
            q_load = node_features[node, 3]  # 无功功率
            
            # 计算流入/流出功率
            p_flow = torch.tensor(0.0, device=node_features.device)
            q_flow = torch.tensor(0.0, device=node_features.device)
            
            # 遍历与该节点相连的边
            connected_edges = (edge_index[0] == node) | (edge_index[1] == node)
            
            if connected_edges.any():
                edge_indices = torch.where(connected_edges)[0]
                for idx in edge_indices:
                    # 简化的功率流计算
                    power_flow = self._compute_power_flow(
                        node_features, edge_index[:, idx], edge_params[idx]
                    )
                    p_flow += power_flow[0] * edge_probs[idx]
                    q_flow += power_flow[1] * edge_probs[idx]
            
            # 功率不平衡
            p_imbalance = (p_load + p_flow) ** 2
            q_imbalance = (q_load + q_flow) ** 2
            
            power_imbalances.append(p_imbalance + q_imbalance)
        
        return torch.stack(power_imbalances).mean()
    
    def _compute_power_flow(self, node_features: torch.Tensor, 
                          edge: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """计算边上的功率流"""
        src, dst = edge
        
        # 简化计算：P ≈ V²/R, Q ≈ V²/X
        v_magnitude = torch.sqrt(
            node_features[src, 0]**2 + node_features[src, 1]**2
        )
        
        p_flow = v_magnitude**2 / (params[0] + 1e-6)
        q_flow = v_magnitude**2 / (params[1] + 1e-6)
        
        return torch.stack([p_flow, q_flow])
    
    def _voltage_magnitude_constraint(self, node_features: torch.Tensor) -> torch.Tensor:
        """电压幅值约束"""
        # 计算电压幅值
        voltage_magnitudes = torch.sqrt(
            node_features[:, 0]**2 + node_features[:, 1]**2
        )
        
        # 标幺值约束：电压应在合理范围内（0.95-1.05）
        lower_bound = 0.95
        upper_bound = 1.05
        
        # 软约束
        lower_violations = F.relu(lower_bound - voltage_magnitudes)
        upper_violations = F.relu(voltage_magnitudes - upper_bound)
        
        return (lower_violations**2 + upper_violations**2).mean()


class PhysicsConstrainedLoss(nn.Module):
    """综合物理约束损失函数"""
    
    def __init__(self, config: PhysicsConfig):
        super().__init__()
        self.config = config
        
        # 各类物理约束
        self.kcl_loss = KirchhoffCurrentLaw(config.voltage_tolerance)
        self.kvl_loss = KirchhoffVoltageLaw(config.current_tolerance)
        self.radial_loss = RadialityConstraint()
        self.power_flow_loss = PowerFlowConsistency()
        
        # 基础损失函数
        self.topology_loss = nn.BCEWithLogitsLoss()
        self.parameter_loss = nn.MSELoss()
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor], 
                data: Data) -> Dict[str, torch.Tensor]:
        """
        计算综合损失
        
        Args:
            predictions: 模型预测结果
            targets: 真实标签  
            data: 图数据
            
        Returns:
            各项损失的字典
        """
        losses = {}
        
        edge_logits = predictions['edge_logits']
        edge_params = predictions['edge_params']
        edge_probs = torch.sigmoid(edge_logits)
        
        # 1. 拓扑分类损失
        if 'edge_labels' in targets:
            losses['topology'] = self.config.alpha_topology * self.topology_loss(
                edge_logits, targets['edge_labels']
            )
        else:
            losses['topology'] = torch.tensor(0.0, device=edge_logits.device)
        
        # 2. 参数回归损失
        if 'edge_params' in targets and targets['edge_params'].size(0) > 0:
            min_len = min(edge_params.size(0), targets['edge_params'].size(0))
            losses['parameter'] = self.config.alpha_parameter * self.parameter_loss(
                edge_params[:min_len], targets['edge_params'][:min_len]
            )
        else:
            losses['parameter'] = torch.tensor(0.0, device=edge_params.device)
        
        # 3. 基尔霍夫电流定律
        losses['kcl'] = self.config.alpha_kcl * self.kcl_loss(
            data.x, data.edge_index, edge_probs, edge_params
        )
        
        # 4. 基尔霍夫电压定律
        losses['kvl'] = self.config.alpha_kvl * self.kvl_loss(
            data.x, data.edge_index, edge_probs, edge_params
        )
        
        # 5. 径向性约束
        losses['radial'] = 0.1 * self.radial_loss(edge_probs, data.num_nodes)
        
        # 6. 稀疏性损失
        losses['sparsity'] = self.config.alpha_sparsity * edge_probs.mean()
        
        # 7. 地理距离损失
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            geo_distances = data.edge_attr.squeeze()
            losses['geographic'] = self.config.alpha_geographic * (
                edge_probs * geo_distances
            ).mean()
        else:
            losses['geographic'] = torch.tensor(0.0, device=edge_probs.device)
        
        # 8. 功率流一致性（可选）
        if hasattr(self.config, 'use_power_flow') and self.config.use_power_flow:
            losses['power_flow'] = 0.01 * self.power_flow_loss(
                data.x, data.edge_index, edge_probs, edge_params
            )
        
        # 总损失
        losses['total'] = sum(losses.values())
        
        return losses
    
    def create_edge_labels(self, candidate_edges: torch.Tensor, 
                          true_edges: torch.Tensor) -> torch.Tensor:
        """创建边标签"""
        labels = torch.zeros(candidate_edges.size(1), device=candidate_edges.device)
        
        if true_edges.size(1) == 0:
            return labels
        
        for i, (src, dst) in enumerate(candidate_edges.t()):
            # 检查是否为真实边（双向检查）
            is_true_edge = (
                ((true_edges[0] == src) & (true_edges[1] == dst)) |
                ((true_edges[0] == dst) & (true_edges[1] == src))
            ).any()
            
            if is_true_edge:
                labels[i] = 1.0
        
        return labels