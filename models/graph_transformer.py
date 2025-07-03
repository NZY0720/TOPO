#!/usr/bin/env python3
"""
Physics-informed Graph Transformer for Power Grid Topology Reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GATConv, GCNConv
from torch_geometric.data import Data
from typing import Tuple, Optional, Dict, Any
import math

from ..config.base_config import ModelConfig


class MultiHeadGraphAttention(nn.Module):
    """多头图注意力机制"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_head)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.size(0)
        
        # 线性变换
        Q = self.w_q(x).view(batch_size, self.n_heads, self.d_head)
        K = self.w_k(x).view(batch_size, self.n_heads, self.d_head)
        V = self.w_v(x).view(batch_size, self.n_heads, self.d_head)
        
        # 构建图上的注意力
        src, dst = edge_index
        
        # 计算注意力分数
        q_i = Q[src]  # [num_edges, n_heads, d_head]
        k_j = K[dst]  # [num_edges, n_heads, d_head]
        v_j = V[dst]  # [num_edges, n_heads, d_head]
        
        # 注意力权重
        attn_scores = (q_i * k_j).sum(dim=-1) * self.scale  # [num_edges, n_heads]
        
        # 边属性调制（可选）
        if edge_attr is not None:
            # 简单的边属性编码
            edge_weights = torch.sigmoid(edge_attr.sum(dim=-1, keepdim=True))  # [num_edges, 1]
            attn_scores = attn_scores * edge_weights
        
        # Softmax归一化（针对每个节点）
        attn_weights = torch.zeros(batch_size, self.n_heads, device=x.device)
        for i in range(batch_size):
            mask = (src == i)
            if mask.any():
                attn_weights[i] = F.softmax(attn_scores[mask], dim=0).mean(dim=0)
        
        # 应用注意力
        out = torch.zeros_like(Q)
        for i in range(batch_size):
            mask = (dst == i)
            if mask.any():
                weighted_v = v_j[mask] * attn_scores[mask].unsqueeze(-1)
                out[i] = weighted_v.mean(dim=0)
        
        # 输出投影
        out = out.view(batch_size, self.d_model)
        return self.w_o(out)


class GraphTransformerLayer(nn.Module):
    """图变换器层"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # 图注意力
        self.self_attn = MultiHeadGraphAttention(d_model, n_heads, dropout)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力 + 残差连接
        attn_out = self.self_attn(x, edge_index, edge_attr)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 前馈网络 + 残差连接
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class PhysicsGraphTransformer(nn.Module):
    """物理约束图变换器"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 输入投影
        self.input_projection = nn.Linear(config.d_input, config.d_hidden)
        
        # 图变换器层
        self.transformer_layers = nn.ModuleList([
            GraphTransformerLayer(
                d_model=config.d_hidden,
                n_heads=config.n_heads,
                d_ff=config.d_hidden * 4,
                dropout=config.dropout
            ) for _ in range(config.n_layers)
        ])
        
        # 边特征编码器
        self.edge_encoder = nn.Sequential(
            nn.Linear(1, config.d_hidden // 4),  # 边距离特征
            nn.ReLU(),
            nn.Linear(config.d_hidden // 4, config.d_hidden // 2)
        )
        
        # 输出头
        self.edge_classifier = nn.Sequential(
            nn.Linear(config.d_hidden * 2 + config.d_hidden // 2, config.d_hidden),
            nn.ReLU() if config.activation == 'relu' else nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_hidden, config.d_hidden // 2),
            nn.ReLU() if config.activation == 'relu' else nn.GELU(),
            nn.Linear(config.d_hidden // 2, 1)
        )
        
        self.param_predictor = nn.Sequential(
            nn.Linear(config.d_hidden * 2 + config.d_hidden // 2, config.d_hidden),
            nn.ReLU() if config.activation == 'relu' else nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_hidden, config.d_hidden // 2),
            nn.ReLU() if config.activation == 'relu' else nn.GELU(),
            nn.Linear(config.d_hidden // 2, config.n_edge_params)
        )
        
        # 参数初始化
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 输入投影
        h = self.input_projection(x)
        
        # 边特征编码
        edge_feat = self.edge_encoder(edge_attr)
        
        # 图变换器编码
        for layer in self.transformer_layers:
            h = layer(h, edge_index, edge_feat)
        
        # 构建边特征
        src, dst = edge_index
        edge_features = torch.cat([
            h[src],           # 源节点特征
            h[dst],           # 目标节点特征
            edge_feat         # 边特征
        ], dim=1)
        
        # 边分类（存在概率）
        edge_logits = self.edge_classifier(edge_features).squeeze(-1)
        
        # 参数预测
        edge_params = self.param_predictor(edge_features)
        
        # 确保参数为正
        edge_params = F.softplus(edge_params) + 1e-6
        
        return edge_logits, edge_params
    
    def predict_topology(self, data: Data, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """预测拓扑结构"""
        self.eval()
        with torch.no_grad():
            edge_logits, edge_params = self.forward(data)
            edge_probs = torch.sigmoid(edge_logits)
            
            # 选择边
            selected_mask = edge_probs > threshold
            selected_edges = data.edge_index[:, selected_mask]
            selected_params = edge_params[selected_mask]
            selected_probs = edge_probs[selected_mask]
            
            return {
                'edge_index': selected_edges,
                'edge_params': selected_params,
                'edge_probs': selected_probs,
                'all_probs': edge_probs,
                'all_params': edge_params
            }
    
    def get_attention_weights(self, data: Data, layer_idx: int = -1) -> torch.Tensor:
        """获取注意力权重（用于可视化）"""
        self.eval()
        with torch.no_grad():
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            
            # 前向传播到指定层
            h = self.input_projection(x)
            edge_feat = self.edge_encoder(edge_attr)
            
            target_layer = self.transformer_layers[layer_idx]
            
            # 获取注意力权重
            attn_weights = target_layer.self_attn(h, edge_index, edge_feat)
            
            return attn_weights


class HybridGraphTransformer(PhysicsGraphTransformer):
    """混合图变换器（结合多种图卷积）"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # 多种图卷积层
        self.gat_layers = nn.ModuleList([
            GATConv(config.d_hidden, config.d_hidden // config.n_heads, 
                   heads=config.n_heads, dropout=config.dropout, concat=True)
            for _ in range(config.n_layers // 2)
        ])
        
        self.gcn_layers = nn.ModuleList([
            GCNConv(config.d_hidden, config.d_hidden)
            for _ in range(config.n_layers // 2)
        ])
        
        # 特征融合
        self.feature_fusion = nn.Linear(config.d_hidden * 2, config.d_hidden)
        
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """混合前向传播"""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 输入投影
        h = self.input_projection(x)
        
        # GAT分支
        h_gat = h
        for gat_layer in self.gat_layers:
            h_gat = F.relu(gat_layer(h_gat, edge_index))
            h_gat = F.dropout(h_gat, p=self.config.dropout, training=self.training)
        
        # GCN分支
        h_gcn = h
        for gcn_layer in self.gcn_layers:
            h_gcn = F.relu(gcn_layer(h_gcn, edge_index))
            h_gcn = F.dropout(h_gcn, p=self.config.dropout, training=self.training)
        
        # 特征融合
        h_combined = self.feature_fusion(torch.cat([h_gat, h_gcn], dim=1))
        
        # 边特征构建和输出
        edge_feat = self.edge_encoder(edge_attr)
        src, dst = edge_index
        edge_features = torch.cat([
            h_combined[src],
            h_combined[dst],
            edge_feat
        ], dim=1)
        
        edge_logits = self.edge_classifier(edge_features).squeeze(-1)
        edge_params = self.param_predictor(edge_features)
        edge_params = F.softplus(edge_params) + 1e-6
        
        return edge_logits, edge_params


def create_model(config: ModelConfig, input_dim: int) -> PhysicsGraphTransformer:
    """创建模型的工厂函数"""
    config.d_input = input_dim
    
    model_type = getattr(config, 'model_type', 'standard')
    
    if model_type == 'hybrid':
        return HybridGraphTransformer(config)
    else:
        return PhysicsGraphTransformer(config)


def count_parameters(model: nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, input_shape: Tuple[int, ...]) -> str:
    """生成模型摘要"""
    num_params = count_parameters(model)
    
    summary = f"""
    Model Summary:
    ==============
    Model Type: {model.__class__.__name__}
    Total Parameters: {num_params:,}
    
    Architecture:
    - Input Dimension: {input_shape[-1]}
    - Hidden Dimension: {model.config.d_hidden}
    - Number of Heads: {model.config.n_heads}
    - Number of Layers: {model.config.n_layers}
    - Dropout Rate: {model.config.dropout}
    - Output Parameters: {model.config.n_edge_params}
    """
    
    return summary