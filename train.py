#!/usr/bin/env python3
"""
Physics-informed Graph Transformer for Power Grid Topology Reconstruction
基于您的配电网数据的完整Python实现 - 修复设备不匹配问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, mean_absolute_error
import networkx as nx
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class DataLoader:
    """数据加载和预处理类"""
    
    def __init__(self, data_path: str = "./"):
        self.data_path = data_path
        self.buses = None
        self.lines = None
        self.voltage_ts = None
        self.loading_ts = None
        self.loads = None
        self.generators = None
        
    def load_all_data(self):
        """加载所有数据文件"""
        try:
            # 加载核心数据
            self.buses = pd.read_csv(f"{self.data_path}/1-MV-urban--0-sw_bus_with_local_coords.csv")
            self.lines = pd.read_csv(f"{self.data_path}/1-MV-urban--0-sw_lines_with_coordinates.csv")
            self.voltage_ts = pd.read_csv(f"{self.data_path}/1-MV-urban--0-sw_voltage_timeseries.csv")
            self.loading_ts = pd.read_csv(f"{self.data_path}/1-MV-urban--0-sw_loading_timeseries.csv")
            
            # 可选数据
            try:
                self.loads = pd.read_csv(f"{self.data_path}/1MVurban0sw_loads_with_coordinates.csv")
                self.generators = pd.read_csv(f"{self.data_path}/1MVurban0sw_generators_with_coordinates.csv")
            except:
                print("Warning: 负载和发电机数据文件未找到，使用模拟数据")
                
            print(f"数据加载完成:")
            print(f"  - 节点数: {len(self.buses)}")
            print(f"  - 线路数: {len(self.lines)}")
            print(f"  - 时间步数: {len(self.voltage_ts)}")
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            
    def mask_unobservable(self, ratio: float = 0.25):
        """随机遮蔽部分节点作为不可观测"""
        n_nodes = len(self.buses)
        n_masked = int(n_nodes * ratio)
        
        # 随机选择要遮蔽的节点
        masked_indices = np.random.choice(n_nodes, n_masked, replace=False)
        self.buses['observed'] = True
        self.buses.loc[masked_indices, 'observed'] = False
        
        print(f"遮蔽了 {n_masked} 个节点 ({ratio*100:.1f}%)")
        return masked_indices
        
    def build_node_features(self, time_step: int = 0) -> torch.Tensor:
        """构建节点特征矩阵"""
        features = []
        
        for idx, bus in self.buses.iterrows():
            # 电气特征
            if bus.get('observed', True):
                # 电压特征（从时间序列获取）
                voltage_col = f'bus_{idx}'
                if voltage_col in self.voltage_ts.columns and time_step < len(self.voltage_ts):
                    voltage = self.voltage_ts.iloc[time_step][voltage_col]
                else:
                    voltage = 1.0  # 标幺值
                    
                # 将电压转换为实部虚部（假设相角为0）
                v_real = voltage * np.cos(0)
                v_imag = voltage * np.sin(0)
                
                # 功率特征（从负载数据获取或模拟）
                if self.loads is not None:
                    load_data = self.loads[self.loads['bus'] == idx]
                    if not load_data.empty:
                        p_load = load_data.iloc[0]['p_mw']
                        q_load = load_data.iloc[0]['q_mvar']
                    else:
                        p_load = np.random.normal(0.05, 0.02)  # MW
                        q_load = np.random.normal(0.02, 0.01)  # MVar
                else:
                    p_load = np.random.normal(0.05, 0.02)
                    q_load = np.random.normal(0.02, 0.01)
            else:
                # 不可观测节点特征置零
                v_real = v_imag = p_load = q_load = 0.0
                
            # 空间和结构特征
            x_coord = bus['x'] if pd.notna(bus['x']) else 0.0
            y_coord = bus['y'] if pd.notna(bus['y']) else 0.0
            volt_level = bus['voltLvl'] if pd.notna(bus['voltLvl']) else 1
            
            # 元路径嵌入（简化为节点类型编码）
            bus_type = bus.get('type', 'b')
            type_encoding = {'b': 0, 'db': 1, 'auxiliary': 2}.get(bus_type, 0)
            
            # 合并所有特征
            node_feat = [
                v_real, v_imag, p_load, q_load,  # 电气特征
                volt_level, x_coord, y_coord,     # 空间特征
                type_encoding, bus.get('observed', True)  # 结构特征
            ]
            
            features.append(node_feat)
            
        return torch.tensor(features, dtype=torch.float32)
    
    def build_candidate_graph(self, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """基于地理坐标构建候选图"""
        coords = self.buses[['x', 'y']].fillna(0).values
        
        # 使用k-近邻构建候选边
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        edges = []
        edge_distances = []
        
        for i in range(len(coords)):
            for j in range(1, len(indices[i])):  # 跳过自己
                neighbor = indices[i][j]
                if i < neighbor:  # 避免重复边
                    edges.append([i, neighbor])
                    edge_distances.append(distances[i][j])
                    
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_distances, dtype=torch.float32).unsqueeze(1)
        
        return edge_index, edge_attr
    
    def get_true_topology(self) -> torch.Tensor:
        """获取真实拓扑（从线路数据）"""
        true_edges = []
        true_params = []
        
        for _, line in self.lines.iterrows():
            if line['in_service'] and pd.notna(line['from_bus']) and pd.notna(line['to_bus']):
                from_bus = int(line['from_bus'])
                to_bus = int(line['to_bus'])
                
                # 确保节点索引在有效范围内
                if from_bus < len(self.buses) and to_bus < len(self.buses):
                    true_edges.append([from_bus, to_bus])
                    
                    # 线路参数
                    r_per_km = line.get('r_ohm_per_km', 0.1)
                    x_per_km = line.get('x_ohm_per_km', 0.1)
                    length = line.get('length_km', 0.1)
                    
                    r_total = r_per_km * length
                    x_total = x_per_km * length
                    true_params.append([r_total, x_total])
                    
        if true_edges:
            true_edge_index = torch.tensor(true_edges, dtype=torch.long).t().contiguous()
            true_edge_params = torch.tensor(true_params, dtype=torch.float32)
        else:
            # 如果没有真实边，创建空张量
            true_edge_index = torch.empty((2, 0), dtype=torch.long)
            true_edge_params = torch.empty((0, 2), dtype=torch.float32)
            
        return true_edge_index, true_edge_params


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        
        # 线性投影
        Q = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_head)
        
        # 基于图结构的掩码（可选）
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.out_linear(context)


class PhysicsGraphTransformer(nn.Module):
    """物理约束图变换器模型"""
    
    def __init__(self, d_in: int, d_hidden: int, n_heads: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.d_hidden = d_hidden
        
        # 输入投影
        self.input_projection = nn.Linear(d_in, d_hidden)
        
        # Graph Transformer层
        self.transformer_layers = nn.ModuleList([
            TransformerConv(d_hidden, d_hidden // n_heads, heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_hidden) for _ in range(n_layers)
        ])
        
        # 输出头
        self.edge_classifier = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1)
        )
        
        self.param_predictor = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 2)  # R, X
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 输入投影
        h = self.input_projection(x)
        
        # Graph Transformer编码
        for transformer, norm in zip(self.transformer_layers, self.layer_norms):
            h_new = transformer(h, edge_index)
            h = norm(h + h_new)  # 残差连接
            
        # 边特征构建
        src, dst = edge_index
        edge_features = torch.cat([h[src], h[dst]], dim=1)
        
        # 边分类（存在概率）
        edge_logits = self.edge_classifier(edge_features).squeeze(-1)
        
        # 参数预测
        edge_params = self.param_predictor(edge_features)
        edge_params = F.softplus(edge_params)  # 确保参数为正
        
        return edge_logits, edge_params


class PhysicsLoss(nn.Module):
    """物理约束损失函数"""
    
    def __init__(self, alpha_kcl: float = 1.0, alpha_kvl: float = 1.0):
        super().__init__()
        self.alpha_kcl = alpha_kcl
        self.alpha_kvl = alpha_kvl
        
    def kirchhoff_current_law(self, node_features: torch.Tensor, edge_index: torch.Tensor, 
                            edge_probs: torch.Tensor, edge_params: torch.Tensor) -> torch.Tensor:
        """基尔霍夫电流定律约束"""
        device = node_features.device
        n_nodes = node_features.size(0)
        kcl_violations = []
        
        for node in range(n_nodes):
            current_sum = torch.tensor(0.0, device=device)  # 修复：指定设备
            
            # 找到与该节点相连的所有边
            src_mask = edge_index[0] == node
            dst_mask = edge_index[1] == node
            
            if src_mask.any():
                # 作为源节点的边
                src_indices = torch.where(src_mask)[0]
                for idx in src_indices:
                    # 简化的电流计算：V_diff / Z
                    v_diff = torch.abs(node_features[node, 0] - node_features[edge_index[1, idx], 0])
                    impedance = edge_params[idx, 0] + edge_params[idx, 1]  # R + jX的模
                    current = v_diff / (impedance + 1e-6)
                    current_sum -= current * edge_probs[idx]
                    
            if dst_mask.any():
                # 作为目标节点的边
                dst_indices = torch.where(dst_mask)[0]
                for idx in dst_indices:
                    v_diff = torch.abs(node_features[edge_index[0, idx], 0] - node_features[node, 0])
                    impedance = edge_params[idx, 0] + edge_params[idx, 1]
                    current = v_diff / (impedance + 1e-6)
                    current_sum += current * edge_probs[idx]
                    
            kcl_violations.append(current_sum ** 2)
            
        return torch.stack(kcl_violations).mean()
    
    def kirchhoff_voltage_law(self, edge_index: torch.Tensor, edge_probs: torch.Tensor, 
                            edge_params: torch.Tensor) -> torch.Tensor:
        """基尔霍夫电压定律约束（简化版）"""
        # 简化：对于径向网络，KVL自动满足
        # 这里实现一个基于阻抗一致性的约束
        if edge_params.size(0) == 0:
            return torch.tensor(0.0, device=edge_params.device)  # 修复：指定设备
            
        # 检查相似线路的参数一致性
        param_variance = torch.var(edge_params, dim=0).mean()
        return param_variance
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_logits: torch.Tensor, edge_params: torch.Tensor) -> torch.Tensor:
        edge_probs = torch.sigmoid(edge_logits)
        
        kcl_loss = self.kirchhoff_current_law(node_features, edge_index, edge_probs, edge_params)
        kvl_loss = self.kirchhoff_voltage_law(edge_index, edge_probs, edge_params)
        
        return self.alpha_kcl * kcl_loss + self.alpha_kvl * kvl_loss


class PowerGridTrainer:
    """训练管理器"""
    
    def __init__(self, model: PhysicsGraphTransformer, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10)
        
        # 损失函数
        self.topo_criterion = nn.BCEWithLogitsLoss()
        self.param_criterion = nn.MSELoss()
        self.physics_criterion = PhysicsLoss()
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        
    def compute_loss(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                    edge_logits: torch.Tensor, edge_params: torch.Tensor,
                    true_edges: torch.Tensor, true_params: torch.Tensor,
                    edge_distances: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算综合损失"""
        
        # 1. 拓扑损失
        if true_edges.size(1) > 0:
            # 创建边标签矩阵 - 修复：指定设备
            edge_labels = torch.zeros(edge_logits.size(0), device=edge_logits.device)
            for i, (src, dst) in enumerate(edge_index.t()):
                is_true_edge = ((true_edges[0] == src) & (true_edges[1] == dst)) | \
                              ((true_edges[0] == dst) & (true_edges[1] == src))
                if is_true_edge.any():
                    edge_labels[i] = 1.0
                    
            topo_loss = self.topo_criterion(edge_logits, edge_labels)
        else:
            topo_loss = torch.tensor(0.0, device=edge_logits.device)
            
        # 2. 参数损失（只对真实边计算）
        if true_edges.size(1) > 0 and true_params.size(0) > 0:
            param_loss = self.param_criterion(edge_params[:true_params.size(0)], true_params)
        else:
            param_loss = torch.tensor(0.0, device=edge_params.device)
            
        # 3. 物理约束损失
        physics_loss = self.physics_criterion(node_features, edge_index, edge_logits, edge_params)
        
        # 4. 稀疏性损失
        sparsity_loss = torch.sigmoid(edge_logits).mean()
        
        # 5. 地理距离损失
        geo_loss = (torch.sigmoid(edge_logits) * edge_distances.squeeze()).mean()
        
        # 综合损失
        total_loss = (topo_loss + param_loss + 0.1 * physics_loss + 
                     0.01 * sparsity_loss + 0.01 * geo_loss)
        
        return {
            'total': total_loss,
            'topology': topo_loss,
            'parameter': param_loss,
            'physics': physics_loss,
            'sparsity': sparsity_loss,
            'geographic': geo_loss
        }
    
    def train_epoch(self, data_loader: DataLoader, time_steps: List[int]) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_losses = {'total': 0, 'topology': 0, 'parameter': 0, 'physics': 0, 'sparsity': 0, 'geographic': 0}
        
        for time_step in time_steps:
            self.optimizer.zero_grad()
            
            # 准备数据
            node_features = data_loader.build_node_features(time_step).to(self.device)
            edge_index, edge_distances = data_loader.build_candidate_graph()
            edge_index = edge_index.to(self.device)
            edge_distances = edge_distances.to(self.device)
            
            true_edge_index, true_edge_params = data_loader.get_true_topology()
            true_edge_index = true_edge_index.to(self.device)
            true_edge_params = true_edge_params.to(self.device)
            
            # 前向传播
            edge_logits, edge_params = self.model(node_features, edge_index)
            
            # 计算损失
            losses = self.compute_loss(node_features, edge_index, edge_logits, edge_params,
                                     true_edge_index, true_edge_params, edge_distances)
            
            # 反向传播
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 累计损失
            for key, value in losses.items():
                epoch_losses[key] += value.item()
                
        # 平均损失
        n_steps = len(time_steps)
        return {key: value / n_steps for key, value in epoch_losses.items()}
    
    def train(self, data_loader: DataLoader, epochs: int = 100, time_steps: Optional[List[int]] = None):
        """完整训练流程"""
        if time_steps is None:
            time_steps = list(range(min(20, len(data_loader.voltage_ts))))
            
        print(f"开始训练 {epochs} epochs...")
        
        for epoch in range(epochs):
            # 训练
            train_losses = self.train_epoch(data_loader, time_steps)
            self.train_losses.append(train_losses)
            
            # 学习率调度
            self.scheduler.step(train_losses['total'])
            
            # 打印进度
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}: "
                      f"Total={train_losses['total']:.4f}, "
                      f"Topo={train_losses['topology']:.4f}, "
                      f"Param={train_losses['parameter']:.4f}, "
                      f"Physics={train_losses['physics']:.4f}")
                      
        print("训练完成!")
    
    def infer_topology(self, data_loader: DataLoader, time_step: int = 0, threshold: float = 0.5):
        """推断网络拓扑"""
        self.model.eval()
        
        with torch.no_grad():
            # 准备数据
            node_features = data_loader.build_node_features(time_step).to(self.device)
            edge_index, edge_distances = data_loader.build_candidate_graph()
            edge_index = edge_index.to(self.device)
            
            # 推断
            edge_logits, edge_params = self.model(node_features, edge_index)
            edge_probs = torch.sigmoid(edge_logits)
            
            # 选择边
            selected_mask = edge_probs > threshold
            selected_edges = edge_index[:, selected_mask]
            selected_params = edge_params[selected_mask]
            selected_probs = edge_probs[selected_mask]
            
            # 确保径向性（最小生成树）
            if selected_edges.size(1) > 0:
                radial_edges = self.ensure_radial_topology(selected_edges, selected_probs, len(data_loader.buses))
            else:
                radial_edges = selected_edges
                
        return {
            'edges': selected_edges.cpu(),
            'radial_edges': radial_edges.cpu() if torch.is_tensor(radial_edges) else radial_edges,
            'parameters': selected_params.cpu(),
            'probabilities': selected_probs.cpu()
        }
    
    def ensure_radial_topology(self, edge_index: torch.Tensor, edge_weights: torch.Tensor, n_nodes: int):
        """确保拓扑为径向（树结构）"""
        # 转换为NetworkX图
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        
        edges_with_weights = []
        for i, (src, dst) in enumerate(edge_index.t()):
            weight = 1.0 / (edge_weights[i].item() + 1e-6)  # 权重越高，概率越大
            edges_with_weights.append((src.item(), dst.item(), weight))
            
        G.add_weighted_edges_from(edges_with_weights)
        
        # 最小生成树
        if G.number_of_edges() > 0:
            mst = nx.minimum_spanning_tree(G)
            radial_edges = torch.tensor(list(mst.edges())).t()
        else:
            radial_edges = torch.empty((2, 0), dtype=torch.long)
            
        return radial_edges


def evaluate_results(predicted_edges: torch.Tensor, true_edges: torch.Tensor, 
                    predicted_params: torch.Tensor, true_params: torch.Tensor) -> Dict[str, float]:
    """评估预测结果"""
    metrics = {}
    
    # 拓扑F1分数
    if true_edges.size(1) > 0 and predicted_edges.size(1) > 0:
        # 创建邻接矩阵进行比较
        n_nodes = max(true_edges.max().item(), predicted_edges.max().item()) + 1
        
        true_adj = torch.zeros(n_nodes, n_nodes)
        pred_adj = torch.zeros(n_nodes, n_nodes)
        
        for src, dst in true_edges.t():
            true_adj[src, dst] = true_adj[dst, src] = 1
            
        for src, dst in predicted_edges.t():
            pred_adj[src, dst] = pred_adj[dst, src] = 1
            
        # 计算F1分数
        true_flat = true_adj.flatten().numpy()
        pred_flat = pred_adj.flatten().numpy()
        
        metrics['topology_f1'] = f1_score(true_flat, pred_flat)
        metrics['topology_precision'] = (pred_flat * true_flat).sum() / (pred_flat.sum() + 1e-6)
        metrics['topology_recall'] = (pred_flat * true_flat).sum() / (true_flat.sum() + 1e-6)
    
    # 参数MAE
    if true_params.size(0) > 0 and predicted_params.size(0) > 0:
        min_len = min(true_params.size(0), predicted_params.size(0))
        metrics['param_mae_r'] = mean_absolute_error(
            true_params[:min_len, 0].numpy(), 
            predicted_params[:min_len, 0].numpy()
        )
        metrics['param_mae_x'] = mean_absolute_error(
            true_params[:min_len, 1].numpy(), 
            predicted_params[:min_len, 1].numpy()
        )
    
    return metrics


def visualize_results(data_loader: DataLoader, results: Dict, save_path: str = None):
    """可视化结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 网络拓扑可视化
    ax = axes[0, 0]
    coords = data_loader.buses[['x', 'y']].fillna(0).values
    
    # 绘制节点
    ax.scatter(coords[:, 0], coords[:, 1], c='lightblue', s=50, alpha=0.7, label='Buses')
    
    # 绘制预测边
    if 'edges' in results and results['edges'].size(1) > 0:
        for src, dst in results['edges'].t():
            x_coords = [coords[src, 0], coords[dst, 0]]
            y_coords = [coords[src, 1], coords[dst, 1]]
            ax.plot(x_coords, y_coords, 'r-', alpha=0.6, linewidth=1)
    
    ax.set_title('预测网络拓扑')
    ax.set_xlabel('X坐标')
    ax.set_ylabel('Y坐标')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 径向拓扑
    ax = axes[0, 1]
    ax.scatter(coords[:, 0], coords[:, 1], c='lightgreen', s=50, alpha=0.7, label='Buses')
    
    if 'radial_edges' in results and torch.is_tensor(results['radial_edges']) and results['radial_edges'].size(1) > 0:
        for src, dst in results['radial_edges'].t():
            x_coords = [coords[src, 0], coords[dst, 0]]
            y_coords = [coords[src, 1], coords[dst, 1]]
            ax.plot(x_coords, y_coords, 'g-', alpha=0.8, linewidth=2)
    
    ax.set_title('径向网络拓扑')
    ax.set_xlabel('X坐标')
    ax.set_ylabel('Y坐标')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 参数分布
    ax = axes[1, 0]
    if 'parameters' in results and results['parameters'].size(0) > 0:
        params = results['parameters'].numpy()
        ax.scatter(params[:, 0], params[:, 1], alpha=0.6)
        ax.set_xlabel('电阻 R (Ω)')
        ax.set_ylabel('电抗 X (Ω)')
        ax.set_title('预测线路参数分布')
        ax.grid(True, alpha=0.3)
    
    # 4. 置信度分布
    ax = axes[1, 1]
    if 'probabilities' in results and results['probabilities'].size(0) > 0:
        probs = results['probabilities'].numpy()
        ax.hist(probs, bins=20, alpha=0.7, edgecolor='black')
        ax.set_xlabel('边存在概率')
        ax.set_ylabel('频数')
        ax.set_title('边预测置信度分布')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """主函数"""
    print("=" * 60)
    print("Physics-informed Graph Transformer for Power Grid")
    print("=" * 60)
    
    # 1. 数据加载
    print("\n1. 加载数据...")
    data_loader = DataLoader("./local_coords_results_1_MV_urban__0_sw")  # 修改为您的数据路径
    data_loader.load_all_data()
    
    # 2. 遮蔽部分节点
    print("\n2. 模拟部分可观测性...")
    data_loader.mask_unobservable(ratio=0.25)
    
    # 3. 构建模型
    print("\n3. 构建模型...")
    sample_features = data_loader.build_node_features(0)
    d_in = sample_features.size(1)
    
    model = PhysicsGraphTransformer(
        d_in=d_in,
        d_hidden=128,
        n_heads=4,
        n_layers=3,
        dropout=0.1
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 4. 训练
    print("\n4. 开始训练...")
    trainer = PowerGridTrainer(model, device)
    trainer.train(data_loader, epochs=100)
    
    # 5. 推断
    print("\n5. 拓扑推断...")
    results = trainer.infer_topology(data_loader, time_step=0, threshold=0.5)
    
    print(f"推断出 {results['edges'].size(1)} 条边")
    if torch.is_tensor(results['radial_edges']):
        print(f"径向拓扑包含 {results['radial_edges'].size(1)} 条边")
    
    # 6. 评估
    print("\n6. 评估结果...")
    true_edges, true_params = data_loader.get_true_topology()
    metrics = evaluate_results(results['edges'], true_edges, 
                             results['parameters'], true_params)
    
    print("评估指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 7. 可视化
    print("\n7. 可视化结果...")
    visualize_results(data_loader, results, 'power_grid_results.png')
    
    # 8. 训练曲线
    if trainer.train_losses:
        plt.figure(figsize=(12, 8))
        losses_df = pd.DataFrame(trainer.train_losses)
        
        for i, col in enumerate(['total', 'topology', 'parameter', 'physics']):
            plt.subplot(2, 2, i+1)
            plt.plot(losses_df[col])
            plt.title(f'{col.capitalize()} Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    print("\n" + "=" * 60)
    print("程序执行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()