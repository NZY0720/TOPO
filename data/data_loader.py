#!/usr/bin/env python3
"""
Data loading and preprocessing for Power Grid Topology Reconstruction
"""

import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from ..config.base_config import DataConfig
from ..utils.graph_utils import create_candidate_graph, ensure_connected_graph
from ..utils.io_utils import load_csv_safe


class PowerGridDataLoader:
    """电力网格数据加载器"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.buses = None
        self.lines = None
        self.voltage_ts = None
        self.loading_ts = None
        self.loads = None
        self.generators = None
        self.scaler = None
        
        # 缓存的特征
        self._node_features_cache = {}
        self._candidate_graph_cache = None
        self._true_topology_cache = None
        
    def load_data(self, data_path: Optional[str] = None) -> None:
        """加载所有数据文件"""
        if data_path is None:
            data_path = self.config.data_path
            
        self.data_path = data_path
        
        try:
            # 加载核心数据文件
            self.buses = load_csv_safe(
                os.path.join(data_path, self.config.bus_file)
            )
            self.lines = load_csv_safe(
                os.path.join(data_path, self.config.line_file)
            )
            self.voltage_ts = load_csv_safe(
                os.path.join(data_path, self.config.voltage_ts_file)
            )
            self.loading_ts = load_csv_safe(
                os.path.join(data_path, self.config.loading_ts_file)
            )
            
            # 加载可选数据文件
            if self.config.loads_file:
                loads_path = os.path.join(data_path, self.config.loads_file)
                if os.path.exists(loads_path):
                    self.loads = load_csv_safe(loads_path)
                    
            if self.config.generators_file:
                gen_path = os.path.join(data_path, self.config.generators_file)
                if os.path.exists(gen_path):
                    self.generators = load_csv_safe(gen_path)
            
            # 数据验证和清理
            self._validate_and_clean_data()
            
            print(f"✅ 数据加载完成:")
            print(f"   - 节点数: {len(self.buses)}")
            print(f"   - 线路数: {len(self.lines)}")
            print(f"   - 时间步数: {len(self.voltage_ts)}")
            print(f"   - 负载点数: {len(self.loads) if self.loads is not None else 0}")
            print(f"   - 发电机数: {len(self.generators) if self.generators is not None else 0}")
            
        except Exception as e:
            raise RuntimeError(f"数据加载失败: {e}")
    
    def _validate_and_clean_data(self) -> None:
        """验证和清理数据"""
        # 检查必要的列
        required_bus_cols = ['x', 'y']
        required_line_cols = ['from_bus', 'to_bus', 'in_service']
        
        for col in required_bus_cols:
            if col not in self.buses.columns:
                raise ValueError(f"节点数据缺少必要列: {col}")
                
        for col in required_line_cols:
            if col not in self.lines.columns:
                raise ValueError(f"线路数据缺少必要列: {col}")
        
        # 填充缺失值
        self.buses['x'] = self.buses['x'].fillna(0.0)
        self.buses['y'] = self.buses['y'].fillna(0.0)
        self.buses['voltLvl'] = self.buses.get('voltLvl', 1.0).fillna(1.0)
        self.buses['type'] = self.buses.get('type', 'b').fillna('b')
        
        # 确保线路参数存在
        self.lines['r_ohm_per_km'] = self.lines.get('r_ohm_per_km', 0.1).fillna(0.1)
        self.lines['x_ohm_per_km'] = self.lines.get('x_ohm_per_km', 0.1).fillna(0.1)
        self.lines['length_km'] = self.lines.get('length_km', 0.1).fillna(0.1)
        
        # 重置索引确保一致性
        self.buses = self.buses.reset_index(drop=True)
        self.lines = self.lines.reset_index(drop=True)
    
    def mask_unobservable_nodes(self, ratio: Optional[float] = None, 
                               seed: Optional[int] = None) -> np.ndarray:
        """随机遮蔽部分节点作为不可观测"""
        if ratio is None:
            ratio = self.config.unobservable_ratio
            
        if seed is not None:
            np.random.seed(seed)
            
        n_nodes = len(self.buses)
        n_masked = int(n_nodes * ratio)
        
        # 随机选择要遮蔽的节点
        masked_indices = np.random.choice(n_nodes, n_masked, replace=False)
        
        # 设置可观测性标志
        self.buses['observed'] = True
        self.buses.loc[masked_indices, 'observed'] = False
        
        print(f"🎭 遮蔽了 {n_masked}/{n_nodes} 个节点 ({ratio*100:.1f}%)")
        
        # 清除缓存
        self._node_features_cache.clear()
        
        return masked_indices
    
    def build_node_features(self, time_step: int = 0, use_cache: bool = True) -> torch.Tensor:
        """构建节点特征矩阵"""
        cache_key = f"features_{time_step}"
        
        if use_cache and cache_key in self._node_features_cache:
            return self._node_features_cache[cache_key]
        
        features = []
        n_nodes = len(self.buses)
        
        for idx in range(n_nodes):
            bus = self.buses.iloc[idx]
            node_features = self._extract_node_features(bus, idx, time_step)
            features.append(node_features)
        
        # 转换为张量
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        
        # 特征归一化
        if self.config.normalize_features:
            if self.scaler is None:
                self.scaler = StandardScaler()
                feature_tensor = torch.tensor(
                    self.scaler.fit_transform(feature_tensor.numpy()),
                    dtype=torch.float32
                )
            else:
                feature_tensor = torch.tensor(
                    self.scaler.transform(feature_tensor.numpy()),
                    dtype=torch.float32
                )
        
        # 添加噪声（数据增强）
        if self.config.add_noise and self.config.noise_std > 0:
            noise = torch.randn_like(feature_tensor) * self.config.noise_std
            feature_tensor += noise
        
        # 缓存特征
        if use_cache:
            self._node_features_cache[cache_key] = feature_tensor
            
        return feature_tensor
    
    def _extract_node_features(self, bus: pd.Series, bus_idx: int, time_step: int) -> List[float]:
        """提取单个节点的特征"""
        features = []
        
        # 1. 电气测量特征
        if bus.get('observed', True):
            # 电压特征
            voltage = self._get_voltage_measurement(bus_idx, time_step)
            v_real = voltage * np.cos(0)  # 假设相角为0
            v_imag = voltage * np.sin(0)
            
            # 功率特征
            p_load, q_load = self._get_power_measurements(bus_idx, time_step)
        else:
            # 不可观测节点特征置零
            v_real = v_imag = p_load = q_load = 0.0
        
        features.extend([v_real, v_imag, p_load, q_load])
        
        # 2. 空间特征
        x_coord = float(bus['x'])
        y_coord = float(bus['y'])
        
        # 空间抖动（数据增强）
        if self.config.spatial_jitter > 0:
            x_coord += np.random.normal(0, self.config.spatial_jitter)
            y_coord += np.random.normal(0, self.config.spatial_jitter)
            
        features.extend([x_coord, y_coord])
        
        # 3. 结构特征
        volt_level = float(bus['voltLvl'])
        bus_type_encoding = self._encode_bus_type(bus.get('type', 'b'))
        is_observed = float(bus.get('observed', True))
        
        features.extend([volt_level, bus_type_encoding, is_observed])
        
        # 4. 网络拓扑特征（可选）
        degree = self._get_node_degree(bus_idx)
        features.append(degree)
        
        return features
    
    def _get_voltage_measurement(self, bus_idx: int, time_step: int) -> float:
        """获取电压测量值"""
        voltage_col = f'bus_{bus_idx}'
        
        if (self.voltage_ts is not None and 
            voltage_col in self.voltage_ts.columns and 
            time_step < len(self.voltage_ts)):
            return float(self.voltage_ts.iloc[time_step][voltage_col])
        else:
            # 默认标幺值
            return 1.0 + np.random.normal(0, 0.02)
    
    def _get_power_measurements(self, bus_idx: int, time_step: int) -> Tuple[float, float]:
        """获取功率测量值"""
        if self.loads is not None:
            load_data = self.loads[self.loads['bus'] == bus_idx]
            if not load_data.empty:
                p_load = float(load_data.iloc[0].get('p_mw', 0))
                q_load = float(load_data.iloc[0].get('q_mvar', 0))
                return p_load, q_load
        
        # 模拟负载数据
        p_load = np.random.normal(0.05, 0.02)
        q_load = np.random.normal(0.02, 0.01)
        return p_load, q_load
    
    def _encode_bus_type(self, bus_type: str) -> float:
        """编码节点类型"""
        type_mapping = {'b': 0, 'db': 1, 'auxiliary': 2}
        return float(type_mapping.get(bus_type, 0))
    
    def _get_node_degree(self, bus_idx: int) -> float:
        """获取节点度数（在真实拓扑中）"""
        if self.lines is None:
            return 0.0
            
        degree = 0
        for _, line in self.lines.iterrows():
            if (line.get('in_service', True) and 
                (line.get('from_bus') == bus_idx or line.get('to_bus') == bus_idx)):
                degree += 1
                
        return float(degree)
    
    def build_candidate_graph(self, k: Optional[int] = None, 
                            method: str = 'knn') -> Tuple[torch.Tensor, torch.Tensor]:
        """构建候选图"""
        if k is None:
            k = self.config.candidate_k_neighbors
            
        if self._candidate_graph_cache is None:
            coords = self.buses[['x', 'y']].values
            edge_index, edge_distances = create_candidate_graph(
                coords, k=k, method=method
            )
            
            # 确保图连通性
            if edge_index.size(1) > 0:
                edge_index = ensure_connected_graph(edge_index, len(self.buses))
            
            self._candidate_graph_cache = (
                torch.tensor(edge_index, dtype=torch.long),
                torch.tensor(edge_distances, dtype=torch.float32).unsqueeze(1)
            )
        
        return self._candidate_graph_cache
    
    def get_true_topology(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取真实拓扑结构"""
        if self._true_topology_cache is not None:
            return self._true_topology_cache
            
        true_edges = []
        true_params = []
        
        for _, line in self.lines.iterrows():
            if (line.get('in_service', True) and 
                pd.notna(line.get('from_bus')) and 
                pd.notna(line.get('to_bus'))):
                
                from_bus = int(line['from_bus'])
                to_bus = int(line['to_bus'])
                
                # 确保节点索引有效
                if from_bus < len(self.buses) and to_bus < len(self.buses):
                    true_edges.append([from_bus, to_bus])
                    
                    # 计算线路参数
                    r_per_km = float(line.get('r_ohm_per_km', 0.1))
                    x_per_km = float(line.get('x_ohm_per_km', 0.1))
                    length = float(line.get('length_km', 0.1))
                    
                    r_total = r_per_km * length
                    x_total = x_per_km * length
                    true_params.append([r_total, x_total])
        
        if true_edges:
            true_edge_index = torch.tensor(true_edges, dtype=torch.long).t().contiguous()
            true_edge_params = torch.tensor(true_params, dtype=torch.float32)
        else:
            true_edge_index = torch.empty((2, 0), dtype=torch.long)
            true_edge_params = torch.empty((0, 2), dtype=torch.float32)
        
        self._true_topology_cache = (true_edge_index, true_edge_params)
        return self._true_topology_cache
    
    def create_graph_data(self, time_step: int = 0) -> Data:
        """创建PyTorch Geometric图数据对象"""
        # 节点特征
        x = self.build_node_features(time_step)
        
        # 候选边
        edge_index, edge_attr = self.build_candidate_graph()
        
        # 真实拓扑（用于训练）
        true_edge_index, true_edge_params = self.get_true_topology()
        
        # 创建数据对象
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            true_edge_index=true_edge_index,
            true_edge_params=true_edge_params,
            num_nodes=len(self.buses)
        )
        
        return data
    
    def get_data_info(self) -> Dict[str, Union[int, float]]:
        """获取数据集信息"""
        if self.buses is None:
            return {}
            
        info = {
            'num_nodes': len(self.buses),
            'num_lines': len(self.lines) if self.lines is not None else 0,
            'num_time_steps': len(self.voltage_ts) if self.voltage_ts is not None else 0,
            'observable_ratio': self.buses.get('observed', True).mean() if 'observed' in self.buses.columns else 1.0,
            'feature_dim': len(self._extract_node_features(self.buses.iloc[0], 0, 0)),
            'spatial_extent_x': self.buses['x'].max() - self.buses['x'].min(),
            'spatial_extent_y': self.buses['y'].max() - self.buses['y'].min(),
        }
        
        return info
    
    def clear_cache(self) -> None:
        """清除所有缓存"""
        self._node_features_cache.clear()
        self._candidate_graph_cache = None
        self._true_topology_cache = None