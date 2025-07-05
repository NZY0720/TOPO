#!/usr/bin/env python3
"""
Data loading and preprocessing for Power Grid Topology Reconstruction
Updated for actual data format
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
    """电力网格数据加载器 - 适配实际数据格式"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.buses = None
        self.lines = None
        self.voltage_ts = None
        self.loading_ts = None
        self.loads = None
        self.generators = None
        self.final_state = None
        self.scaler = None
        
        # 缓存的特征
        self._node_features_cache = {}
        self._candidate_graph_cache = None
        self._true_topology_cache = None
        
        # 实际数据文件名映射
        self.file_mapping = {
            'buses': '1MVurban0sw_bus_with_local_coords.csv',
            'lines': '1MVurban0sw_lines_with_coordinates.csv', 
            'voltage_ts': '1MVurban0sw_voltage_timeseries.csv',
            'loading_ts': '1MVurban0sw_loading_timeseries.csv',
            'loads': '1MVurban0sw_loads_with_coordinates.csv',
            'generators': '1MVurban0sw_generators_with_coordinates.csv',
            'final_state': '1MVurban0sw_final_state_with_coords.csv',
            'summary_stats': '1MVurban0sw_summary_statistics.csv',
            'original_coords': '1MVurban0sw_original_coordinates.csv'
        }
        
    def load_data(self, data_path: Optional[str] = None) -> None:
        """加载所有数据文件"""
        if data_path is None:
            data_path = self.config.data_path
            
        self.data_path = data_path
        
        try:
            # 1. 加载节点数据 (buses)
            buses_file = os.path.join(data_path, self.file_mapping['buses'])
            self.buses = load_csv_safe(buses_file)
            print(f"✅ 节点数据加载完成: {len(self.buses)} 个节点")
            
            # 2. 加载线路数据 (lines)
            lines_file = os.path.join(data_path, self.file_mapping['lines'])
            self.lines = load_csv_safe(lines_file)
            print(f"✅ 线路数据加载完成: {len(self.lines)} 条线路")
            
            # 3. 加载电压时间序列
            voltage_file = os.path.join(data_path, self.file_mapping['voltage_ts'])
            self.voltage_ts = load_csv_safe(voltage_file)
            print(f"✅ 电压时间序列加载完成: {len(self.voltage_ts)} 个时间步")
            
            # 4. 加载负载时间序列
            loading_file = os.path.join(data_path, self.file_mapping['loading_ts'])
            self.loading_ts = load_csv_safe(loading_file)
            print(f"✅ 负载时间序列加载完成: {len(self.loading_ts)} 个时间步")
            
            # 5. 加载负载数据 (可选)
            loads_file = os.path.join(data_path, self.file_mapping['loads'])
            if os.path.exists(loads_file):
                self.loads = load_csv_safe(loads_file)
                print(f"✅ 负载数据加载完成: {len(self.loads)} 个负载点")
            
            # 6. 加载发电机数据 (可选)
            gen_file = os.path.join(data_path, self.file_mapping['generators'])
            if os.path.exists(gen_file):
                self.generators = load_csv_safe(gen_file)
                print(f"✅ 发电机数据加载完成: {len(self.generators)} 个发电机")
            
            # 7. 加载最终状态数据 (可选)
            final_file = os.path.join(data_path, self.file_mapping['final_state'])
            if os.path.exists(final_file):
                self.final_state = load_csv_safe(final_file)
                print(f"✅ 最终状态数据加载完成: {len(self.final_state)} 个状态")
            
            # 数据验证和清理
            self._validate_and_clean_data()
            
            print(f"\n📊 数据加载总结:")
            print(f"   - 节点数: {len(self.buses)}")
            print(f"   - 线路数: {len(self.lines)}")
            print(f"   - 时间步数: {len(self.voltage_ts)}")
            print(f"   - 负载点数: {len(self.loads) if self.loads is not None else 0}")
            print(f"   - 发电机数: {len(self.generators) if self.generators is not None else 0}")
            
        except Exception as e:
            raise RuntimeError(f"数据加载失败: {e}")
    
    def _validate_and_clean_data(self) -> None:
        """验证和清理数据"""
        print("🔧 数据验证和清理...")
        
        # 1. 验证节点数据
        if self.buses is not None:
            # 检查必要的列
            required_bus_cols = ['x', 'y', 'vn_kv', 'type']
            missing_cols = [col for col in required_bus_cols if col not in self.buses.columns]
            if missing_cols:
                print(f"⚠️  节点数据缺少列: {missing_cols}")
            
            # 数据清理
            self.buses['x'] = pd.to_numeric(self.buses['x'], errors='coerce').fillna(0.0)
            self.buses['y'] = pd.to_numeric(self.buses['y'], errors='coerce').fillna(0.0)
            self.buses['vn_kv'] = pd.to_numeric(self.buses['vn_kv'], errors='coerce').fillna(1.0)
            self.buses['voltLvl'] = self.buses.get('voltLvl', 1).fillna(1)
            self.buses['type'] = self.buses.get('type', 'b').fillna('b')
            
            # 重置索引确保连续性
            self.buses = self.buses.reset_index(drop=True)
            print(f"   ✅ 节点数据验证完成: {len(self.buses)} 个有效节点")
        
        # 2. 验证线路数据
        if self.lines is not None:
            required_line_cols = ['from_bus', 'to_bus', 'in_service']
            missing_cols = [col for col in required_line_cols if col not in self.lines.columns]
            if missing_cols:
                print(f"⚠️  线路数据缺少列: {missing_cols}")
            
            # 数据清理
            self.lines['from_bus'] = pd.to_numeric(self.lines['from_bus'], errors='coerce')
            self.lines['to_bus'] = pd.to_numeric(self.lines['to_bus'], errors='coerce')
            self.lines['in_service'] = self.lines.get('in_service', True).fillna(True)
            
            # 线路参数处理
            self.lines['r_ohm_per_km'] = pd.to_numeric(
                self.lines.get('r_ohm_per_km', 0.1), errors='coerce'
            ).fillna(0.1)
            self.lines['x_ohm_per_km'] = pd.to_numeric(
                self.lines.get('x_ohm_per_km', 0.1), errors='coerce'
            ).fillna(0.1)
            self.lines['length_km'] = pd.to_numeric(
                self.lines.get('length_km', 0.1), errors='coerce'
            ).fillna(0.1)
            
            # 移除无效线路
            invalid_lines = (
                self.lines['from_bus'].isna() | 
                self.lines['to_bus'].isna() |
                (self.lines['from_bus'] >= len(self.buses)) |
                (self.lines['to_bus'] >= len(self.buses))
            )
            
            if invalid_lines.any():
                print(f"⚠️  移除 {invalid_lines.sum()} 条无效线路")
                self.lines = self.lines[~invalid_lines].reset_index(drop=True)
            
            print(f"   ✅ 线路数据验证完成: {len(self.lines)} 条有效线路")
        
        # 3. 验证时间序列数据
        if self.voltage_ts is not None:
            # 检查节点电压列是否存在
            expected_bus_cols = [f'bus_{i}' for i in range(len(self.buses))]
            available_bus_cols = [col for col in expected_bus_cols if col in self.voltage_ts.columns]
            
            if len(available_bus_cols) < len(self.buses):
                print(f"⚠️  电压时间序列缺少部分节点数据")
                print(f"     期望: {len(self.buses)} 个节点，实际: {len(available_bus_cols)} 个节点")
            
            print(f"   ✅ 电压时间序列验证完成: {len(self.voltage_ts)} 个时间步")
        
        # 4. 验证负载数据
        if self.loads is not None:
            # 确保负载的bus索引有效
            valid_load_buses = self.loads['bus'] < len(self.buses)
            if not valid_load_buses.all():
                print(f"⚠️  移除 {(~valid_load_buses).sum()} 个无效负载")
                self.loads = self.loads[valid_load_buses].reset_index(drop=True)
            
            print(f"   ✅ 负载数据验证完成: {len(self.loads)} 个负载点")
        
        # 5. 验证发电机数据
        if self.generators is not None:
            # 确保发电机的bus索引有效
            valid_gen_buses = self.generators['bus'] < len(self.buses)
            if not valid_gen_buses.all():
                print(f"⚠️  移除 {(~valid_gen_buses).sum()} 个无效发电机")
                self.generators = self.generators[valid_gen_buses].reset_index(drop=True)
            
            print(f"   ✅ 发电机数据验证完成: {len(self.generators)} 个发电机")
    
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
            # 电压特征（从时间序列获取）
            voltage = self._get_voltage_measurement(bus_idx, time_step)
            
            # 将电压转换为实部虚部（假设相角为0，简化处理）
            v_real = voltage * np.cos(0)  
            v_imag = voltage * np.sin(0)
            
            # 功率特征（从负载数据获取或模拟）
            p_load, q_load = self._get_power_measurements(bus_idx, time_step)
            
            # 发电机功率（如果该节点有发电机）
            p_gen, q_gen = self._get_generator_measurements(bus_idx)
            
        else:
            # 不可观测节点特征置零
            v_real = v_imag = p_load = q_load = p_gen = q_gen = 0.0
        
        features.extend([v_real, v_imag, p_load, q_load, p_gen, q_gen])
        
        # 2. 空间特征
        x_coord = float(bus['x'])
        y_coord = float(bus['y'])
        
        # 空间抖动（数据增强）
        if self.config.spatial_jitter > 0:
            x_coord += np.random.normal(0, self.config.spatial_jitter)
            y_coord += np.random.normal(0, self.config.spatial_jitter)
            
        features.extend([x_coord, y_coord])
        
        # 3. 结构特征
        volt_level = float(bus.get('vn_kv', 1.0))  # 使用实际电压等级
        bus_type_encoding = self._encode_bus_type(bus.get('type', 'b'))
        is_observed = float(bus.get('observed', True))
        
        # 添加子网信息（如果有）
        subnet_encoding = self._encode_subnet(bus.get('subnet', 'default'))
        
        features.extend([volt_level, bus_type_encoding, is_observed, subnet_encoding])
        
        # 4. 网络拓扑特征
        degree = self._get_node_degree(bus_idx)
        features.append(degree)
        
        return features
    
    def _get_voltage_measurement(self, bus_idx: int, time_step: int) -> float:
        """获取电压测量值"""
        voltage_col = f'bus_{bus_idx}'
        
        if (self.voltage_ts is not None and 
            voltage_col in self.voltage_ts.columns and 
            time_step < len(self.voltage_ts)):
            voltage = self.voltage_ts.iloc[time_step][voltage_col]
            return float(voltage) if pd.notna(voltage) else 1.0
        else:
            # 默认标幺值
            return 1.0 + np.random.normal(0, 0.02)
    
    def _get_power_measurements(self, bus_idx: int, time_step: int) -> Tuple[float, float]:
        """获取功率测量值"""
        p_load = q_load = 0.0
        
        if self.loads is not None:
            # 查找该节点的负载
            bus_loads = self.loads[self.loads['bus'] == bus_idx]
            if not bus_loads.empty:
                for _, load in bus_loads.iterrows():
                    p_load += float(load.get('p_mw', 0))
                    q_load += float(load.get('q_mvar', 0))
        
        # 如果没有负载数据，使用小的随机值
        if p_load == 0 and q_load == 0:
            p_load = max(0, np.random.normal(0.02, 0.01))
            q_load = max(0, np.random.normal(0.01, 0.005))
        
        return p_load, q_load
    
    def _get_generator_measurements(self, bus_idx: int) -> Tuple[float, float]:
        """获取发电机功率测量值"""
        p_gen = q_gen = 0.0
        
        if self.generators is not None:
            # 查找该节点的发电机
            bus_gens = self.generators[self.generators['bus'] == bus_idx]
            if not bus_gens.empty:
                for _, gen in bus_gens.iterrows():
                    if gen.get('in_service', True):
                        p_gen += float(gen.get('p_mw', 0))
                        q_gen += float(gen.get('q_mvar', 0))
        
        return p_gen, q_gen
    
    def _encode_bus_type(self, bus_type: str) -> float:
        """编码节点类型"""
        type_mapping = {
            'b': 0,      # 普通节点
            'db': 1,     # 分布式节点  
            'auxiliary': 2,  # 辅助节点
            'n': 0,      # 节点
            'gen': 3,    # 发电机节点
            'load': 4    # 负载节点
        }
        return float(type_mapping.get(str(bus_type).lower(), 0))
    
    def _encode_subnet(self, subnet: str) -> float:
        """编码子网信息"""
        if pd.isna(subnet) or subnet == 'default':
            return 0.0
        
        # 简单的哈希编码
        return float(hash(str(subnet)) % 10)
    
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
            if edge_index.shape[1] > 0:
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
        
        if self.lines is not None:
            for _, line in self.lines.iterrows():
                # 检查线路是否投运且有效
                if (line.get('in_service', True) and 
                    pd.notna(line.get('from_bus')) and 
                    pd.notna(line.get('to_bus'))):
                    
                    from_bus = int(line['from_bus'])
                    to_bus = int(line['to_bus'])
                    
                    # 确保节点索引有效
                    if (from_bus < len(self.buses) and to_bus < len(self.buses) and
                        from_bus >= 0 and to_bus >= 0 and from_bus != to_bus):
                        
                        true_edges.append([from_bus, to_bus])
                        
                        # 计算线路参数
                        r_per_km = float(line.get('r_ohm_per_km', 0.1))
                        x_per_km = float(line.get('x_ohm_per_km', 0.1))
                        length = float(line.get('length_km', 0.1))
                        
                        r_total = r_per_km * length
                        x_total = x_per_km * length
                        
                        # 确保参数为正数
                        r_total = max(r_total, 1e-6)
                        x_total = max(x_total, 1e-6)
                        
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
            'num_loads': len(self.loads) if self.loads is not None else 0,
            'num_generators': len(self.generators) if self.generators is not None else 0,
            'observable_ratio': self.buses.get('observed', True).mean() if 'observed' in self.buses.columns else 1.0,
            'feature_dim': len(self._extract_node_features(self.buses.iloc[0], 0, 0)),
            'spatial_extent_x': self.buses['x'].max() - self.buses['x'].min(),
            'spatial_extent_y': self.buses['y'].max() - self.buses['y'].min(),
        }
        
        # 电压等级分布
        if 'vn_kv' in self.buses.columns:
            info['voltage_levels'] = sorted(self.buses['vn_kv'].unique().tolist())
        
        # 线路长度统计
        if self.lines is not None and 'length_km' in self.lines.columns:
            info['avg_line_length'] = float(self.lines['length_km'].mean())
            info['max_line_length'] = float(self.lines['length_km'].max())
            info['min_line_length'] = float(self.lines['length_km'].min())
        
        return info
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """获取网络统计信息"""
        stats = {}
        
        if self.buses is not None:
            stats['nodes'] = {
                'total': len(self.buses),
                'voltage_levels': self.buses['vn_kv'].value_counts().to_dict() if 'vn_kv' in self.buses.columns else {},
                'node_types': self.buses['type'].value_counts().to_dict() if 'type' in self.buses.columns else {}
            }
        
        if self.lines is not None:
            stats['lines'] = {
                'total': len(self.lines),
                'in_service': self.lines['in_service'].sum() if 'in_service' in self.lines.columns else len(self.lines),
                'avg_length': float(self.lines['length_km'].mean()) if 'length_km' in self.lines.columns else 0,
                'total_length': float(self.lines['length_km'].sum()) if 'length_km' in self.lines.columns else 0
            }
        
        if self.loads is not None:
            stats['loads'] = {
                'total': len(self.loads),
                'total_p': float(self.loads['p_mw'].sum()) if 'p_mw' in self.loads.columns else 0,
                'total_q': float(self.loads['q_mvar'].sum()) if 'q_mvar' in self.loads.columns else 0
            }
        
        if self.generators is not None:
            stats['generators'] = {
                'total': len(self.generators),
                'in_service': self.generators['in_service'].sum() if 'in_service' in self.generators.columns else len(self.generators),
                'total_p': float(self.generators['p_mw'].sum()) if 'p_mw' in self.generators.columns else 0,
                'total_q': float(self.generators['q_mvar'].sum()) if 'q_mvar' in self.generators.columns else 0
            }
        
        return stats
    
    def clear_cache(self) -> None:
        """清除所有缓存"""
        self._node_features_cache.clear()
        self._candidate_graph_cache = None
        self._true_topology_cache = None
    
    def save_processed_data(self, output_path: str) -> None:
        """保存处理后的数据"""
        processed_data = {
            'buses': self.buses,
            'lines': self.lines,
            'loads': self.loads,
            'generators': self.generators,
            'config': self.config.__dict__
        }
        
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(processed_data, f)
        
        print(f"✅ 处理后的数据已保存到: {output_path}")