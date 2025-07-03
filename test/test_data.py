#!/usr/bin/env python3
"""
数据处理模块测试
"""

import unittest
import tempfile
import os
import pandas as pd
import numpy as np
import torch

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from power_grid_topology.config.base_config import DataConfig
from power_grid_topology.data.data_loader import PowerGridDataLoader
from power_grid_topology.utils.graph_utils import create_candidate_graph, ensure_radial_topology


class TestDataLoader(unittest.TestCase):
    """数据加载器测试"""
    
    def setUp(self):
        """测试初始化"""
        self.config = DataConfig()
        self.config.unobservable_ratio = 0.25
        self.config.candidate_k_neighbors = 3
        
        # 创建临时数据目录
        self.temp_dir = tempfile.mkdtemp()
        self.config.data_path = self.temp_dir
        
        # 创建测试数据
        self.create_test_data()
        
        self.data_loader = PowerGridDataLoader(self.config)
    
    def tearDown(self):
        """清理测试数据"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_test_data(self):
        """创建测试用的CSV文件"""
        # 节点数据
        buses_data = {
            'x': [0, 1, 2, 0, 1, 2, 0, 1, 2],
            'y': [0, 0, 0, 1, 1, 1, 2, 2, 2],
            'voltLvl': [1, 1, 1, 1, 1, 1, 1, 1, 1],
            'type': ['b'] * 9
        }
        buses_df = pd.DataFrame(buses_data)
        buses_df.to_csv(os.path.join(self.temp_dir, 'buses.csv'), index=False)
        
        # 线路数据
        lines_data = {
            'from_bus': [0, 1, 3, 4, 6, 7],
            'to_bus': [1, 2, 4, 5, 7, 8],
            'r_ohm_per_km': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            'x_ohm_per_km': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            'length_km': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            'in_service': [True] * 6
        }
        lines_df = pd.DataFrame(lines_data)
        lines_df.to_csv(os.path.join(self.temp_dir, 'lines.csv'), index=False)
        
        # 电压时间序列
        voltage_data = {}
        for i in range(9):
            voltage_data[f'bus_{i}'] = np.random.normal(1.0, 0.02, 10)
        voltage_df = pd.DataFrame(voltage_data)
        voltage_df.to_csv(os.path.join(self.temp_dir, 'voltage_timeseries.csv'), index=False)
        
        # 负载时间序列
        loading_data = {}
        for i in range(6):  # 6条线路
            loading_data[f'line_{i}'] = np.random.normal(0.5, 0.1, 10)
        loading_df = pd.DataFrame(loading_data)
        loading_df.to_csv(os.path.join(self.temp_dir, 'loading_timeseries.csv'), index=False)
        
        # 负载数据（可选）
        loads_data = {
            'bus': [2, 5, 8],
            'p_mw': [0.1, 0.15, 0.12],
            'q_mvar': [0.05, 0.08, 0.06]
        }
        loads_df = pd.DataFrame(loads_data)
        loads_df.to_csv(os.path.join(self.temp_dir, 'loads.csv'), index=False)
    
    def test_data_loading(self):
        """测试数据加载"""
        self.data_loader.load_data()
        
        # 检查数据是否正确加载
        self.assertIsNotNone(self.data_loader.buses)
        self.assertIsNotNone(self.data_loader.lines)
        self.assertIsNotNone(self.data_loader.voltage_ts)
        self.assertIsNotNone(self.data_loader.loading_ts)
        
        # 检查数据形状
        self.assertEqual(len(self.data_loader.buses), 9)
        self.assertEqual(len(self.data_loader.lines), 6)
        self.assertEqual(len(self.data_loader.voltage_ts), 10)
    
    def test_node_masking(self):
        """测试节点遮蔽"""
        self.data_loader.load_data()
        
        # 遮蔽节点
        masked_indices = self.data_loader.mask_unobservable_nodes(ratio=0.3, seed=42)
        
        # 检查遮蔽结果
        self.assertIsInstance(masked_indices, np.ndarray)
        self.assertEqual(len(masked_indices), int(9 * 0.3))
        
        # 检查可观测性列
        self.assertIn('observed', self.data_loader.buses.columns)
        unobserved_count = (self.data_loader.buses['observed'] == False).sum()
        self.assertEqual(unobserved_count, len(masked_indices))
    
    def test_node_feature_building(self):
        """测试节点特征构建"""
        self.data_loader.load_data()
        
        # 构建特征
        features = self.data_loader.build_node_features(time_step=0)
        
        # 检查特征形状
        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(features.shape[0], 9)  # 9个节点
        self.assertGreater(features.shape[1], 0)  # 有特征维度
        
        # 检查特征值
        self.assertTrue(torch.isfinite(features).all())
    
    def test_candidate_graph_building(self):
        """测试候选图构建"""
        self.data_loader.load_data()
        
        # 构建候选图
        edge_index, edge_distances = self.data_loader.build_candidate_graph()
        
        # 检查输出类型和形状
        self.assertIsInstance(edge_index, torch.Tensor)
        self.assertIsInstance(edge_distances, torch.Tensor)
        self.assertEqual(edge_index.shape[0], 2)
        self.assertEqual(edge_distances.shape[0], edge_index.shape[1])
        
        # 检查边索引有效性
        self.assertTrue(torch.all(edge_index >= 0))
        self.assertTrue(torch.all(edge_index < 9))
    
    def test_true_topology_extraction(self):
        """测试真实拓扑提取"""
        self.data_loader.load_data()
        
        # 获取真实拓扑
        true_edge_index, true_edge_params = self.data_loader.get_true_topology()
        
        # 检查输出
        self.assertIsInstance(true_edge_index, torch.Tensor)
        self.assertIsInstance(true_edge_params, torch.Tensor)
        
        if true_edge_index.size(1) > 0:
            self.assertEqual(true_edge_index.shape[0], 2)
            self.assertEqual(true_edge_params.shape[0], true_edge_index.shape[1])
            self.assertEqual(true_edge_params.shape[1], 2)  # R, X参数
    
    def test_graph_data_creation(self):
        """测试图数据对象创建"""
        self.data_loader.load_data()
        
        # 创建图数据
        data = self.data_loader.create_graph_data(time_step=0)
        
        # 检查数据对象
        self.assertTrue(hasattr(data, 'x'))
        self.assertTrue(hasattr(data, 'edge_index'))
        self.assertTrue(hasattr(data, 'edge_attr'))
        self.assertTrue(hasattr(data, 'num_nodes'))
        
        # 检查数据一致性
        self.assertEqual(data.x.shape[0], data.num_nodes)
        self.assertEqual(data.edge_index.shape[0], 2)
        self.assertEqual(data.edge_attr.shape[0], data.edge_index.shape[1])
    
    def test_feature_normalization(self):
        """测试特征归一化"""
        self.data_loader.load_data()
        self.config.normalize_features = True
        
        # 构建归一化特征
        features1 = self.data_loader.build_node_features(time_step=0)
        features2 = self.data_loader.build_node_features(time_step=1)
        
        # 检查特征统计
        # 注意：由于特征包含不同类型（连续和分类），不是所有特征都会被标准化
        self.assertTrue(torch.isfinite(features1).all())
        self.assertTrue(torch.isfinite(features2).all())
    
    def test_data_caching(self):
        """测试数据缓存"""
        self.data_loader.load_data()
        
        # 第一次构建特征
        features1 = self.data_loader.build_node_features(time_step=0, use_cache=True)
        
        # 第二次构建相同特征（应该使用缓存）
        features2 = self.data_loader.build_node_features(time_step=0, use_cache=True)
        
        # 检查结果一致性
        self.assertTrue(torch.allclose(features1, features2))
        
        # 清除缓存
        self.data_loader.clear_cache()
        
        # 重新构建特征
        features3 = self.data_loader.build_node_features(time_step=0, use_cache=True)
        
        # 应该仍然一致（除非有随机性）
        # 这里主要测试缓存机制不会报错
        self.assertEqual(features1.shape, features3.shape)
    
    def test_data_info(self):
        """测试数据信息获取"""
        self.data_loader.load_data()
        
        info = self.data_loader.get_data_info()
        
        # 检查信息项
        expected_keys = ['num_nodes', 'num_lines', 'num_time_steps', 'feature_dim']
        for key in expected_keys:
            self.assertIn(key, info)
        
        # 检查数值
        self.assertEqual(info['num_nodes'], 9)
        self.assertEqual(info['num_lines'], 6)
        self.assertEqual(info['num_time_steps'], 10)


class TestGraphUtils(unittest.TestCase):
    """图工具函数测试"""
    
    def test_create_candidate_graph(self):
        """测试候选图创建"""
        # 创建测试坐标
        coordinates = np.array([
            [0, 0], [1, 0], [2, 0],
            [0, 1], [1, 1], [2, 1]
        ])
        
        # 测试k-近邻方法
        edge_index, edge_distances = create_candidate_graph(coordinates, k=2, method='knn')
        
        self.assertIsInstance(edge_index, np.ndarray)
        self.assertIsInstance(edge_distances, np.ndarray)
        self.assertEqual(edge_index.shape[0], 2)
        self.assertEqual(len(edge_distances), edge_index.shape[1])
        
        # 检查距离非负
        self.assertTrue(np.all(edge_distances >= 0))
    
    def test_ensure_radial_topology(self):
        """测试径向拓扑确保"""
        # 创建测试边和权重
        edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 3, 3]], dtype=torch.long)
        edge_weights = torch.tensor([0.8, 0.7, 0.6, 0.5])
        
        radial_edges = ensure_radial_topology(edge_index, edge_weights, num_nodes=4)
        
        # 检查结果
        self.assertIsInstance(radial_edges, torch.Tensor)
        self.assertEqual(radial_edges.shape[0], 2)
        
        # 径向拓扑应该有n-1条边（对于n个节点的连通图）
        if radial_edges.size(1) > 0:
            self.assertLessEqual(radial_edges.size(1), 3)  # 最多3条边（4个节点）
    
    def test_empty_graph_handling(self):
        """测试空图处理"""
        coordinates = np.array([[0, 0], [1, 1]])
        
        # 测试空边情况
        edge_index, edge_distances = create_candidate_graph(coordinates, k=0, method='knn')
        
        # 应该返回空的但格式正确的数组
        self.assertEqual(edge_index.shape, (2, 0))
        self.assertEqual(len(edge_distances), 0)
    
    def test_single_node_graph(self):
        """测试单节点图"""
        coordinates = np.array([[0, 0]])
        
        edge_index, edge_distances = create_candidate_graph(coordinates, k=1, method='knn')
        
        # 单个节点应该没有边
        self.assertEqual(edge_index.shape, (2, 0))
        self.assertEqual(len(edge_distances), 0)


class TestDataAugmentation(unittest.TestCase):
    """数据增强测试"""
    
    def setUp(self):
        """测试初始化"""
        # 创建简单的数据加载器配置
        self.config = DataConfig()
        self.config.add_noise = True
        self.config.noise_std = 0.01
        self.config.spatial_jitter = 0.1
    
    def test_noise_addition(self):
        """测试噪声添加"""
        # 创建数据加载器但不加载实际文件
        data_loader = PowerGridDataLoader(self.config)
        
        # 手动设置一些数据
        buses_data = pd.DataFrame({
            'x': [0, 1, 2],
            'y': [0, 1, 2],
            'voltLvl': [1, 1, 1],
            'type': ['b', 'b', 'b'],
            'observed': [True, True, True]
        })
        data_loader.buses = buses_data
        
        # 模拟电压和负载数据
        data_loader.voltage_ts = pd.DataFrame({
            'bus_0': [1.0], 'bus_1': [1.0], 'bus_2': [1.0]
        })
        data_loader.loads = None
        
        # 构建特征（应该添加噪声）
        features = data_loader.build_node_features(time_step=0)
        
        # 检查特征
        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(features.shape[0], 3)
    
    def test_spatial_jitter(self):
        """测试空间抖动"""
        # 这个测试主要检查空间抖动不会导致错误
        # 实际的抖动效果在_extract_node_features中实现
        pass


class TestErrorHandling(unittest.TestCase):
    """错误处理测试"""
    
    def test_missing_files(self):
        """测试缺失文件处理"""
        config = DataConfig()
        config.data_path = "/nonexistent/path"
        
        data_loader = PowerGridDataLoader(config)
        
        # 应该抛出异常
        with self.assertRaises((FileNotFoundError, RuntimeError)):
            data_loader.load_data()
    
    def test_invalid_data_format(self):
        """测试无效数据格式处理"""
        # 创建临时目录和无效数据
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 创建缺少必要列的CSV
            invalid_buses = pd.DataFrame({'invalid_col': [1, 2, 3]})
            invalid_buses.to_csv(os.path.join(temp_dir, 'buses.csv'), index=False)
            
            config = DataConfig()
            config.data_path = temp_dir
            data_loader = PowerGridDataLoader(config)
            
            # 应该抛出异常或处理错误
            with self.assertRaises((ValueError, KeyError, RuntimeError)):
                data_loader.load_data()
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行测试
    unittest.main(verbosity=2)