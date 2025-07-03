#!/usr/bin/env python3
"""
模型组件测试
"""

import unittest
import torch
import numpy as np
from torch_geometric.data import Data

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from power_grid_topology.config.base_config import ModelConfig, PhysicsConfig
from power_grid_topology.models.graph_transformer import PhysicsGraphTransformer, create_model
from power_grid_topology.models.physics_loss import PhysicsConstrainedLoss


class TestGraphTransformer(unittest.TestCase):
    """图变换器模型测试"""
    
    def setUp(self):
        """测试初始化"""
        self.config = ModelConfig()
        self.config.d_input = 10
        self.config.d_hidden = 64
        self.config.n_heads = 4
        self.config.n_layers = 2
        self.config.dropout = 0.1
        
        self.model = PhysicsGraphTransformer(self.config)
        
        # 创建测试数据
        self.num_nodes = 10
        self.num_edges = 15
        
        self.test_data = Data(
            x=torch.randn(self.num_nodes, self.config.d_input),
            edge_index=torch.randint(0, self.num_nodes, (2, self.num_edges)),
            edge_attr=torch.randn(self.num_edges, 1),
            num_nodes=self.num_nodes
        )
    
    def test_model_initialization(self):
        """测试模型初始化"""
        self.assertIsInstance(self.model, PhysicsGraphTransformer)
        self.assertEqual(self.model.config.d_hidden, 64)
        self.assertEqual(self.model.config.n_heads, 4)
    
    def test_forward_pass(self):
        """测试前向传播"""
        self.model.eval()
        
        with torch.no_grad():
            edge_logits, edge_params = self.model(self.test_data)
        
        # 检查输出形状
        self.assertEqual(edge_logits.shape, (self.num_edges,))
        self.assertEqual(edge_params.shape, (self.num_edges, 2))
        
        # 检查输出值范围
        self.assertTrue(torch.all(edge_params > 0))  # 参数应该为正
    
    def test_model_parameters(self):
        """测试模型参数"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.assertGreater(total_params, 0)
        self.assertEqual(total_params, trainable_params)  # 所有参数都应该可训练
    
    def test_predict_topology(self):
        """测试拓扑预测"""
        self.model.eval()
        
        predictions = self.model.predict_topology(self.test_data, threshold=0.5)
        
        # 检查返回的键
        expected_keys = {'edge_index', 'edge_params', 'edge_probs', 'all_probs', 'all_params'}
        self.assertTrue(expected_keys.issubset(predictions.keys()))
        
        # 检查预测边的数量
        predicted_edges = predictions['edge_index']
        self.assertLessEqual(predicted_edges.size(1), self.num_edges)
    
    def test_different_input_sizes(self):
        """测试不同输入大小"""
        # 测试不同节点数量
        for num_nodes in [5, 10, 20]:
            num_edges = min(15, num_nodes * 2)
            
            test_data = Data(
                x=torch.randn(num_nodes, self.config.d_input),
                edge_index=torch.randint(0, num_nodes, (2, num_edges)),
                edge_attr=torch.randn(num_edges, 1),
                num_nodes=num_nodes
            )
            
            with torch.no_grad():
                edge_logits, edge_params = self.model(test_data)
            
            self.assertEqual(edge_logits.shape, (num_edges,))
            self.assertEqual(edge_params.shape, (num_edges, 2))
    
    def test_gradient_flow(self):
        """测试梯度流"""
        self.model.train()
        
        # 前向传播
        edge_logits, edge_params = self.model(self.test_data)
        
        # 简单损失
        loss = edge_logits.mean() + edge_params.mean()
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        has_gradients = False
        for param in self.model.parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_gradients = True
                break
        
        self.assertTrue(has_gradients, "模型应该有非零梯度")


class TestPhysicsLoss(unittest.TestCase):
    """物理约束损失函数测试"""
    
    def setUp(self):
        """测试初始化"""
        self.config = PhysicsConfig()
        self.loss_fn = PhysicsConstrainedLoss(self.config)
        
        # 创建测试数据
        self.num_nodes = 8
        self.num_edges = 10
        
        self.test_data = Data(
            x=torch.randn(self.num_nodes, 9),  # 节点特征
            edge_index=torch.randint(0, self.num_nodes, (2, self.num_edges)),
            edge_attr=torch.randn(self.num_edges, 1),
            true_edge_index=torch.randint(0, self.num_nodes, (2, 5)),
            true_edge_params=torch.rand(5, 2) + 0.1,  # 确保为正
            num_nodes=self.num_nodes
        )
        
        # 模拟预测结果
        self.predictions = {
            'edge_logits': torch.randn(self.num_edges),
            'edge_params': torch.rand(self.num_edges, 2) + 0.1
        }
    
    def test_loss_computation(self):
        """测试损失计算"""
        # 创建目标标签
        targets = {
            'edge_labels': torch.randint(0, 2, (self.num_edges,)).float(),
            'edge_params': self.test_data.true_edge_params
        }
        
        losses = self.loss_fn(self.predictions, targets, self.test_data)
        
        # 检查损失项
        expected_keys = {'total', 'topology', 'parameter', 'kcl', 'kvl', 'radial', 'sparsity', 'geographic'}
        self.assertTrue(expected_keys.issubset(losses.keys()))
        
        # 检查损失值
        for key, loss_value in losses.items():
            self.assertIsInstance(loss_value, torch.Tensor)
            self.assertTrue(torch.isfinite(loss_value))
            self.assertGreaterEqual(loss_value.item(), 0)  # 损失应该非负
    
    def test_edge_label_creation(self):
        """测试边标签创建"""
        candidate_edges = self.test_data.edge_index
        true_edges = self.test_data.true_edge_index
        
        labels = self.loss_fn.create_edge_labels(candidate_edges, true_edges)
        
        self.assertEqual(labels.shape, (self.num_edges,))
        self.assertTrue(torch.all((labels == 0) | (labels == 1)))  # 标签应该是0或1
    
    def test_physics_constraints(self):
        """测试物理约束"""
        edge_probs = torch.sigmoid(self.predictions['edge_logits'])
        edge_params = self.predictions['edge_params']
        
        # 测试KCL约束
        kcl_loss = self.loss_fn.kcl_loss(
            self.test_data.x, self.test_data.edge_index, edge_probs, edge_params
        )
        self.assertIsInstance(kcl_loss, torch.Tensor)
        self.assertGreaterEqual(kcl_loss.item(), 0)
        
        # 测试KVL约束
        kvl_loss = self.loss_fn.kvl_loss(
            self.test_data.x, self.test_data.edge_index, edge_probs, edge_params
        )
        self.assertIsInstance(kvl_loss, torch.Tensor)
        self.assertGreaterEqual(kvl_loss.item(), 0)
    
    def test_zero_edges_case(self):
        """测试无边情况"""
        empty_data = Data(
            x=torch.randn(self.num_nodes, 9),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 1)),
            true_edge_index=torch.empty((2, 0), dtype=torch.long),
            true_edge_params=torch.empty((0, 2)),
            num_nodes=self.num_nodes
        )
        
        empty_predictions = {
            'edge_logits': torch.empty(0),
            'edge_params': torch.empty((0, 2))
        }
        
        targets = {
            'edge_labels': torch.empty(0),
            'edge_params': torch.empty((0, 2))
        }
        
        losses = self.loss_fn(empty_predictions, targets, empty_data)
        
        # 损失应该是有限的
        for key, loss_value in losses.items():
            self.assertTrue(torch.isfinite(loss_value))


class TestModelFactory(unittest.TestCase):
    """模型工厂函数测试"""
    
    def test_create_standard_model(self):
        """测试创建标准模型"""
        config = ModelConfig()
        config.d_input = 10
        config.model_type = 'standard'
        
        model = create_model(config, input_dim=10)
        
        self.assertIsInstance(model, PhysicsGraphTransformer)
        self.assertEqual(model.config.d_input, 10)
    
    def test_create_hybrid_model(self):
        """测试创建混合模型"""
        config = ModelConfig()
        config.d_input = 10
        config.model_type = 'hybrid'
        
        try:
            from power_grid_topology.models.graph_transformer import HybridGraphTransformer
            model = create_model(config, input_dim=10)
            self.assertIsInstance(model, HybridGraphTransformer)
        except ImportError:
            # 如果混合模型未实现，跳过测试
            self.skipTest("混合模型未实现")
    
    def test_model_parameter_counting(self):
        """测试参数计数"""
        from power_grid_topology.models.graph_transformer import count_parameters
        
        config = ModelConfig()
        config.d_input = 10
        model = create_model(config, input_dim=10)
        
        param_count = count_parameters(model)
        self.assertIsInstance(param_count, int)
        self.assertGreater(param_count, 0)


class TestModelIntegration(unittest.TestCase):
    """模型集成测试"""
    
    def test_end_to_end_training_step(self):
        """测试端到端训练步骤"""
        # 创建模型
        config = ModelConfig()
        config.d_input = 9
        model = create_model(config, input_dim=9)
        
        # 创建损失函数
        physics_config = PhysicsConfig()
        loss_fn = PhysicsConstrainedLoss(physics_config)
        
        # 创建优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # 创建测试数据
        num_nodes = 10
        num_edges = 15
        
        data = Data(
            x=torch.randn(num_nodes, 9),
            edge_index=torch.randint(0, num_nodes, (2, num_edges)),
            edge_attr=torch.randn(num_edges, 1),
            true_edge_index=torch.randint(0, num_nodes, (2, 8)),
            true_edge_params=torch.rand(8, 2) + 0.1,
            num_nodes=num_nodes
        )
        
        # 训练步骤
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        edge_logits, edge_params = model(data)
        
        # 计算损失
        predictions = {
            'edge_logits': edge_logits,
            'edge_params': edge_params
        }
        
        targets = {
            'edge_labels': loss_fn.create_edge_labels(data.edge_index, data.true_edge_index),
            'edge_params': data.true_edge_params
        }
        
        losses = loss_fn(predictions, targets, data)
        
        # 反向传播
        losses['total'].backward()
        optimizer.step()
        
        # 检查训练是否成功
        self.assertTrue(torch.isfinite(losses['total']))
        self.assertGreater(losses['total'].item(), 0)
    
    def test_model_saving_loading(self):
        """测试模型保存和加载"""
        import tempfile
        import os
        
        # 创建模型
        config = ModelConfig()
        config.d_input = 9
        model = create_model(config, input_dim=9)
        
        # 获取原始参数
        original_state = model.state_dict()
        
        # 保存模型
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
            torch.save(model.state_dict(), tmp_file.name)
            
            # 创建新模型并加载参数
            new_model = create_model(config, input_dim=9)
            new_model.load_state_dict(torch.load(tmp_file.name))
            
            # 比较参数
            for key in original_state.keys():
                self.assertTrue(torch.allclose(original_state[key], new_model.state_dict()[key]))
            
            # 清理临时文件
            os.unlink(tmp_file.name)
    
    def test_model_inference_mode(self):
        """测试模型推理模式"""
        config = ModelConfig()
        config.d_input = 9
        model = create_model(config, input_dim=9)
        
        # 创建测试数据
        data = Data(
            x=torch.randn(5, 9),
            edge_index=torch.randint(0, 5, (2, 8)),
            edge_attr=torch.randn(8, 1),
            num_nodes=5
        )
        
        # 测试训练模式
        model.train()
        with torch.no_grad():
            train_output = model(data)
        
        # 测试评估模式
        model.eval()
        with torch.no_grad():
            eval_output = model(data)
        
        # 输出应该有相同的形状
        self.assertEqual(train_output[0].shape, eval_output[0].shape)
        self.assertEqual(train_output[1].shape, eval_output[1].shape)


if __name__ == '__main__':
    # 设置随机种子以确保测试可重现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行测试
    unittest.main(verbosity=2)