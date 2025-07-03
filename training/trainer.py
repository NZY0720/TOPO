#!/usr/bin/env python3
"""
Training module for Physics-informed Graph Transformer
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from ..config.base_config import Config, TrainingConfig
from ..data.data_loader import PowerGridDataLoader
from ..models.graph_transformer import PhysicsGraphTransformer
from ..models.physics_loss import PhysicsConstrainedLoss
from ..evaluation.metrics import TopologyMetrics, ParameterMetrics
from ..utils.logging_utils import setup_logger, log_metrics
from ..utils.io_utils import save_checkpoint, load_checkpoint
from ..visualization.training_viz import plot_training_curves


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        self.early_stop = self.counter >= self.patience
        return self.early_stop


class PowerGridTrainer:
    """电力网格拓扑重建训练器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.training.device)
        
        # 初始化组件
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.data_loader = None
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'metrics': []
        }
        
        # 早停
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience
        )
        
        # 设置输出目录
        self.setup_output_directories()
        
        # 设置日志
        self.logger = setup_logger(
            'trainer', 
            os.path.join(config.experiment.log_dir, 'training.log')
        )
        
        # Tensorboard
        if config.experiment.use_tensorboard:
            self.writer = SummaryWriter(
                log_dir=os.path.join(config.experiment.log_dir, 'tensorboard')
            )
        else:
            self.writer = None
            
        # Weights & Biases
        if config.experiment.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=config.experiment.wandb_project,
                    name=config.experiment.name,
                    config=config.to_dict()
                )
                self.use_wandb = True
            except ImportError:
                self.logger.warning("wandb not installed, skipping W&B logging")
                self.use_wandb = False
        else:
            self.use_wandb = False
    
    def setup_output_directories(self):
        """设置输出目录"""
        dirs = [
            self.config.experiment.output_dir,
            self.config.experiment.log_dir,
            self.config.experiment.checkpoint_dir,
            self.config.experiment.figure_dir
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def setup_model(self, input_dim: int):
        """设置模型"""
        from ..models.graph_transformer import create_model
        
        self.config.model.d_input = input_dim
        self.model = create_model(self.config.model, input_dim).to(self.device)
        
        self.logger.info(f"Model created with {self._count_parameters()} parameters")
        
        # 损失函数
        self.criterion = PhysicsConstrainedLoss(self.config.physics)
        
        # 优化器
        self.setup_optimizer()
        
        # 学习率调度器
        self.setup_scheduler()
    
    def setup_optimizer(self):
        """设置优化器"""
        config = self.config.training
        
        if config.optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=config.betas
            )
        elif config.optimizer.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=config.betas
            )
        elif config.optimizer.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    
    def setup_scheduler(self):
        """设置学习率调度器"""
        config = self.config.training
        
        if config.scheduler == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config.scheduler_factor,
                patience=config.scheduler_patience,
                min_lr=config.scheduler_min_lr,
                verbose=True
            )
        elif config.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.epochs,
                eta_min=config.scheduler_min_lr
            )
        elif config.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.epochs // 3,
                gamma=config.scheduler_factor
            )
        else:
            self.scheduler = None
    
    def train_epoch(self, data_loader: PowerGridDataLoader, 
                   time_steps: List[int]) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_losses = {}
        num_batches = len(time_steps)
        
        with tqdm(time_steps, desc=f"Epoch {self.current_epoch}") as pbar:
            for step, time_step in enumerate(pbar):
                # 准备数据
                data = data_loader.create_graph_data(time_step).to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                edge_logits, edge_params = self.model(data)
                
                # 计算损失
                predictions = {
                    'edge_logits': edge_logits,
                    'edge_params': edge_params
                }
                
                targets = {
                    'edge_labels': self.criterion.create_edge_labels(
                        data.edge_index, data.true_edge_index
                    ),
                    'edge_params': data.true_edge_params
                }
                
                losses = self.criterion(predictions, targets, data)
                
                # 反向传播
                losses['total'].backward()
                
                # 梯度裁剪
                if self.config.training.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip_norm
                    )
                
                self.optimizer.step()
                
                # 累计损失
                for key, value in losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0.0
                    epoch_losses[key] += value.item()
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{losses['total'].item():.4f}",
                    'topo': f"{losses['topology'].item():.4f}",
                    'physics': f"{losses['kcl'].item() + losses['kvl'].item():.4f}"
                })
        
        # 平均损失
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        return epoch_losses
    
    def validate_epoch(self, data_loader: PowerGridDataLoader,
                      time_steps: List[int]) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        val_losses = {}
        
        with torch.no_grad():
            for time_step in time_steps[:5]:  # 只用部分时间步验证
                data = data_loader.create_graph_data(time_step).to(self.device)
                
                edge_logits, edge_params = self.model(data)
                
                predictions = {
                    'edge_logits': edge_logits,
                    'edge_params': edge_params
                }
                
                targets = {
                    'edge_labels': self.criterion.create_edge_labels(
                        data.edge_index, data.true_edge_index
                    ),
                    'edge_params': data.true_edge_params
                }
                
                losses = self.criterion(predictions, targets, data)
                
                for key, value in losses.items():
                    if key not in val_losses:
                        val_losses[key] = 0.0
                    val_losses[key] += value.item()
        
        # 平均损失
        num_val_steps = min(5, len(time_steps))
        for key in val_losses:
            val_losses[key] /= num_val_steps
            
        return val_losses
    
    def evaluate_model(self, data_loader: PowerGridDataLoader, 
                      time_step: int = 0) -> Dict[str, float]:
        """评估模型性能"""
        self.model.eval()
        
        with torch.no_grad():
            data = data_loader.create_graph_data(time_step).to(self.device)
            
            # 预测拓扑
            predictions = self.model.predict_topology(
                data, threshold=self.config.evaluation.edge_threshold
            )
            
            # 计算指标
            metrics = {}
            
            # 拓扑指标
            if self.config.evaluation.compute_topology_metrics:
                topo_metrics = TopologyMetrics()
                topo_results = topo_metrics.compute(
                    predictions['edge_index'].cpu(),
                    data.true_edge_index.cpu(),
                    data.num_nodes
                )
                metrics.update(topo_results)
            
            # 参数指标
            if self.config.evaluation.compute_parameter_metrics:
                param_metrics = ParameterMetrics()
                if data.true_edge_params.size(0) > 0:
                    param_results = param_metrics.compute(
                        predictions['edge_params'].cpu(),
                        data.true_edge_params.cpu()
                    )
                    metrics.update(param_results)
        
        return metrics
    
    def train(self, data_loader: PowerGridDataLoader):
        """完整训练流程"""
        self.data_loader = data_loader
        
        # 获取输入维度
        sample_data = data_loader.create_graph_data(0)
        input_dim = sample_data.x.size(1)
        
        # 设置模型
        self.setup_model(input_dim)
        
        # 训练时间步
        time_steps = self.config.data.time_steps or list(range(20))
        val_time_steps = time_steps[:len(time_steps)//4]  # 25%用于验证
        train_time_steps = time_steps[len(time_steps)//4:]  # 75%用于训练
        
        self.logger.info(f"开始训练 {self.config.training.epochs} epochs")
        self.logger.info(f"训练时间步: {len(train_time_steps)}, 验证时间步: {len(val_time_steps)}")
        
        start_time = time.time()
        
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            # 训练
            train_losses = self.train_epoch(data_loader, train_time_steps)
            self.training_history['train_losses'].append(train_losses)
            
            # 验证
            if epoch % self.config.training.validate_every == 0:
                val_losses = self.validate_epoch(data_loader, val_time_steps)
                self.training_history['val_losses'].append(val_losses)
                
                # 评估指标
                metrics = self.evaluate_model(data_loader)
                self.training_history['metrics'].append(metrics)
                
                # 学习率调度
                if self.scheduler and self.config.training.scheduler == 'reduce_on_plateau':
                    self.scheduler.step(val_losses['total'])
                
                # 早停检查
                if self.early_stopping(val_losses['total']):
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                # 保存最佳模型
                if val_losses['total'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total']
                    self.save_checkpoint('best_model.pth')
                
                # 日志记录
                if epoch % self.config.training.log_every == 0:
                    self._log_epoch_results(epoch, train_losses, val_losses, metrics)
            
            # 学习率调度（非plateau类型）
            if self.scheduler and self.config.training.scheduler != 'reduce_on_plateau':
                self.scheduler.step()
            
            # 定期保存检查点
            if epoch % self.config.training.save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
        
        # 训练结束
        end_time = time.time()
        training_time = end_time - start_time
        
        self.logger.info(f"训练完成! 总时间: {training_time:.2f}s")
        
        # 保存最终模型
        self.save_checkpoint('final_model.pth')
        
        # 绘制训练曲线
        if self.config.evaluation.save_plots:
            self.plot_training_history()
        
        # 清理
        if self.writer:
            self.writer.close()
            
        if self.use_wandb:
            import wandb
            wandb.finish()
    
    def _log_epoch_results(self, epoch: int, train_losses: Dict[str, float],
                          val_losses: Dict[str, float], metrics: Dict[str, float]):
        """记录epoch结果"""
        # 控制台输出
        self.logger.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_losses['total']:.4f} | "
            f"Val Loss: {val_losses['total']:.4f} | "
            f"Topo F1: {metrics.get('f1_score', 0):.3f}"
        )
        
        # Tensorboard
        if self.writer:
            for key, value in train_losses.items():
                self.writer.add_scalar(f'Train/{key}', value, epoch)
            for key, value in val_losses.items():
                self.writer.add_scalar(f'Val/{key}', value, epoch)
            for key, value in metrics.items():
                self.writer.add_scalar(f'Metrics/{key}', value, epoch)
            self.writer.add_scalar('Learning_Rate', 
                                 self.optimizer.param_groups[0]['lr'], epoch)
        
        # Weights & Biases
        if self.use_wandb:
            import wandb
            log_dict = {
                'epoch': epoch,
                **{f'train_{k}': v for k, v in train_losses.items()},
                **{f'val_{k}': v for k, v in val_losses.items()},
                **{f'metric_{k}': v for k, v in metrics.items()},
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            wandb.log(log_dict)
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': self.config.to_dict()
        }
        
        filepath = os.path.join(self.config.experiment.checkpoint_dir, filename)
        save_checkpoint(checkpoint, filepath)
        self.logger.info(f"检查点已保存: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = load_checkpoint(filepath, self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        self.logger.info(f"检查点已加载: {filepath}")
    
    def plot_training_history(self):
        """绘制训练历史"""
        if not self.training_history['train_losses']:
            return
            
        save_path = os.path.join(
            self.config.experiment.figure_dir, 
            'training_curves.png'
        )
        
        plot_training_curves(
            self.training_history,
            save_path=save_path,
            dpi=self.config.evaluation.plot_dpi
        )
        
        self.logger.info(f"训练曲线已保存: {save_path}")
    
    def _count_parameters(self) -> int:
        """计算模型参数数量"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)