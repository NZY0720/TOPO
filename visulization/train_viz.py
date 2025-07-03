#!/usr/bin/env python3
"""
Training process visualization for Power Grid Topology Reconstruction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
warnings.filterwarnings('ignore')

# 设置样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class TrainingVisualizer:
    """训练过程可视化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi
        
        # 颜色配置
        self.colors = {
            'train': '#2E86AB',
            'val': '#A23B72',
            'total': '#F18F01',
            'topology': '#43AA8B',
            'parameter': '#F8961E',
            'physics': '#277DA1',
            'sparsity': '#90E0EF',
            'geographic': '#F94144'
        }
    
    def plot_training_curves(self, training_history: Dict[str, List],
                           save_path: Optional[str] = None,
                           show_validation: bool = True) -> plt.Figure:
        """绘制训练曲线"""
        
        train_losses = training_history.get('train_losses', [])
        val_losses = training_history.get('val_losses', [])
        
        if not train_losses:
            raise ValueError("训练历史为空")
        
        # 确定子图数量
        loss_keys = list(train_losses[0].keys()) if train_losses else []
        n_plots = len(loss_keys)
        
        # 计算子图布局
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=self.figsize, dpi=self.dpi)
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        # 绘制每种损失
        for i, loss_key in enumerate(loss_keys):
            ax = axes[i] if i < len(axes) else None
            if ax is None:
                continue
                
            # 提取训练损失
            train_values = [epoch_loss[loss_key] for epoch_loss in train_losses]
            epochs = range(1, len(train_values) + 1)
            
            ax.plot(epochs, train_values, 
                   color=self.colors.get(loss_key, self.colors['train']),
                   label=f'训练 {loss_key}', linewidth=2, alpha=0.8)
            
            # 绘制验证损失（如果有）
            if show_validation and val_losses:
                # 验证损失可能不是每个epoch都有
                val_epochs = []
                val_values = []
                
                for j, val_epoch_loss in enumerate(val_losses):
                    if loss_key in val_epoch_loss:
                        val_epochs.append((j + 1) * (len(train_losses) // len(val_losses)))
                        val_values.append(val_epoch_loss[loss_key])
                
                if val_values:
                    ax.plot(val_epochs, val_values,
                           color=self.colors.get(loss_key, self.colors['val']),
                           label=f'验证 {loss_key}', linewidth=2, alpha=0.8,
                           linestyle='--', marker='o', markersize=4)
            
            ax.set_title(f'{loss_key.capitalize()} Loss', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 使用对数刻度（如果值变化很大）
            if train_values and max(train_values) / min(train_values) > 100:
                ax.set_yscale('log')
        
        # 隐藏多余的子图
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('训练过程监控', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        return fig
    
    def plot_metrics_evolution(self, training_history: Dict[str, List],
                             save_path: Optional[str] = None) -> plt.Figure:
        """绘制评估指标演化"""
        
        metrics_history = training_history.get('metrics', [])
        if not metrics_history:
            raise ValueError("没有评估指标历史")
        
        # 提取指标
        all_metrics = {}
        for epoch_metrics in metrics_history:
            for category, metrics_dict in epoch_metrics.items():
                if isinstance(metrics_dict, dict):
                    for metric_name, value in metrics_dict.items():
                        key = f"{category}_{metric_name}"
                        if key not in all_metrics:
                            all_metrics[key] = []
                        all_metrics[key].append(value)
                else:
                    if category not in all_metrics:
                        all_metrics[category] = []
                    all_metrics[category].append(metrics_dict)
        
        # 选择关键指标
        key_metrics = [
            'topology_f1_score', 'topology_precision', 'topology_recall',
            'parameters_param_mae_avg', 'composite_score'
        ]
        
        available_metrics = [m for m in key_metrics if m in all_metrics]
        
        if not available_metrics:
            # 如果没有关键指标，选择前几个可用指标
            available_metrics = list(all_metrics.keys())[:6]
        
        n_metrics = len(available_metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=self.figsize, dpi=self.dpi)
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, metric_name in enumerate(available_metrics):
            ax = axes[i] if i < len(axes) else None
            if ax is None:
                continue
                
            values = all_metrics[metric_name]
            epochs = range(1, len(values) + 1)
            
            ax.plot(epochs, values, linewidth=2, marker='o', markersize=4,
                   color=self.colors.get(metric_name.split('_')[0], '#2E86AB'))
            
            ax.set_title(metric_name.replace('_', ' ').title(), fontsize=10, fontweight='bold')
            ax.set_xlabel('Validation Step')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            
            # 添加最佳值标记
            if values:
                best_idx = np.argmax(values) if 'f1' in metric_name or 'precision' in metric_name or 'recall' in metric_name or 'composite' in metric_name else np.argmin(values)
                ax.axhline(y=values[best_idx], color='red', linestyle='--', alpha=0.5)
                ax.text(0.02, 0.98, f'最佳: {values[best_idx]:.3f}', transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 隐藏多余的子图
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('评估指标演化', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        return fig
    
    def plot_loss_composition(self, training_history: Dict[str, List],
                            save_path: Optional[str] = None) -> plt.Figure:
        """绘制损失组成分析"""
        
        train_losses = training_history.get('train_losses', [])
        if not train_losses:
            raise ValueError("训练历史为空")
        
        # 提取各个损失组件
        loss_components = {}
        for epoch_loss in train_losses:
            for loss_name, value in epoch_loss.items():
                if loss_name != 'total':
                    if loss_name not in loss_components:
                        loss_components[loss_name] = []
                    loss_components[loss_name].append(value)
        
        if not loss_components:
            raise ValueError("没有找到损失组件")
        
        epochs = range(1, len(train_losses) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)
        
        # 1. 堆叠面积图
        loss_matrix = np.array([loss_components[name] for name in loss_components.keys()])
        ax1.stackplot(epochs, *loss_matrix, 
                     labels=list(loss_components.keys()),
                     colors=[self.colors.get(name, f'C{i}') for i, name in enumerate(loss_components.keys())],
                     alpha=0.7)
        
        ax1.set_title('损失组成演化（堆叠图）', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss Value')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. 相对比例图
        # 计算每个epoch各损失的相对比例
        total_losses = [sum(epoch_loss[name] for name in loss_components.keys() if name in epoch_loss) 
                       for epoch_loss in train_losses]
        
        relative_losses = {}
        for name in loss_components.keys():
            relative_losses[name] = [loss_components[name][i] / (total_losses[i] + 1e-8) 
                                   for i in range(len(total_losses))]
        
        for name, values in relative_losses.items():
            ax2.plot(epochs, values, label=name, linewidth=2,
                    color=self.colors.get(name, f'C{len(relative_losses)}'))
        
        ax2.set_title('损失相对比例演化', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Relative Proportion')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        return fig
    
    def plot_learning_rate_schedule(self, learning_rates: List[float],
                                   save_path: Optional[str] = None) -> plt.Figure:
        """绘制学习率调度"""
        
        if not learning_rates:
            raise ValueError("学习率历史为空")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=self.dpi)
        
        epochs = range(1, len(learning_rates) + 1)
        ax.plot(epochs, learning_rates, linewidth=2, color=self.colors['train'], marker='o', markersize=3)
        
        ax.set_title('学习率调度', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # 标记重要的学习率变化点
        lr_changes = []
        for i in range(1, len(learning_rates)):
            if abs(learning_rates[i] - learning_rates[i-1]) / learning_rates[i-1] > 0.1:
                lr_changes.append(i)
        
        for change_point in lr_changes:
            ax.axvline(x=change_point + 1, color='red', linestyle='--', alpha=0.5)
            ax.text(change_point + 1, learning_rates[change_point], 
                   f'{learning_rates[change_point]:.2e}',
                   rotation=90, verticalalignment='bottom')
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        return fig
    
    def plot_comprehensive_training_summary(self, training_history: Dict[str, List],
                                          learning_rates: Optional[List[float]] = None,
                                          save_path: Optional[str] = None) -> plt.Figure:
        """绘制综合训练摘要"""
        
        fig = plt.figure(figsize=(20, 15), dpi=self.dpi)
        
        # 创建复杂的子图布局
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. 主要损失曲线 (上半部分)
        ax_main_loss = fig.add_subplot(gs[0:2, 0:3])
        self._plot_main_losses(ax_main_loss, training_history)
        
        # 2. 学习率 (右上角)
        ax_lr = fig.add_subplot(gs[0, 3])
        if learning_rates:
            self._plot_lr_subplot(ax_lr, learning_rates)
        
        # 3. 关键指标 (右上角下方)
        ax_metrics = fig.add_subplot(gs[1, 3])
        self._plot_key_metrics_subplot(ax_metrics, training_history)
        
        # 4. 损失组件 (下半部分左侧)
        ax_components = fig.add_subplot(gs[2:4, 0:2])
        self._plot_loss_components_subplot(ax_components, training_history)
        
        # 5. 训练统计 (下半部分右侧)
        ax_stats = fig.add_subplot(gs[2:4, 2:4])
        self._plot_training_statistics(ax_stats, training_history)
        
        plt.suptitle('训练过程综合分析', fontsize=18, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        return fig
    
    def _plot_main_losses(self, ax: plt.Axes, training_history: Dict[str, List]):
        """绘制主要损失曲线"""
        train_losses = training_history.get('train_losses', [])
        val_losses = training_history.get('val_losses', [])
        
        if train_losses:
            total_train = [epoch_loss.get('total', 0) for epoch_loss in train_losses]
            epochs = range(1, len(total_train) + 1)
            ax.plot(epochs, total_train, color=self.colors['train'], 
                   linewidth=2, label='训练总损失')
            
            if val_losses:
                total_val = [epoch_loss.get('total', 0) for epoch_loss in val_losses]
                val_epochs = np.linspace(1, len(total_train), len(total_val))
                ax.plot(val_epochs, total_val, color=self.colors['val'],
                       linewidth=2, linestyle='--', marker='o', 
                       markersize=4, label='验证总损失')
        
        ax.set_title('总损失演化', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_lr_subplot(self, ax: plt.Axes, learning_rates: List[float]):
        """绘制学习率子图"""
        epochs = range(1, len(learning_rates) + 1)
        ax.plot(epochs, learning_rates, color=self.colors['parameter'], linewidth=2)
        ax.set_title('学习率', fontweight='bold', fontsize=10)
        ax.set_xlabel('Epoch')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    def _plot_key_metrics_subplot(self, ax: plt.Axes, training_history: Dict[str, List]):
        """绘制关键指标子图"""
        metrics_history = training_history.get('metrics', [])
        if not metrics_history:
            ax.text(0.5, 0.5, '无指标数据', ha='center', va='center', transform=ax.transAxes)
            return
        
        # 提取F1分数
        f1_scores = []
        for epoch_metrics in metrics_history:
            topo_metrics = epoch_metrics.get('topology', {})
            f1_scores.append(topo_metrics.get('f1_score', 0))
        
        if f1_scores:
            steps = range(1, len(f1_scores) + 1)
            ax.plot(steps, f1_scores, color=self.colors['topology'], 
                   linewidth=2, marker='o', markersize=3)
            ax.set_title('F1分数', fontweight='bold', fontsize=10)
            ax.set_xlabel('验证步骤')
            ax.grid(True, alpha=0.3)
    
    def _plot_loss_components_subplot(self, ax: plt.Axes, training_history: Dict[str, List]):
        """绘制损失组件子图"""
        train_losses = training_history.get('train_losses', [])
        if not train_losses:
            return
        
        # 选择主要损失组件
        main_components = ['topology', 'parameter', 'physics']
        epochs = range(1, len(train_losses) + 1)
        
        for component in main_components:
            if component in train_losses[0]:
                values = [epoch_loss.get(component, 0) for epoch_loss in train_losses]
                ax.plot(epochs, values, label=component.title(), 
                       color=self.colors.get(component, 'gray'), linewidth=2)
        
        ax.set_title('主要损失组件', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_training_statistics(self, ax: plt.Axes, training_history: Dict[str, List]):
        """绘制训练统计信息"""
        train_losses = training_history.get('train_losses', [])
        metrics_history = training_history.get('metrics', [])
        
        stats_text = []
        
        if train_losses:
            total_epochs = len(train_losses)
            final_loss = train_losses[-1].get('total', 0)
            initial_loss = train_losses[0].get('total', 0)
            improvement = (initial_loss - final_loss) / initial_loss * 100 if initial_loss > 0 else 0
            
            stats_text.extend([
                f"总训练轮数: {total_epochs}",
                f"初始损失: {initial_loss:.4f}",
                f"最终损失: {final_loss:.4f}",
                f"损失改善: {improvement:.1f}%",
                ""
            ])
        
        if metrics_history:
            # 最佳指标
            best_f1 = 0
            best_precision = 0
            best_recall = 0
            
            for epoch_metrics in metrics_history:
                topo_metrics = epoch_metrics.get('topology', {})
                best_f1 = max(best_f1, topo_metrics.get('f1_score', 0))
                best_precision = max(best_precision, topo_metrics.get('precision', 0))
                best_recall = max(best_recall, topo_metrics.get('recall', 0))
            
            stats_text.extend([
                "最佳指标:",
                f"  F1分数: {best_f1:.3f}",
                f"  精确率: {best_precision:.3f}",
                f"  召回率: {best_recall:.3f}"
            ])
        
        # 显示统计信息
        ax.text(0.05, 0.95, '\n'.join(stats_text), transform=ax.transAxes,
               verticalalignment='top', fontsize=11, 
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('训练统计', fontweight='bold')


def plot_training_curves(training_history: Dict[str, List], 
                        save_path: Optional[str] = None,
                        **kwargs) -> plt.Figure:
    """便捷函数：绘制训练曲线"""
    visualizer = TrainingVisualizer()
    return visualizer.plot_training_curves(training_history, save_path, **kwargs)


def plot_comprehensive_summary(training_history: Dict[str, List],
                             learning_rates: Optional[List[float]] = None,
                             save_path: Optional[str] = None) -> plt.Figure:
    """便捷函数：绘制综合训练摘要"""
    visualizer = TrainingVisualizer()
    return visualizer.plot_comprehensive_training_summary(
        training_history, learning_rates, save_path
    )