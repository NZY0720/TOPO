#!/usr/bin/env python3
"""
Network topology visualization for Power Grid Topology Reconstruction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import networkx as nx
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
import torch
from typing import Dict, List, Tuple, Optional, Union, Any

# 设置matplotlib中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class NetworkTopologyVisualizer:
    """网络拓扑可视化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 12), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi
        
        # 颜色配置
        self.colors = {
            'bus_observable': '#2E86AB',     # 蓝色 - 可观测节点
            'bus_unobservable': '#A23B72',   # 紫色 - 不可观测节点
            'bus_generator': '#F18F01',      # 橙色 - 发电机节点
            'line_true': '#43AA8B',          # 绿色 - 真实线路
            'line_predicted': '#F8961E',     # 橙色 - 预测线路
            'line_correct': '#277DA1',       # 深蓝 - 正确预测
            'line_false_positive': '#F94144', # 红色 - 误报
            'line_false_negative': '#90E0EF', # 浅蓝 - 漏报
            'background': '#FEFEFE',         # 背景色
            'grid': '#E0E0E0'               # 网格线
        }
        
        # 样式配置
        self.node_sizes = {
            'default': 100,
            'generator': 150,
            'important': 120
        }
        
        self.line_widths = {
            'default': 1.5,
            'thick': 2.5,
            'thin': 1.0
        }
    
    def plot_topology_comparison(self, predicted_edges: torch.Tensor,
                               true_edges: torch.Tensor,
                               coordinates: np.ndarray,
                               node_labels: Optional[List[str]] = None,
                               node_types: Optional[List[str]] = None,
                               observed_mask: Optional[np.ndarray] = None,
                               save_path: Optional[str] = None,
                               title: str = "拓扑对比图") -> plt.Figure:
        """绘制预测与真实拓扑对比图"""
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 子图标题
        subplot_titles = [
            "真实拓扑", "预测拓扑",
            "预测正确性分析", "误差统计"
        ]
        
        # 1. 真实拓扑
        self._plot_single_topology(
            axes[0, 0], true_edges, coordinates,
            node_labels, node_types, observed_mask,
            edge_color=self.colors['line_true'],
            title=subplot_titles[0]
        )
        
        # 2. 预测拓扑
        self._plot_single_topology(
            axes[0, 1], predicted_edges, coordinates,
            node_labels, node_types, observed_mask,
            edge_color=self.colors['line_predicted'],
            title=subplot_titles[1]
        )
        
        # 3. 预测正确性分析
        self._plot_prediction_analysis(
            axes[1, 0], predicted_edges, true_edges, coordinates,
            node_labels, node_types, observed_mask,
            title=subplot_titles[2]
        )
        
        # 4. 误差统计
        self._plot_error_statistics(
            axes[1, 1], predicted_edges, true_edges,
            coordinates, title=subplot_titles[3]
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
        
        return fig
    
    def _plot_single_topology(self, ax: plt.Axes, edges: torch.Tensor,
                            coordinates: np.ndarray,
                            node_labels: Optional[List[str]] = None,
                            node_types: Optional[List[str]] = None,
                            observed_mask: Optional[np.ndarray] = None,
                            edge_color: str = None,
                            node_color: str = None,
                            title: str = "") -> None:
        """绘制单个拓扑图"""
        
        n_nodes = len(coordinates)
        
        # 绘制节点
        for i in range(n_nodes):
            x, y = coordinates[i]
            
            # 确定节点颜色和大小
            if observed_mask is not None:
                color = self.colors['bus_observable'] if observed_mask[i] else self.colors['bus_unobservable']
            else:
                color = node_color or self.colors['bus_observable']
            
            size = self.node_sizes['generator'] if (node_types and node_types[i] == 'generator') else self.node_sizes['default']
            
            ax.scatter(x, y, c=color, s=size, alpha=0.8, edgecolors='white', linewidths=1.5, zorder=3)
            
            # 节点标签
            if node_labels and i < len(node_labels):
                ax.annotate(node_labels[i], (x, y), xytext=(5, 5), 
                          textcoords='offset points', fontsize=8, 
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        # 绘制边
        if edges.size(1) > 0:
            for i in range(edges.size(1)):
                src, dst = edges[0, i].item(), edges[1, i].item()
                if src < n_nodes and dst < n_nodes:
                    x_coords = [coordinates[src, 0], coordinates[dst, 0]]
                    y_coords = [coordinates[src, 1], coordinates[dst, 1]]
                    
                    ax.plot(x_coords, y_coords, 
                           color=edge_color or self.colors['line_true'],
                           linewidth=self.line_widths['default'],
                           alpha=0.7, zorder=1)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        ax.set_aspect('equal')
        
        # 添加图例
        self._add_legend(ax, observed_mask is not None, node_types is not None)
    
    def _plot_prediction_analysis(self, ax: plt.Axes, predicted_edges: torch.Tensor,
                                true_edges: torch.Tensor, coordinates: np.ndarray,
                                node_labels: Optional[List[str]] = None,
                                node_types: Optional[List[str]] = None,
                                observed_mask: Optional[np.ndarray] = None,
                                title: str = "") -> None:
        """绘制预测正确性分析图"""
        
        n_nodes = len(coordinates)
        
        # 转换为集合便于比较
        pred_set = set()
        true_set = set()
        
        if predicted_edges.size(1) > 0:
            for i in range(predicted_edges.size(1)):
                src, dst = predicted_edges[0, i].item(), predicted_edges[1, i].item()
                pred_set.add((min(src, dst), max(src, dst)))
        
        if true_edges.size(1) > 0:
            for i in range(true_edges.size(1)):
                src, dst = true_edges[0, i].item(), true_edges[1, i].item()
                true_set.add((min(src, dst), max(src, dst)))
        
        # 分类边
        correct_edges = pred_set & true_set  # 正确预测
        false_positives = pred_set - true_set  # 误报
        false_negatives = true_set - pred_set  # 漏报
        
        # 绘制节点
        for i in range(n_nodes):
            x, y = coordinates[i]
            color = self.colors['bus_observable'] if (observed_mask is None or observed_mask[i]) else self.colors['bus_unobservable']
            size = self.node_sizes['default']
            
            ax.scatter(x, y, c=color, s=size, alpha=0.8, edgecolors='white', linewidths=1.5, zorder=3)
        
        # 绘制不同类型的边
        edge_categories = [
            (correct_edges, self.colors['line_correct'], '正确预测'),
            (false_positives, self.colors['line_false_positive'], '误报'),
            (false_negatives, self.colors['line_false_negative'], '漏报')
        ]
        
        for edges_set, color, label in edge_categories:
            for src, dst in edges_set:
                if src < n_nodes and dst < n_nodes:
                    x_coords = [coordinates[src, 0], coordinates[dst, 0]]
                    y_coords = [coordinates[src, 1], coordinates[dst, 1]]
                    
                    ax.plot(x_coords, y_coords, color=color,
                           linewidth=self.line_widths['default'],
                           alpha=0.8, label=label, zorder=2)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        ax.set_aspect('equal')
        
        # 添加统计信息
        stats_text = f"正确: {len(correct_edges)}\n误报: {len(false_positives)}\n漏报: {len(false_negatives)}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 图例（去重）
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    def _plot_error_statistics(self, ax: plt.Axes, predicted_edges: torch.Tensor,
                             true_edges: torch.Tensor, coordinates: np.ndarray,
                             title: str = "") -> None:
        """绘制误差统计图"""
        
        # 计算距离分布
        pred_distances = []
        true_distances = []
        
        if predicted_edges.size(1) > 0:
            for i in range(predicted_edges.size(1)):
                src, dst = predicted_edges[0, i].item(), predicted_edges[1, i].item()
                if src < len(coordinates) and dst < len(coordinates):
                    dist = np.linalg.norm(coordinates[src] - coordinates[dst])
                    pred_distances.append(dist)
        
        if true_edges.size(1) > 0:
            for i in range(true_edges.size(1)):
                src, dst = true_edges[0, i].item(), true_edges[1, i].item()
                if src < len(coordinates) and dst < len(coordinates):
                    dist = np.linalg.norm(coordinates[src] - coordinates[dst])
                    true_distances.append(dist)
        
        # 绘制直方图
        bins = np.linspace(0, max(max(pred_distances, default=1), max(true_distances, default=1)), 20)
        
        ax.hist(true_distances, bins=bins, alpha=0.6, label='真实边距离',
               color=self.colors['line_true'], density=True)
        ax.hist(pred_distances, bins=bins, alpha=0.6, label='预测边距离',
               color=self.colors['line_predicted'], density=True)
        
        ax.set_xlabel('边长度')
        ax.set_ylabel('密度')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _add_legend(self, ax: plt.Axes, has_observability: bool = False,
                   has_types: bool = False) -> None:
        """添加图例"""
        legend_elements = []
        
        # 节点图例
        if has_observability:
            legend_elements.extend([
                plt.scatter([], [], c=self.colors['bus_observable'], s=100, label='可观测节点'),
                plt.scatter([], [], c=self.colors['bus_unobservable'], s=100, label='不可观测节点')
            ])
        
        if has_types:
            legend_elements.append(
                plt.scatter([], [], c=self.colors['bus_generator'], s=150, label='发电机节点')
            )
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    def plot_network_with_probabilities(self, edges: torch.Tensor,
                                      edge_probs: torch.Tensor,
                                      coordinates: np.ndarray,
                                      threshold: float = 0.5,
                                      save_path: Optional[str] = None,
                                      title: str = "边预测概率图") -> plt.Figure:
        """绘制带有边预测概率的网络图"""
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=self.dpi)
        
        n_nodes = len(coordinates)
        
        # 绘制节点
        ax.scatter(coordinates[:, 0], coordinates[:, 1], 
                  c=self.colors['bus_observable'], s=self.node_sizes['default'],
                  alpha=0.8, edgecolors='white', linewidths=1.5, zorder=3)
        
        # 按概率绘制边
        if edges.size(1) > 0:
            # 创建颜色映射
            norm = Normalize(vmin=0, vmax=1)
            cmap = plt.cm.RdYlGn  # 红-黄-绿色图
            
            for i in range(edges.size(1)):
                src, dst = edges[0, i].item(), edges[1, i].item()
                if src < n_nodes and dst < n_nodes:
                    prob = edge_probs[i].item()
                    
                    # 根据概率设置边的颜色和粗细
                    color = cmap(norm(prob))
                    alpha = 0.3 + 0.7 * prob  # 概率越高越不透明
                    linewidth = self.line_widths['thin'] + (self.line_widths['thick'] - self.line_widths['thin']) * prob
                    
                    # 是否超过阈值
                    linestyle = '-' if prob > threshold else '--'
                    
                    x_coords = [coordinates[src, 0], coordinates[dst, 0]]
                    y_coords = [coordinates[src, 1], coordinates[dst, 1]]
                    
                    ax.plot(x_coords, y_coords, color=color, alpha=alpha,
                           linewidth=linewidth, linestyle=linestyle, zorder=1)
        
        # 添加颜色条
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('边存在概率', rotation=270, labelpad=20)
        
        # 添加阈值线
        ax.axhline(y=ax.get_ylim()[0], color='red', linestyle='--', alpha=0.5)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        ax.set_aspect('equal')
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], color='green', linewidth=2, label=f'概率 > {threshold}'),
            plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--', label=f'概率 ≤ {threshold}')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        return fig
    
    def plot_radial_topology(self, edges: torch.Tensor, coordinates: np.ndarray,
                           root_node: int = 0, save_path: Optional[str] = None,
                           title: str = "径向网络拓扑") -> plt.Figure:
        """绘制径向网络拓扑（树结构）"""
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=self.dpi)
        
        # 创建NetworkX图
        G = nx.Graph()
        G.add_nodes_from(range(len(coordinates)))
        
        if edges.size(1) > 0:
            edge_list = edges.t().numpy().tolist()
            G.add_edges_from(edge_list)
        
        # 如果图是连通的，计算从根节点的距离
        node_colors = []
        if nx.is_connected(G):
            distances = nx.single_source_shortest_path_length(G, root_node)
            max_dist = max(distances.values()) if distances else 0
            
            for i in range(len(coordinates)):
                dist = distances.get(i, max_dist + 1)
                # 根据距离设置颜色深度
                intensity = 1.0 - (dist / (max_dist + 1)) * 0.7
                node_colors.append(plt.cm.Blues(intensity))
        else:
            node_colors = [self.colors['bus_observable']] * len(coordinates)
        
        # 绘制节点
        for i, (x, y) in enumerate(coordinates):
            size = self.node_sizes['important'] if i == root_node else self.node_sizes['default']
            color = 'red' if i == root_node else node_colors[i]
            
            ax.scatter(x, y, c=color, s=size, alpha=0.8, 
                      edgecolors='white', linewidths=2, zorder=3)
            
            # 标记根节点
            if i == root_node:
                ax.annotate('ROOT', (x, y), xytext=(10, 10),
                          textcoords='offset points', fontsize=10, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))
        
        # 绘制边（树的边用较粗的线）
        if edges.size(1) > 0:
            for i in range(edges.size(1)):
                src, dst = edges[0, i].item(), edges[1, i].item()
                if src < len(coordinates) and dst < len(coordinates):
                    x_coords = [coordinates[src, 0], coordinates[dst, 0]]
                    y_coords = [coordinates[src, 1], coordinates[dst, 1]]
                    
                    ax.plot(x_coords, y_coords, color=self.colors['line_true'],
                           linewidth=self.line_widths['thick'], alpha=0.8, zorder=2)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        ax.set_aspect('equal')
        
        # 添加统计信息
        stats_text = f"节点数: {len(coordinates)}\n边数: {edges.size(1)}\n连通: {'是' if nx.is_connected(G) else '否'}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        return fig


def plot_topology_comparison(predictions: Dict[str, torch.Tensor],
                           targets: Dict[str, torch.Tensor],
                           coordinates: np.ndarray,
                           save_path: Optional[str] = None,
                           **kwargs) -> plt.Figure:
    """便捷函数：绘制拓扑对比图"""
    visualizer = NetworkTopologyVisualizer()
    
    return visualizer.plot_topology_comparison(
        predictions['edge_index'],
        targets['true_edge_index'],
        coordinates,
        save_path=save_path,
        **kwargs
    )


def plot_network_probabilities(edges: torch.Tensor, edge_probs: torch.Tensor,
                             coordinates: np.ndarray, save_path: Optional[str] = None,
                             **kwargs) -> plt.Figure:
    """便捷函数：绘制边概率图"""
    visualizer = NetworkTopologyVisualizer()
    
    return visualizer.plot_network_with_probabilities(
        edges, edge_probs, coordinates,
        save_path=save_path, **kwargs
    )