#!/usr/bin/env python3
"""
Analysis and error visualization for Power Grid Topology Reconstruction
深度分析和错误分析可视化模块
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 设置现代样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class AnalysisVisualizer:
    """分析可视化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 12), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi
        
        # 颜色配置
        self.colors = {
            'correct': '#43AA8B',           # 绿色 - 正确预测
            'false_positive': '#F94144',    # 红色 - 误报
            'false_negative': '#90E0EF',    # 浅蓝 - 漏报
            'true_positive': '#277DA1',     # 深蓝 - 真正例
            'true_negative': '#A23B72',     # 紫色 - 真负例
            'parameter_good': '#2E86AB',    # 蓝色 - 参数预测良好
            'parameter_bad': '#F18F01',     # 橙色 - 参数预测较差
            'physics_violation': '#FF6B6B', # 物理违反
            'background': '#FEFEFE',        # 背景色
            'grid': '#E0E0E0'              # 网格线
        }


class ErrorAnalysisVisualizer(AnalysisVisualizer):
    """错误分析可视化器"""
    
    def plot_error_analysis(self, error_analysis: Dict[str, Any],
                           coordinates: Optional[np.ndarray] = None,
                           save_path: Optional[str] = None,
                           title: str = "错误分析报告") -> plt.Figure:
        """绘制完整的错误分析报告"""
        
        fig = plt.figure(figsize=(20, 15), dpi=self.dpi)
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. 错误类型分布 (左上)
        ax1 = fig.add_subplot(gs[0, 0:2])
        self._plot_error_distribution(ax1, error_analysis)
        
        # 2. 距离误差分析 (右上)  
        ax2 = fig.add_subplot(gs[0, 2:4])
        self._plot_distance_error_analysis(ax2, error_analysis)
        
        # 3. 空间误差分布 (中间左)
        if coordinates is not None:
            ax3 = fig.add_subplot(gs[1, 0:2])
            self._plot_spatial_error_distribution(ax3, error_analysis, coordinates)
        
        # 4. 参数误差分析 (中间右)
        ax4 = fig.add_subplot(gs[1, 2:4])
        self._plot_parameter_error_analysis(ax4, error_analysis)
        
        # 5. 误差统计摘要 (下方左)
        ax5 = fig.add_subplot(gs[2, 0:2])
        self._plot_error_statistics_summary(ax5, error_analysis)
        
        # 6. 误差相关性分析 (下方右)
        ax6 = fig.add_subplot(gs[2, 2:4])
        self._plot_error_correlation_analysis(ax6, error_analysis)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        return fig
    
    def _plot_error_distribution(self, ax: plt.Axes, error_analysis: Dict[str, Any]):
        """绘制错误类型分布"""
        error_summary = error_analysis.get('error_summary', {})
        
        categories = ['误报边', '漏报边', '正确预测']
        values = [
            error_summary.get('total_false_positives', 0),
            error_summary.get('total_false_negatives', 0),
            error_summary.get('total_correct_predictions', 0)
        ]
        colors = [self.colors['false_positive'], self.colors['false_negative'], self.colors['correct']]
        
        # 饼图
        if sum(values) > 0:
            wedges, texts, autotexts = ax.pie(values, labels=categories, colors=colors,
                                            autopct='%1.1f%%', startangle=90)
            
            # 美化文本
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title('错误类型分布', fontsize=12, fontweight='bold')
    
    def _plot_distance_error_analysis(self, ax: plt.Axes, error_analysis: Dict[str, Any]):
        """绘制距离误差分析"""
        error_details = error_analysis.get('error_details', {})
        fp_edges = error_details.get('false_positives', [])
        fn_edges = error_details.get('false_negatives', [])
        
        fp_distances = [edge['distance'] for edge in fp_edges if 'distance' in edge]
        fn_distances = [edge['distance'] for edge in fn_edges if 'distance' in edge]
        
        if fp_distances or fn_distances:
            bins = np.linspace(0, max(max(fp_distances, default=1), max(fn_distances, default=1)), 20)
            
            if fp_distances:
                ax.hist(fp_distances, bins=bins, alpha=0.6, label='误报边距离',
                       color=self.colors['false_positive'], density=True)
            
            if fn_distances:
                ax.hist(fn_distances, bins=bins, alpha=0.6, label='漏报边距离',
                       color=self.colors['false_negative'], density=True)
            
            ax.set_xlabel('边长度')
            ax.set_ylabel('密度')
            ax.legend()
        else:
            ax.text(0.5, 0.5, '无距离数据', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title('距离误差分析', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_spatial_error_distribution(self, ax: plt.Axes, error_analysis: Dict[str, Any], 
                                       coordinates: np.ndarray):
        """绘制空间误差分布"""
        error_details = error_analysis.get('error_details', {})
        fp_edges = error_details.get('false_positives', [])
        fn_edges = error_details.get('false_negatives', [])
        
        # 绘制所有节点
        ax.scatter(coordinates[:, 0], coordinates[:, 1], 
                  c='lightgray', s=50, alpha=0.5, label='节点')
        
        # 绘制误报边
        for edge_info in fp_edges:
            if 'edge' in edge_info:
                src, dst = edge_info['edge']
                if src < len(coordinates) and dst < len(coordinates):
                    x_coords = [coordinates[src, 0], coordinates[dst, 0]]
                    y_coords = [coordinates[src, 1], coordinates[dst, 1]]
                    ax.plot(x_coords, y_coords, color=self.colors['false_positive'],
                           alpha=0.7, linewidth=2, label='误报边' if 'edge_info' not in locals() else "")
        
        # 绘制漏报边
        for edge_info in fn_edges:
            if 'edge' in edge_info:
                src, dst = edge_info['edge']
                if src < len(coordinates) and dst < len(coordinates):
                    x_coords = [coordinates[src, 0], coordinates[dst, 0]]
                    y_coords = [coordinates[src, 1], coordinates[dst, 1]]
                    ax.plot(x_coords, y_coords, color=self.colors['false_negative'],
                           alpha=0.7, linewidth=2, linestyle='--', 
                           label='漏报边' if 'fn_plotted' not in locals() else "")
                    locals()['fn_plotted'] = True
        
        ax.set_title('空间误差分布', fontsize=12, fontweight='bold')
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _plot_parameter_error_analysis(self, ax: plt.Axes, error_analysis: Dict[str, Any]):
        """绘制参数误差分析"""
        param_errors = error_analysis.get('error_details', {}).get('parameter_errors', [])
        
        if param_errors:
            r_errors = [e['r_error'] for e in param_errors]
            x_errors = [e['x_error'] for e in param_errors]
            
            # 散点图：R误差 vs X误差
            ax.scatter(r_errors, x_errors, alpha=0.6, 
                      c=self.colors['parameter_bad'], s=50)
            
            # 添加对角线
            max_error = max(max(r_errors), max(x_errors))
            ax.plot([0, max_error], [0, max_error], 'r--', alpha=0.5, label='R=X线')
            
            ax.set_xlabel('电阻(R)预测误差')
            ax.set_ylabel('电抗(X)预测误差')
            ax.legend()
            
            # 添加统计信息
            r_mean, x_mean = np.mean(r_errors), np.mean(x_errors)
            ax.text(0.02, 0.98, f'R误差均值: {r_mean:.4f}\nX误差均值: {x_mean:.4f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, '无参数误差数据', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title('参数预测误差分析', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_error_statistics_summary(self, ax: plt.Axes, error_analysis: Dict[str, Any]):
        """绘制误差统计摘要"""
        error_summary = error_analysis.get('error_summary', {})
        error_patterns = error_analysis.get('error_details', {}).get('error_patterns', {})
        
        # 创建统计摘要文本
        summary_text = []
        summary_text.append("=== 误差统计摘要 ===\n")
        
        # 基本统计
        summary_text.append(f"误报边数: {error_summary.get('total_false_positives', 0)}")
        summary_text.append(f"漏报边数: {error_summary.get('total_false_negatives', 0)}")
        summary_text.append(f"平均参数误差: {error_summary.get('avg_parameter_error', 0):.4f}\n")
        
        # 距离统计
        if 'fp_distance_stats' in error_patterns:
            fp_stats = error_patterns['fp_distance_stats']
            summary_text.append("误报边距离统计:")
            summary_text.append(f"  均值: {fp_stats.get('mean', 0):.2f}")
            summary_text.append(f"  标准差: {fp_stats.get('std', 0):.2f}")
            summary_text.append(f"  中位数: {fp_stats.get('median', 0):.2f}\n")
        
        if 'fn_distance_stats' in error_patterns:
            fn_stats = error_patterns['fn_distance_stats']
            summary_text.append("漏报边距离统计:")
            summary_text.append(f"  均值: {fn_stats.get('mean', 0):.2f}")
            summary_text.append(f"  标准差: {fn_stats.get('std', 0):.2f}")
            summary_text.append(f"  中位数: {fn_stats.get('median', 0):.2f}\n")
        
        # 参数误差统计
        if 'parameter_error_stats' in error_patterns:
            param_stats = error_patterns['parameter_error_stats']
            summary_text.append("参数误差统计:")
            summary_text.append(f"  R误差均值: {param_stats.get('r_error_mean', 0):.4f}")
            summary_text.append(f"  X误差均值: {param_stats.get('x_error_mean', 0):.4f}")
            summary_text.append(f"  R误差标准差: {param_stats.get('r_error_std', 0):.4f}")
            summary_text.append(f"  X误差标准差: {param_stats.get('x_error_std', 0):.4f}")
        
        # 显示文本
        ax.text(0.05, 0.95, '\n'.join(summary_text), transform=ax.transAxes,
               verticalalignment='top', fontsize=10, fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('误差统计摘要', fontsize=12, fontweight='bold')
    
    def _plot_error_correlation_analysis(self, ax: plt.Axes, error_analysis: Dict[str, Any]):
        """绘制误差相关性分析"""
        param_errors = error_analysis.get('error_details', {}).get('parameter_errors', [])
        
        if len(param_errors) > 5:  # 至少需要一些数据点
            # 创建误差数据框
            error_df = pd.DataFrame(param_errors)
            
            # 选择数值列进行相关性分析
            numeric_cols = ['r_error', 'x_error', 'error', 'r_pred', 'x_pred', 'r_true', 'x_true']
            available_cols = [col for col in numeric_cols if col in error_df.columns]
            
            if len(available_cols) >= 2:
                corr_matrix = error_df[available_cols].corr()
                
                # 绘制相关性热力图
                sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                           square=True, fmt='.2f', ax=ax)
                ax.set_title('误差相关性分析', fontsize=12, fontweight='bold')
            else:
                ax.text(0.5, 0.5, '数据不足\n无法进行相关性分析', 
                       ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, '数据点过少\n无法进行相关性分析', 
                   ha='center', va='center', transform=ax.transAxes)


class PerformanceAnalysisVisualizer(AnalysisVisualizer):
    """性能分析可视化器"""
    
    def plot_threshold_analysis(self, threshold_results: Dict[str, Any],
                               save_path: Optional[str] = None,
                               title: str = "阈值敏感性分析") -> plt.Figure:
        """绘制阈值敏感性分析"""
        
        results = threshold_results.get('threshold_results', [])
        if not results:
            raise ValueError("阈值分析结果为空")
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        
        # 提取数据
        thresholds = [r['threshold'] for r in results]
        f1_scores = [r['f1_score'] for r in results]
        precisions = [r['precision'] for r in results]
        recalls = [r['recall'] for r in results]
        num_edges = [r['num_predicted_edges'] for r in results]
        
        # 1. F1分数曲线
        axes[0, 0].plot(thresholds, f1_scores, 'o-', color=self.colors['correct'], linewidth=2)
        axes[0, 0].axvline(x=threshold_results.get('best_f1_threshold', 0.5), 
                          color='red', linestyle='--', alpha=0.7, label='最佳阈值')
        axes[0, 0].set_title('F1分数 vs 阈值')
        axes[0, 0].set_xlabel('阈值')
        axes[0, 0].set_ylabel('F1分数')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 精确率-召回率曲线
        axes[0, 1].plot(thresholds, precisions, 'o-', color=self.colors['true_positive'], 
                       linewidth=2, label='精确率')
        axes[0, 1].plot(thresholds, recalls, 's-', color=self.colors['false_negative'], 
                       linewidth=2, label='召回率')
        axes[0, 1].set_title('精确率-召回率 vs 阈值')
        axes[0, 1].set_xlabel('阈值')
        axes[0, 1].set_ylabel('分数')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 预测边数量
        axes[1, 0].plot(thresholds, num_edges, 'o-', color=self.colors['parameter_good'], linewidth=2)
        if 'optimal_metrics' in threshold_results:
            true_edges = threshold_results['optimal_metrics']['f1'].get('num_true_edges', 0)
            axes[1, 0].axhline(y=true_edges, color='red', linestyle='--', 
                              alpha=0.7, label=f'真实边数: {true_edges}')
        axes[1, 0].set_title('预测边数 vs 阈值')
        axes[1, 0].set_xlabel('阈值')
        axes[1, 0].set_ylabel('边数')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 综合评分
        composite_scores = [r.get('composite_score', 0) for r in results]
        axes[1, 1].plot(thresholds, composite_scores, 'o-', color=self.colors['physics_violation'], linewidth=2)
        axes[1, 1].axvline(x=threshold_results.get('best_composite_threshold', 0.5), 
                          color='red', linestyle='--', alpha=0.7, label='最佳综合阈值')
        axes[1, 1].set_title('综合评分 vs 阈值')
        axes[1, 1].set_xlabel('阈值')
        axes[1, 1].set_ylabel('综合评分')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        return fig
    
    def plot_confusion_matrix_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     class_names: Optional[List[str]] = None,
                                     save_path: Optional[str] = None) -> plt.Figure:
        """绘制混淆矩阵分析"""
        
        if class_names is None:
            class_names = ['无边', '有边']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')
        
        # 1. 原始混淆矩阵
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=axes[0])
        axes[0].set_title('混淆矩阵 (计数)')
        axes[0].set_ylabel('真实标签')
        axes[0].set_xlabel('预测标签')
        
        # 2. 归一化混淆矩阵
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=axes[1])
        axes[1].set_title('混淆矩阵 (归一化)')
        axes[1].set_ylabel('真实标签')
        axes[1].set_xlabel('预测标签')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        return fig
    
    def plot_feature_importance_analysis(self, feature_importance: Dict[str, float],
                                       save_path: Optional[str] = None) -> plt.Figure:
        """绘制特征重要性分析"""
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=self.dpi)
        
        # 排序特征重要性
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features, importance = zip(*sorted_features)
        
        # 绘制水平条形图
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, importance, color=self.colors['parameter_good'], alpha=0.7)
        
        # 美化图表
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # 最重要的特征在顶部
        ax.set_xlabel('重要性分数')
        ax.set_title('特征重要性分析', fontsize=14, fontweight='bold')
        
        # 添加数值标签
        for i, (bar, imp) in enumerate(zip(bars, importance)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{imp:.3f}', va='center', ha='left')
        
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        return fig


class StatisticalAnalysisVisualizer(AnalysisVisualizer):
    """统计分析可视化器"""
    
    def plot_prediction_distribution_analysis(self, predictions: Dict[str, torch.Tensor],
                                             targets: Dict[str, torch.Tensor],
                                             save_path: Optional[str] = None) -> plt.Figure:
        """绘制预测分布分析"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=self.dpi)
        
        # 提取数据
        edge_probs = predictions.get('all_probs', torch.empty(0)).numpy()
        edge_params = predictions.get('all_params', torch.empty(0, 2)).numpy()
        true_params = targets.get('true_edge_params', torch.empty(0, 2)).numpy()
        
        # 1. 边概率分布
        if len(edge_probs) > 0:
            axes[0, 0].hist(edge_probs, bins=30, alpha=0.7, color=self.colors['correct'], 
                           density=True, edgecolor='black')
            axes[0, 0].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='默认阈值')
            axes[0, 0].set_title('边存在概率分布')
            axes[0, 0].set_xlabel('概率')
            axes[0, 0].set_ylabel('密度')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 预测参数分布 - R
        if edge_params.shape[0] > 0:
            axes[0, 1].hist(edge_params[:, 0], bins=30, alpha=0.7, 
                           color=self.colors['parameter_good'], density=True, edgecolor='black')
            axes[0, 1].set_title('预测电阻(R)分布')
            axes[0, 1].set_xlabel('电阻值')
            axes[0, 1].set_ylabel('密度')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 预测参数分布 - X
        if edge_params.shape[0] > 0:
            axes[0, 2].hist(edge_params[:, 1], bins=30, alpha=0.7, 
                           color=self.colors['parameter_bad'], density=True, edgecolor='black')
            axes[0, 2].set_title('预测电抗(X)分布')
            axes[0, 2].set_xlabel('电抗值')
            axes[0, 2].set_ylabel('密度')
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 真实 vs 预测参数对比 - R
        if true_params.shape[0] > 0 and edge_params.shape[0] > 0:
            min_len = min(true_params.shape[0], edge_params.shape[0])
            axes[1, 0].scatter(true_params[:min_len, 0], edge_params[:min_len, 0], 
                              alpha=0.6, color=self.colors['parameter_good'])
            
            # 添加完美预测线
            max_val = max(np.max(true_params[:min_len, 0]), np.max(edge_params[:min_len, 0]))
            axes[1, 0].plot([0, max_val], [0, max_val], 'r--', alpha=0.7, label='完美预测')
            
            axes[1, 0].set_title('电阻(R): 真实 vs 预测')
            axes[1, 0].set_xlabel('真实值')
            axes[1, 0].set_ylabel('预测值')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 真实 vs 预测参数对比 - X
        if true_params.shape[0] > 0 and edge_params.shape[0] > 0:
            min_len = min(true_params.shape[0], edge_params.shape[0])
            axes[1, 1].scatter(true_params[:min_len, 1], edge_params[:min_len, 1], 
                              alpha=0.6, color=self.colors['parameter_bad'])
            
            # 添加完美预测线
            max_val = max(np.max(true_params[:min_len, 1]), np.max(edge_params[:min_len, 1]))
            axes[1, 1].plot([0, max_val], [0, max_val], 'r--', alpha=0.7, label='完美预测')
            
            axes[1, 1].set_title('电抗(X): 真实 vs 预测')
            axes[1, 1].set_xlabel('真实值')
            axes[1, 1].set_ylabel('预测值')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 参数预测误差分布
        if true_params.shape[0] > 0 and edge_params.shape[0] > 0:
            min_len = min(true_params.shape[0], edge_params.shape[0])
            r_errors = np.abs(true_params[:min_len, 0] - edge_params[:min_len, 0])
            x_errors = np.abs(true_params[:min_len, 1] - edge_params[:min_len, 1])
            
            axes[1, 2].hist(r_errors, bins=20, alpha=0.6, label='R误差', 
                           color=self.colors['parameter_good'], density=True)
            axes[1, 2].hist(x_errors, bins=20, alpha=0.6, label='X误差', 
                           color=self.colors['parameter_bad'], density=True)
            
            axes[1, 2].set_title('参数预测误差分布')
            axes[1, 2].set_xlabel('绝对误差')
            axes[1, 2].set_ylabel('密度')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('预测分布统计分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        return fig
    
    def plot_model_confidence_analysis(self, predictions: Dict[str, torch.Tensor],
                                     targets: Dict[str, torch.Tensor],
                                     save_path: Optional[str] = None) -> plt.Figure:
        """绘制模型置信度分析"""
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        
        edge_probs = predictions.get('all_probs', torch.empty(0)).numpy()
        true_edge_index = targets.get('true_edge_index', torch.empty(2, 0))
        candidate_edge_index = predictions.get('edge_index', torch.empty(2, 0))
        
        if len(edge_probs) > 0 and true_edge_index.size(1) > 0:
            # 创建真实标签
            true_labels = self._create_edge_labels(candidate_edge_index, true_edge_index)
            
            # 1. 置信度分布（按真实标签分组）
            true_edge_probs = edge_probs[true_labels == 1]
            false_edge_probs = edge_probs[true_labels == 0]
            
            axes[0, 0].hist(false_edge_probs, bins=30, alpha=0.6, label='真实无边', 
                           color=self.colors['false_positive'], density=True)
            axes[0, 0].hist(true_edge_probs, bins=30, alpha=0.6, label='真实有边', 
                           color=self.colors['true_positive'], density=True)
            axes[0, 0].set_title('置信度分布（按真实标签）')
            axes[0, 0].set_xlabel('预测概率')
            axes[0, 0].set_ylabel('密度')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 校准曲线
            self._plot_calibration_curve(axes[0, 1], true_labels, edge_probs)
            
            # 3. 置信度 vs 准确性
            self._plot_confidence_accuracy(axes[1, 0], true_labels, edge_probs)
            
            # 4. ROC曲线
            self._plot_roc_curve(axes[1, 1], true_labels, edge_probs)
        
        plt.suptitle('模型置信度分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        return fig
    
    def _create_edge_labels(self, candidate_edges: torch.Tensor, true_edges: torch.Tensor) -> np.ndarray:
        """创建边标签"""
        labels = np.zeros(candidate_edges.size(1))
        
        if true_edges.size(1) == 0:
            return labels
        
        for i in range(candidate_edges.size(1)):
            src, dst = candidate_edges[0, i].item(), candidate_edges[1, i].item()
            
            # 检查是否为真实边（考虑无向性）
            is_true_edge = (
                ((true_edges[0] == src) & (true_edges[1] == dst)) |
                ((true_edges[0] == dst) & (true_edges[1] == src))
            ).any()
            
            if is_true_edge:
                labels[i] = 1.0
        
        return labels
    
    def _plot_calibration_curve(self, ax: plt.Axes, y_true: np.ndarray, y_prob: np.ndarray):
        """绘制校准曲线"""
        from sklearn.calibration import calibration_curve
        
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=10
            )
            
            ax.plot(mean_predicted_value, fraction_of_positives, "s-", 
                   color=self.colors['correct'], label='模型校准')
            ax.plot([0, 1], [0, 1], "k:", label='完美校准')
            
            ax.set_title('校准曲线')
            ax.set_xlabel('平均预测概率')
            ax.set_ylabel('真实正例比例')
            ax.legend()
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f'校准曲线计算失败:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_confidence_accuracy(self, ax: plt.Axes, y_true: np.ndarray, y_prob: np.ndarray):
        """绘制置信度vs准确性"""
        # 将概率分组
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        accuracies = []
        counts = []
        
        for i in range(n_bins):
            bin_mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
            if i == n_bins - 1:  # 最后一个bin包含边界
                bin_mask = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
            
            if np.sum(bin_mask) > 0:
                bin_center = (bin_boundaries[i] + bin_boundaries[i + 1]) / 2
                bin_accuracy = np.mean(y_true[bin_mask] == (y_prob[bin_mask] > 0.5))
                bin_count = np.sum(bin_mask)
                
                bin_centers.append(bin_center)
                accuracies.append(bin_accuracy)
                counts.append(bin_count)
        
        if bin_centers:
            # 绘制准确性
            ax.bar(bin_centers, accuracies, width=0.08, alpha=0.7, 
                  color=self.colors['correct'], label='准确性')
            
            # 添加样本数量标签
            ax2 = ax.twinx()
            ax2.plot(bin_centers, counts, 'ro-', alpha=0.7, label='样本数')
            ax2.set_ylabel('样本数')
            ax2.legend(loc='upper right')
            
            ax.set_title('置信度 vs 准确性')
            ax.set_xlabel('预测置信度')
            ax.set_ylabel('准确性')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
    
    def _plot_roc_curve(self, ax: plt.Axes, y_true: np.ndarray, y_prob: np.ndarray):
        """绘制ROC曲线"""
        from sklearn.metrics import roc_curve, auc
        
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=self.colors['correct'], linewidth=2,
                   label=f'ROC曲线 (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', label='随机分类器')
            
            ax.set_title('ROC曲线')
            ax.set_xlabel('假正例率')
            ax.set_ylabel('真正例率')
            ax.legend()
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f'ROC曲线计算失败:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)


# 便捷函数
def plot_error_analysis(error_analysis: Dict[str, Any], 
                       coordinates: Optional[np.ndarray] = None,
                       save_path: Optional[str] = None, **kwargs) -> plt.Figure:
    """便捷函数：绘制错误分析"""
    visualizer = ErrorAnalysisVisualizer()
    return visualizer.plot_error_analysis(error_analysis, coordinates, save_path, **kwargs)


def plot_threshold_analysis(threshold_results: Dict[str, Any],
                          save_path: Optional[str] = None, **kwargs) -> plt.Figure:
    """便捷函数：绘制阈值分析"""
    visualizer = PerformanceAnalysisVisualizer()
    return visualizer.plot_threshold_analysis(threshold_results, save_path, **kwargs)


def plot_prediction_distributions(predictions: Dict[str, torch.Tensor],
                                 targets: Dict[str, torch.Tensor],
                                 save_path: Optional[str] = None) -> plt.Figure:
    """便捷函数：绘制预测分布分析"""
    visualizer = StatisticalAnalysisVisualizer()
    return visualizer.plot_prediction_distribution_analysis(predictions, targets, save_path)


def plot_model_confidence(predictions: Dict[str, torch.Tensor],
                         targets: Dict[str, torch.Tensor],
                         save_path: Optional[str] = None) -> plt.Figure:
    """便捷函数：绘制模型置信度分析"""
    visualizer = StatisticalAnalysisVisualizer()
    return visualizer.plot_model_confidence_analysis(predictions, targets, save_path)