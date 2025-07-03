#!/usr/bin/env python3
"""
Model evaluation and analysis for Power Grid Topology Reconstruction
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import time

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..config.base_config import Config
from ..data.data_loader import PowerGridDataLoader
from ..models.graph_transformer import PhysicsGraphTransformer
from ..evaluation.metrics import ComprehensiveMetrics
from ..utils.graph_utils import ensure_radial_topology
from ..utils.io_utils import load_checkpoint, save_json
from ..utils.logging_utils import setup_logger
from ..visualization.network_viz import plot_topology_comparison
from ..visualization.analysis_viz import plot_error_analysis


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.training.device)
        
        # 初始化评估器
        self.metrics_calculator = ComprehensiveMetrics()
        
        # 设置日志
        self.logger = setup_logger(
            'evaluator',
            os.path.join(config.experiment.log_dir, 'evaluation.log')
        )
        
        # 结果存储
        self.evaluation_results = {}
        
    def load_model(self, model_path: str, model: Optional[PhysicsGraphTransformer] = None) -> PhysicsGraphTransformer:
        """加载训练好的模型"""
        if model is None:
            # 从检查点加载完整模型配置
            checkpoint = load_checkpoint(model_path, self.device)
            model_config = checkpoint.get('config', {}).get('model', {})
            
            from ..models.graph_transformer import create_model
            from ..config.base_config import ModelConfig
            
            model_cfg = ModelConfig(**model_config)
            model = create_model(model_cfg, model_cfg.d_input)
        
        # 加载模型权重
        checkpoint = load_checkpoint(model_path, self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        self.logger.info(f"模型已加载: {model_path}")
        return model
    
    def evaluate_single_prediction(self, model: PhysicsGraphTransformer,
                                 data_loader: PowerGridDataLoader,
                                 time_step: int = 0,
                                 threshold: float = 0.5) -> Dict[str, Any]:
        """评估单个时间步的预测"""
        model.eval()
        
        with torch.no_grad():
            # 准备数据
            data = data_loader.create_graph_data(time_step).to(self.device)
            
            # 模型预测
            start_time = time.time()
            predictions = model.predict_topology(data, threshold=threshold)
            inference_time = time.time() - start_time
            
            # 确保径向拓扑（如果需要）
            if self.config.evaluation.ensure_radial:
                radial_edges = ensure_radial_topology(
                    predictions['edge_index'].cpu(),
                    predictions['edge_probs'].cpu(),
                    data.num_nodes
                )
                predictions['radial_edges'] = radial_edges
            
            # 准备目标数据
            targets = {
                'true_edge_index': data.true_edge_index.cpu(),
                'true_edge_params': data.true_edge_params.cpu()
            }
            
            # 转换预测数据为CPU
            predictions_cpu = {
                'edge_index': predictions['edge_index'].cpu(),
                'edge_params': predictions['edge_params'].cpu(),
                'edge_probs': predictions['edge_probs'].cpu(),
                'all_probs': predictions['all_probs'].cpu(),
                'all_params': predictions['all_params'].cpu()
            }
            
            if 'radial_edges' in predictions:
                predictions_cpu['radial_edges'] = predictions['radial_edges']
            
            # 计算评估指标
            metrics = self.metrics_calculator.compute_all(
                predictions_cpu, targets, data.cpu()
            )
            
            # 添加推理时间
            metrics['inference_time'] = inference_time
            metrics['num_candidate_edges'] = data.edge_index.size(1)
            metrics['threshold'] = threshold
            
            return {
                'predictions': predictions_cpu,
                'targets': targets,
                'metrics': metrics,
                'data_info': {
                    'time_step': time_step,
                    'num_nodes': data.num_nodes,
                    'num_candidate_edges': data.edge_index.size(1)
                }
            }
    
    def evaluate_multiple_timesteps(self, model: PhysicsGraphTransformer,
                                   data_loader: PowerGridDataLoader,
                                   time_steps: Optional[List[int]] = None,
                                   threshold: float = 0.5) -> Dict[str, Any]:
        """评估多个时间步的预测"""
        if time_steps is None:
            time_steps = self.config.data.time_steps[:10]  # 默认评估前10个时间步
        
        self.logger.info(f"开始评估 {len(time_steps)} 个时间步")
        
        timestep_results = []
        aggregate_metrics = {}
        
        for time_step in tqdm(time_steps, desc="评估时间步"):
            try:
                result = self.evaluate_single_prediction(
                    model, data_loader, time_step, threshold
                )
                timestep_results.append(result)
                
                # 聚合指标
                metrics = result['metrics']
                for category, category_metrics in metrics.items():
                    if isinstance(category_metrics, dict):
                        for metric_name, value in category_metrics.items():
                            key = f"{category}_{metric_name}"
                            if key not in aggregate_metrics:
                                aggregate_metrics[key] = []
                            aggregate_metrics[key].append(value)
                    else:
                        if category not in aggregate_metrics:
                            aggregate_metrics[category] = []
                        aggregate_metrics[category].append(category_metrics)
                        
            except Exception as e:
                self.logger.warning(f"时间步 {time_step} 评估失败: {e}")
                continue
        
        # 计算聚合统计
        summary_metrics = {}
        for key, values in aggregate_metrics.items():
            if values:  # 确保列表不为空
                summary_metrics[f"{key}_mean"] = np.mean(values)
                summary_metrics[f"{key}_std"] = np.std(values)
                summary_metrics[f"{key}_min"] = np.min(values)
                summary_metrics[f"{key}_max"] = np.max(values)
        
        return {
            'timestep_results': timestep_results,
            'summary_metrics': summary_metrics,
            'evaluation_config': {
                'time_steps': time_steps,
                'threshold': threshold,
                'num_timesteps': len(timestep_results)
            }
        }
    
    def threshold_analysis(self, model: PhysicsGraphTransformer,
                          data_loader: PowerGridDataLoader,
                          time_step: int = 0,
                          thresholds: Optional[List[float]] = None) -> Dict[str, Any]:
        """阈值敏感性分析"""
        if thresholds is None:
            thresholds = np.arange(0.1, 0.95, 0.05).tolist()
        
        self.logger.info(f"开始阈值分析，测试 {len(thresholds)} 个阈值")
        
        threshold_results = []
        
        for threshold in tqdm(thresholds, desc="测试阈值"):
            result = self.evaluate_single_prediction(
                model, data_loader, time_step, threshold
            )
            
            # 提取关键指标
            topo_metrics = result['metrics'].get('topology', {})
            threshold_results.append({
                'threshold': threshold,
                'f1_score': topo_metrics.get('f1_score', 0.0),
                'precision': topo_metrics.get('precision', 0.0),
                'recall': topo_metrics.get('recall', 0.0),
                'num_predicted_edges': topo_metrics.get('num_predicted_edges', 0),
                'num_true_edges': topo_metrics.get('num_true_edges', 0),
                'composite_score': result['metrics'].get('composite_score', 0.0)
            })
        
        # 找到最佳阈值
        best_f1_idx = max(range(len(threshold_results)), 
                         key=lambda i: threshold_results[i]['f1_score'])
        best_composite_idx = max(range(len(threshold_results)),
                               key=lambda i: threshold_results[i]['composite_score'])
        
        return {
            'threshold_results': threshold_results,
            'best_f1_threshold': thresholds[best_f1_idx],
            'best_composite_threshold': thresholds[best_composite_idx],
            'optimal_metrics': {
                'f1': threshold_results[best_f1_idx],
                'composite': threshold_results[best_composite_idx]
            }
        }
    
    def error_analysis(self, model: PhysicsGraphTransformer,
                      data_loader: PowerGridDataLoader,
                      time_steps: Optional[List[int]] = None) -> Dict[str, Any]:
        """错误分析"""
        if time_steps is None:
            time_steps = self.config.data.time_steps[:5]
        
        self.logger.info("开始错误分析")
        
        error_patterns = {
            'false_positives': [],  # 误报边
            'false_negatives': [],  # 漏报边
            'parameter_errors': [],  # 参数误差
            'spatial_errors': []     # 空间误差模式
        }
        
        for time_step in time_steps:
            result = self.evaluate_single_prediction(
                model, data_loader, time_step,
                threshold=self.config.evaluation.edge_threshold
            )
            
            predicted_edges = result['predictions']['edge_index']
            true_edges = result['targets']['true_edge_index']
            
            # 分析误报和漏报
            fp_edges, fn_edges = self._analyze_prediction_errors(
                predicted_edges, true_edges, data_loader.buses
            )
            
            error_patterns['false_positives'].extend(fp_edges)
            error_patterns['false_negatives'].extend(fn_edges)
            
            # 参数误差分析
            if result['targets']['true_edge_params'].size(0) > 0:
                param_errors = self._analyze_parameter_errors(
                    result['predictions']['edge_params'],
                    result['targets']['true_edge_params']
                )
                error_patterns['parameter_errors'].extend(param_errors)
        
        # 错误统计
        error_summary = {
            'total_false_positives': len(error_patterns['false_positives']),
            'total_false_negatives': len(error_patterns['false_negatives']),
            'avg_parameter_error': np.mean([e['error'] for e in error_patterns['parameter_errors']]) if error_patterns['parameter_errors'] else 0.0,
            'error_patterns': self._identify_error_patterns(error_patterns)
        }
        
        return {
            'error_details': error_patterns,
            'error_summary': error_summary
        }
    
    def _analyze_prediction_errors(self, predicted_edges: torch.Tensor,
                                 true_edges: torch.Tensor,
                                 buses_df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """分析预测错误"""
        # 转换为集合便于比较
        pred_set = set(map(tuple, predicted_edges.t().numpy().tolist()))
        true_set = set(map(tuple, true_edges.t().numpy().tolist()))
        
        # 考虑无向图的对称性
        pred_set_sym = set()
        true_set_sym = set()
        
        for src, dst in pred_set:
            pred_set_sym.add((min(src, dst), max(src, dst)))
        
        for src, dst in true_set:
            true_set_sym.add((min(src, dst), max(src, dst)))
        
        # 找到误报和漏报
        false_positives = pred_set_sym - true_set_sym
        false_negatives = true_set_sym - pred_set_sym
        
        # 添加空间信息
        fp_edges = []
        for src, dst in false_positives:
            if src < len(buses_df) and dst < len(buses_df):
                distance = np.sqrt(
                    (buses_df.iloc[src]['x'] - buses_df.iloc[dst]['x'])**2 +
                    (buses_df.iloc[src]['y'] - buses_df.iloc[dst]['y'])**2
                )
                fp_edges.append({
                    'edge': (src, dst),
                    'distance': distance,
                    'type': 'false_positive'
                })
        
        fn_edges = []
        for src, dst in false_negatives:
            if src < len(buses_df) and dst < len(buses_df):
                distance = np.sqrt(
                    (buses_df.iloc[src]['x'] - buses_df.iloc[dst]['x'])**2 +
                    (buses_df.iloc[src]['y'] - buses_df.iloc[dst]['y'])**2
                )
                fn_edges.append({
                    'edge': (src, dst),
                    'distance': distance,
                    'type': 'false_negative'
                })
        
        return fp_edges, fn_edges
    
    def _analyze_parameter_errors(self, predicted_params: torch.Tensor,
                                true_params: torch.Tensor) -> List[Dict]:
        """分析参数预测错误"""
        min_len = min(predicted_params.size(0), true_params.size(0))
        if min_len == 0:
            return []
        
        errors = []
        pred = predicted_params[:min_len].numpy()
        true = true_params[:min_len].numpy()
        
        for i in range(min_len):
            r_error = abs(pred[i, 0] - true[i, 0])
            x_error = abs(pred[i, 1] - true[i, 1])
            total_error = r_error + x_error
            
            errors.append({
                'edge_idx': i,
                'r_error': r_error,
                'x_error': x_error,
                'error': total_error,
                'r_pred': pred[i, 0],
                'x_pred': pred[i, 1],
                'r_true': true[i, 0],
                'x_true': true[i, 1]
            })
        
        return errors
    
    def _identify_error_patterns(self, error_patterns: Dict[str, List]) -> Dict[str, Any]:
        """识别错误模式"""
        patterns = {}
        
        # 分析误报边的距离分布
        if error_patterns['false_positives']:
            fp_distances = [e['distance'] for e in error_patterns['false_positives']]
            patterns['fp_distance_stats'] = {
                'mean': np.mean(fp_distances),
                'std': np.std(fp_distances),
                'median': np.median(fp_distances)
            }
        
        # 分析漏报边的距离分布
        if error_patterns['false_negatives']:
            fn_distances = [e['distance'] for e in error_patterns['false_negatives']]
            patterns['fn_distance_stats'] = {
                'mean': np.mean(fn_distances),
                'std': np.std(fn_distances),
                'median': np.median(fn_distances)
            }
        
        # 参数误差模式
        if error_patterns['parameter_errors']:
            r_errors = [e['r_error'] for e in error_patterns['parameter_errors']]
            x_errors = [e['x_error'] for e in error_patterns['parameter_errors']]
            
            patterns['parameter_error_stats'] = {
                'r_error_mean': np.mean(r_errors),
                'x_error_mean': np.mean(x_errors),
                'r_error_std': np.std(r_errors),
                'x_error_std': np.std(x_errors)
            }
        
        return patterns
    
    def generate_evaluation_report(self, model: PhysicsGraphTransformer,
                                 data_loader: PowerGridDataLoader,
                                 output_dir: Optional[str] = None) -> str:
        """生成完整的评估报告"""
        if output_dir is None:
            output_dir = self.config.experiment.output_dir
        
        report_dir = os.path.join(output_dir, 'evaluation_report')
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("开始生成评估报告")
        
        # 1. 多时间步评估
        multi_eval = self.evaluate_multiple_timesteps(model, data_loader)
        
        # 2. 阈值分析
        threshold_analysis = self.threshold_analysis(model, data_loader)
        
        # 3. 错误分析
        error_analysis = self.error_analysis(model, data_loader)
        
        # 4. 单个详细评估（用于可视化）
        detailed_eval = self.evaluate_single_prediction(
            model, data_loader, time_step=0,
            threshold=threshold_analysis['best_f1_threshold']
        )
        
        # 保存结果
        results = {
            'multi_timestep_evaluation': multi_eval,
            'threshold_analysis': threshold_analysis,
            'error_analysis': error_analysis,
            'detailed_evaluation': detailed_eval,
            'model_info': self._get_model_info(model),
            'evaluation_config': self.config.evaluation.__dict__
        }
        
        # 保存JSON报告
        report_path = os.path.join(report_dir, 'evaluation_results.json')
        save_json(results, report_path)
        
        # 生成可视化
        if self.config.evaluation.save_plots:
            self._generate_report_visualizations(results, report_dir)
        
        # 生成文本摘要
        summary_path = self._generate_text_summary(results, report_dir)
        
        self.logger.info(f"评估报告已生成: {report_dir}")
        return summary_path
    
    def _get_model_info(self, model: PhysicsGraphTransformer) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'model_type': model.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_config': model.config.__dict__ if hasattr(model, 'config') else {}
        }
    
    def _generate_report_visualizations(self, results: Dict[str, Any], 
                                      output_dir: str):
        """生成报告可视化"""
        # 拓扑对比图
        detailed_eval = results['detailed_evaluation']
        viz_path = os.path.join(output_dir, 'topology_comparison.png')
        
        # 注意：这里需要实际的坐标数据
        # plot_topology_comparison(
        #     detailed_eval['predictions'],
        #     detailed_eval['targets'],
        #     coordinates=...,  # 需要从data_loader获取
        #     save_path=viz_path
        # )
        
        # 错误分析图
        error_viz_path = os.path.join(output_dir, 'error_analysis.png')
        # plot_error_analysis(
        #     results['error_analysis'],
        #     save_path=error_viz_path
        # )
    
    def _generate_text_summary(self, results: Dict[str, Any], 
                              output_dir: str) -> str:
        """生成文本摘要"""
        summary_path = os.path.join(output_dir, 'evaluation_summary.txt')
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("电力网格拓扑重建模型评估报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 模型信息
            model_info = results['model_info']
            f.write("模型信息:\n")
            f.write(f"  类型: {model_info['model_type']}\n")
            f.write(f"  参数数量: {model_info['total_parameters']:,}\n")
            f.write(f"  可训练参数: {model_info['trainable_parameters']:,}\n\n")
            
            # 综合性能指标
            summary_metrics = results['multi_timestep_evaluation']['summary_metrics']
            f.write("综合性能指标:\n")
            
            # 拓扑指标
            f1_mean = summary_metrics.get('topology_f1_score_mean', 0)
            precision_mean = summary_metrics.get('topology_precision_mean', 0)
            recall_mean = summary_metrics.get('topology_recall_mean', 0)
            
            f.write(f"  拓扑F1分数: {f1_mean:.3f} ± {summary_metrics.get('topology_f1_score_std', 0):.3f}\n")
            f.write(f"  精确率: {precision_mean:.3f} ± {summary_metrics.get('topology_precision_std', 0):.3f}\n")
            f.write(f"  召回率: {recall_mean:.3f} ± {summary_metrics.get('topology_recall_std', 0):.3f}\n")
            
            # 参数指标
            if 'parameters_param_mae_avg_mean' in summary_metrics:
                param_mae = summary_metrics['parameters_param_mae_avg_mean']
                f.write(f"  参数MAE: {param_mae:.4f} ± {summary_metrics.get('parameters_param_mae_avg_std', 0):.4f}\n")
            
            # 最佳阈值
            threshold_analysis = results['threshold_analysis']
            f.write(f"\n最佳阈值:\n")
            f.write(f"  F1最优阈值: {threshold_analysis['best_f1_threshold']:.2f}\n")
            f.write(f"  综合最优阈值: {threshold_analysis['best_composite_threshold']:.2f}\n")
            
            # 错误分析
            error_summary = results['error_analysis']['error_summary']
            f.write(f"\n错误分析:\n")
            f.write(f"  误报边数: {error_summary['total_false_positives']}\n")
            f.write(f"  漏报边数: {error_summary['total_false_negatives']}\n")
            f.write(f"  平均参数误差: {error_summary['avg_parameter_error']:.4f}\n")
        
        return summary_path