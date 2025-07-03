#!/usr/bin/env python3
"""
实验运行脚本 - 支持多种实验配置和批量运行
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import itertools

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from power_grid_topology.config.base_config import Config
from power_grid_topology.data.data_loader import PowerGridDataLoader
from power_grid_topology.training.trainer import PowerGridTrainer
from power_grid_topology.evaluation.evaluator import ModelEvaluator
from power_grid_topology.utils.io_utils import load_yaml, save_json
from power_grid_topology.utils.logging_utils import setup_logger


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, base_config: Config, base_output_dir: str):
        self.base_config = base_config
        self.base_output_dir = base_output_dir
        self.experiment_results = []
        
        # 设置主日志记录器
        self.logger = setup_logger(
            'experiment_runner',
            os.path.join(base_output_dir, 'experiment_runner.log')
        )
    
    def run_single_experiment(self, config: Config, experiment_id: str) -> Dict[str, Any]:
        """运行单个实验"""
        
        # 设置实验特定的输出目录
        exp_output_dir = os.path.join(self.base_output_dir, f"exp_{experiment_id}")
        config.experiment.output_dir = exp_output_dir
        config.experiment.log_dir = os.path.join(exp_output_dir, 'logs')
        config.experiment.checkpoint_dir = os.path.join(exp_output_dir, 'models')
        config.experiment.figure_dir = os.path.join(exp_output_dir, 'figures')
        config.experiment.name = f"experiment_{experiment_id}"
        
        # 创建输出目录
        os.makedirs(exp_output_dir, exist_ok=True)
        
        self.logger.info(f"🚀 开始实验 {experiment_id}")
        self.logger.info(f"📁 输出目录: {exp_output_dir}")
        
        try:
            start_time = time.time()
            
            # 1. 加载数据
            self.logger.info("📂 加载数据...")
            data_loader = PowerGridDataLoader(config.data)
            data_loader.load_data(config.data.data_path)
            data_loader.mask_unobservable_nodes(
                config.data.unobservable_ratio,
                seed=config.experiment.seed
            )
            
            # 2. 创建和训练模型
            self.logger.info("🏋️ 开始训练...")
            trainer = PowerGridTrainer(config)
            trainer.train(data_loader)
            
            # 3. 评估模型
            self.logger.info("📊 开始评估...")
            evaluator = ModelEvaluator(config)
            
            # 加载最佳模型
            best_model_path = os.path.join(config.experiment.checkpoint_dir, 'best_model.pth')
            if os.path.exists(best_model_path):
                from power_grid_topology.models.graph_transformer import create_model
                
                sample_data = data_loader.create_graph_data(0)
                input_dim = sample_data.x.size(1)
                model = create_model(config.model, input_dim)
                model = evaluator.load_model(best_model_path, model)
            else:
                self.logger.warning("最佳模型文件不存在，使用最终模型")
                model = trainer.model
            
            # 生成评估报告
            report_path = evaluator.generate_evaluation_report(model, data_loader)
            
            # 4. 收集结果
            training_time = time.time() - start_time
            
            # 获取最终指标
            final_metrics = {}
            if hasattr(trainer, 'training_history') and trainer.training_history['metrics']:
                final_metrics = trainer.training_history['metrics'][-1]
            
            # 实验结果
            experiment_result = {
                'experiment_id': experiment_id,
                'config': config.to_dict(),
                'training_time': training_time,
                'final_metrics': final_metrics,
                'best_val_loss': trainer.best_val_loss,
                'total_epochs': trainer.current_epoch + 1,
                'output_dir': exp_output_dir,
                'report_path': report_path,
                'status': 'completed'
            }
            
            # 保存实验结果
            result_path = os.path.join(exp_output_dir, 'experiment_result.json')
            save_json(experiment_result, result_path)
            
            self.logger.info(f"✅ 实验 {experiment_id} 完成，耗时: {training_time:.2f}秒")
            
            return experiment_result
            
        except Exception as e:
            self.logger.error(f"❌ 实验 {experiment_id} 失败: {e}", exc_info=True)
            
            # 返回失败结果
            return {
                'experiment_id': experiment_id,
                'config': config.to_dict(),
                'status': 'failed',
                'error': str(e),
                'output_dir': exp_output_dir
            }
    
    def run_grid_search(self, param_grid: Dict[str, List[Any]], 
                       data_path: str, max_experiments: Optional[int] = None) -> List[Dict[str, Any]]:
        """运行网格搜索实验"""
        
        self.logger.info("🔍 开始网格搜索实验")
        self.logger.info(f"参数网格: {param_grid}")
        
        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        if max_experiments and len(param_combinations) > max_experiments:
            # 随机采样
            import random
            param_combinations = random.sample(param_combinations, max_experiments)
            self.logger.info(f"随机采样 {max_experiments} 个实验配置")
        
        self.logger.info(f"总实验数量: {len(param_combinations)}")
        
        results = []
        
        for i, param_values in enumerate(param_combinations):
            # 创建实验配置
            config = Config()
            config.__dict__.update(self.base_config.__dict__)
            config.data.data_path = data_path
            
            # 设置参数
            for param_name, param_value in zip(param_names, param_values):
                self._set_nested_param(config, param_name, param_value)
            
            # 设置实验ID
            experiment_id = f"grid_search_{i:03d}"
            
            # 运行实验
            result = self.run_single_experiment(config, experiment_id)
            results.append(result)
            
            # 保存中间结果
            self.experiment_results = results
            self._save_experiment_summary()
        
        self.logger.info(f"🎉 网格搜索完成，共运行 {len(results)} 个实验")
        return results
    
    def run_random_search(self, param_distributions: Dict[str, Any], 
                         data_path: str, n_experiments: int = 10) -> List[Dict[str, Any]]:
        """运行随机搜索实验"""
        
        self.logger.info("🎲 开始随机搜索实验")
        self.logger.info(f"实验数量: {n_experiments}")
        
        results = []
        
        for i in range(n_experiments):
            # 随机采样参数
            config = Config()
            config.__dict__.update(self.base_config.__dict__)
            config.data.data_path = data_path
            
            for param_name, distribution in param_distributions.items():
                if isinstance(distribution, list):
                    # 从列表中随机选择
                    param_value = np.random.choice(distribution)
                elif isinstance(distribution, dict):
                    # 根据分布类型采样
                    if distribution['type'] == 'uniform':
                        param_value = np.random.uniform(distribution['low'], distribution['high'])
                    elif distribution['type'] == 'log_uniform':
                        param_value = np.exp(np.random.uniform(
                            np.log(distribution['low']), np.log(distribution['high'])
                        ))
                    elif distribution['type'] == 'choice':
                        param_value = np.random.choice(distribution['choices'])
                    else:
                        raise ValueError(f"未知的分布类型: {distribution['type']}")
                else:
                    param_value = distribution
                
                self._set_nested_param(config, param_name, param_value)
            
            # 设置实验ID
            experiment_id = f"random_search_{i:03d}"
            
            # 运行实验
            result = self.run_single_experiment(config, experiment_id)
            results.append(result)
            
            # 保存中间结果
            self.experiment_results = results
            self._save_experiment_summary()
        
        self.logger.info(f"🎉 随机搜索完成，共运行 {len(results)} 个实验")
        return results
    
    def _set_nested_param(self, config: Config, param_path: str, value: Any):
        """设置嵌套参数"""
        parts = param_path.split('.')
        obj = config
        
        for part in parts[:-1]:
            obj = getattr(obj, part)
        
        setattr(obj, parts[-1], value)
    
    def _save_experiment_summary(self):
        """保存实验摘要"""
        summary_path = os.path.join(self.base_output_dir, 'experiment_summary.json')
        
        summary = {
            'total_experiments': len(self.experiment_results),
            'completed_experiments': len([r for r in self.experiment_results if r['status'] == 'completed']),
            'failed_experiments': len([r for r in self.experiment_results if r['status'] == 'failed']),
            'results': self.experiment_results
        }
        
        save_json(summary, summary_path)
    
    def analyze_results(self) -> Dict[str, Any]:
        """分析实验结果"""
        if not self.experiment_results:
            return {}
        
        completed_results = [r for r in self.experiment_results if r['status'] == 'completed']
        
        if not completed_results:
            return {'message': '没有成功完成的实验'}
        
        # 提取关键指标
        f1_scores = []
        training_times = []
        val_losses = []
        
        for result in completed_results:
            final_metrics = result.get('final_metrics', {})
            topo_metrics = final_metrics.get('topology', {})
            
            f1_scores.append(topo_metrics.get('f1_score', 0))
            training_times.append(result.get('training_time', 0))
            val_losses.append(result.get('best_val_loss', float('inf')))
        
        # 找到最佳实验
        best_f1_idx = np.argmax(f1_scores)
        best_experiment = completed_results[best_f1_idx]
        
        analysis = {
            'total_experiments': len(self.experiment_results),
            'completed_experiments': len(completed_results),
            'best_experiment': {
                'experiment_id': best_experiment['experiment_id'],
                'f1_score': f1_scores[best_f1_idx],
                'training_time': best_experiment['training_time'],
                'output_dir': best_experiment['output_dir']
            },
            'statistics': {
                'f1_score': {
                    'mean': float(np.mean(f1_scores)),
                    'std': float(np.std(f1_scores)),
                    'min': float(np.min(f1_scores)),
                    'max': float(np.max(f1_scores))
                },
                'training_time': {
                    'mean': float(np.mean(training_times)),
                    'std': float(np.std(training_times)),
                    'min': float(np.min(training_times)),
                    'max': float(np.max(training_times))
                }
            }
        }
        
        # 保存分析结果
        analysis_path = os.path.join(self.base_output_dir, 'experiment_analysis.json')
        save_json(analysis, analysis_path)
        
        return analysis


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="运行电力网格拓扑重建实验")
    
    parser.add_argument('--config', type=str, required=True,
                       help='基础配置文件路径')
    parser.add_argument('--data_path', type=str, required=True,
                       help='数据路径')
    parser.add_argument('--output_dir', type=str, default='./experiments_output',
                       help='实验输出目录')
    parser.add_argument('--experiment_type', type=str, 
                       choices=['single', 'grid_search', 'random_search'],
                       default='single', help='实验类型')
    parser.add_argument('--param_config', type=str, default=None,
                       help='参数配置文件（用于网格搜索和随机搜索）')
    parser.add_argument('--max_experiments', type=int, default=None,
                       help='最大实验数量')
    parser.add_argument('--n_random', type=int, default=10,
                       help='随机搜索实验数量')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 加载基础配置
    print(f"📄 加载配置文件: {args.config}")
    config_dict = load_yaml(args.config)
    base_config = Config.from_dict(config_dict)
    base_config.experiment.seed = args.seed
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建实验运行器
    runner = ExperimentRunner(base_config, args.output_dir)
    
    print(f"🧪 实验类型: {args.experiment_type}")
    print(f"📁 输出目录: {args.output_dir}")
    
    try:
        if args.experiment_type == 'single':
            # 单个实验
            base_config.data.data_path = args.data_path
            result = runner.run_single_experiment(base_config, "single")
            
            if result['status'] == 'completed':
                print(f"✅ 实验完成!")
                print(f"📊 最终F1分数: {result['final_metrics'].get('topology', {}).get('f1_score', 0):.3f}")
            else:
                print(f"❌ 实验失败: {result.get('error', 'Unknown error')}")
        
        elif args.experiment_type == 'grid_search':
            # 网格搜索
            if not args.param_config:
                raise ValueError("网格搜索需要参数配置文件")
            
            param_config = load_yaml(args.param_config)
            param_grid = param_config.get('grid_search', {})
            
            results = runner.run_grid_search(
                param_grid, args.data_path, args.max_experiments
            )
            
            # 分析结果
            analysis = runner.analyze_results()
            print(f"🎉 网格搜索完成!")
            print(f"📊 最佳F1分数: {analysis['best_experiment']['f1_score']:.3f}")
            print(f"🏆 最佳实验: {analysis['best_experiment']['experiment_id']}")
        
        elif args.experiment_type == 'random_search':
            # 随机搜索
            if not args.param_config:
                raise ValueError("随机搜索需要参数配置文件")
            
            param_config = load_yaml(args.param_config)
            param_distributions = param_config.get('random_search', {})
            
            results = runner.run_random_search(
                param_distributions, args.data_path, args.n_random
            )
            
            # 分析结果
            analysis = runner.analyze_results()
            print(f"🎉 随机搜索完成!")
            print(f"📊 最佳F1分数: {analysis['best_experiment']['f1_score']:.3f}")
            print(f"🏆 最佳实验: {analysis['best_experiment']['experiment_id']}")
        
        print(f"\n📁 所有结果保存在: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️  实验被用户中断")
    except Exception as e:
        print(f"\n❌ 实验运行失败: {e}")
        raise


if __name__ == "__main__":
    main()