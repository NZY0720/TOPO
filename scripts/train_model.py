#!/usr/bin/env python3
"""
主训练脚本 - Power Grid Topology Reconstruction
用法：python scripts/train_model.py --config_path config.yaml --data_path ./data
"""

import os
import sys
import argparse
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from power_grid_topology.config.base_config import Config, get_default_config
from power_grid_topology.data.data_loader import PowerGridDataLoader
from power_grid_topology.models.graph_transformer import create_model
from power_grid_topology.training.trainer import PowerGridTrainer
from power_grid_topology.evaluation.evaluator import ModelEvaluator
from power_grid_topology.utils.io_utils import load_yaml, save_json, create_experiment_summary
from power_grid_topology.utils.logging_utils import setup_logger


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="电力网格拓扑重建模型训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基本参数
    parser.add_argument('--config_path', type=str, default=None,
                       help='配置文件路径（YAML格式）')
    parser.add_argument('--data_path', type=str, required=True,
                       help='数据文件夹路径')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='输出目录')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='实验名称')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='学习率')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批大小')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, choices=['standard', 'hybrid'], 
                       default=None, help='模型类型')
    parser.add_argument('--hidden_dim', type=int, default=None,
                       help='隐藏层维度')
    parser.add_argument('--num_heads', type=int, default=None,
                       help='注意力头数')
    parser.add_argument('--num_layers', type=int, default=None,
                       help='网络层数')
    
    # 数据参数
    parser.add_argument('--unobservable_ratio', type=float, default=None,
                       help='不可观测节点比例')
    parser.add_argument('--k_neighbors', type=int, default=None,
                       help='候选图k近邻数')
    
    # 执行选项
    parser.add_argument('--train_only', action='store_true',
                       help='只训练，不评估')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='从检查点恢复训练')
    parser.add_argument('--debug', action='store_true',
                       help='调试模式')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'],
                       default='auto', help='计算设备')
    
    return parser.parse_args()


def setup_config(args):
    """设置配置"""
    # 加载基础配置
    if args.config_path and os.path.exists(args.config_path):
        print(f"📄 加载配置文件: {args.config_path}")
        config_dict = load_yaml(args.config_path)
        config = Config.from_dict(config_dict)
    else:
        print("📄 使用默认配置")
        config = get_default_config()
    
    # 命令行参数覆盖配置
    if args.experiment_name:
        config.experiment.name = args.experiment_name
    
    if args.output_dir:
        config.experiment.output_dir = args.output_dir
        config.experiment.log_dir = os.path.join(args.output_dir, 'logs')
        config.experiment.checkpoint_dir = os.path.join(args.output_dir, 'models')
        config.experiment.figure_dir = os.path.join(args.output_dir, 'figures')
    
    # 数据配置
    config.data.data_path = args.data_path
    if args.unobservable_ratio is not None:
        config.data.unobservable_ratio = args.unobservable_ratio
    if args.k_neighbors is not None:
        config.data.candidate_k_neighbors = args.k_neighbors
    
    # 训练配置
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.device != 'auto':
        config.training.device = args.device
    
    # 模型配置
    if args.model_type is not None:
        config.model.model_type = args.model_type
    if args.hidden_dim is not None:
        config.model.d_hidden = args.hidden_dim
    if args.num_heads is not None:
        config.model.n_heads = args.num_heads
    if args.num_layers is not None:
        config.model.n_layers = args.num_layers
    
    # 调试模式
    if args.debug:
        config.experiment.debug_mode = True
        config.training.epochs = min(10, config.training.epochs)
        config.data.time_steps = list(range(5))
        config.training.log_every = 1
        config.training.validate_every = 2
    
    return config


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 设置PyTorch的确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """主函数"""
    print("🚀 启动电力网格拓扑重建训练")
    print("=" * 60)
    
    # 解析参数
    args = parse_arguments()
    
    # 设置随机种子
    set_seed(args.seed)
    print(f"🎲 随机种子设置为: {args.seed}")
    
    # 设置配置
    config = setup_config(args)
    
    # 创建输出目录
    os.makedirs(config.experiment.output_dir, exist_ok=True)
    os.makedirs(config.experiment.log_dir, exist_ok=True)
    os.makedirs(config.experiment.checkpoint_dir, exist_ok=True)
    os.makedirs(config.experiment.figure_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logger(
        'main', 
        os.path.join(config.experiment.log_dir, 'main.log')
    )
    
    logger.info(f"实验名称: {config.experiment.name}")
    logger.info(f"输出目录: {config.experiment.output_dir}")
    logger.info(f"数据路径: {config.data.data_path}")
    
    try:
        # 1. 加载数据
        print("\n📂 步骤 1/5: 加载数据")
        print("-" * 30)
        
        data_loader = PowerGridDataLoader(config.data)
        data_loader.load_data(config.data.data_path)
        
        # 模拟部分可观测性
        masked_nodes = data_loader.mask_unobservable_nodes(
            config.data.unobservable_ratio,
            seed=args.seed
        )
        
        # 获取数据信息
        data_info = data_loader.get_data_info()
        logger.info(f"数据集信息: {data_info}")
        
        # 2. 创建模型
        print("\n🧠 步骤 2/5: 创建模型")
        print("-" * 30)
        
        # 获取输入特征维度
        sample_data = data_loader.create_graph_data(0)
        input_dim = sample_data.x.size(1)
        
        model = create_model(config.model, input_dim)
        device = torch.device(config.training.device)
        model = model.to(device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"模型类型: {model.__class__.__name__}")
        print(f"总参数数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        print(f"计算设备: {device}")
        
        logger.info(f"模型创建完成，参数数量: {total_params:,}")
        
        # 3. 训练模型
        print("\n🏋️ 步骤 3/5: 训练模型")
        print("-" * 30)
        
        trainer = PowerGridTrainer(config)
        
        # 从检查点恢复（如果指定）
        if args.resume_from and os.path.exists(args.resume_from):
            print(f"📦 从检查点恢复: {args.resume_from}")
            trainer.load_checkpoint(args.resume_from)
        
        # 开始训练
        start_time = time.time()
        trainer.train(data_loader)
        training_time = time.time() - start_time
        
        print(f"✅ 训练完成！耗时: {training_time:.2f}秒")
        logger.info(f"训练完成，耗时: {training_time:.2f}秒")
        
        # 4. 评估模型
        if not args.train_only:
            print("\n📊 步骤 4/5: 评估模型")
            print("-" * 30)
            
            evaluator = ModelEvaluator(config)
            
            # 加载最佳模型
            best_model_path = os.path.join(config.experiment.checkpoint_dir, 'best_model.pth')
            
            if os.path.exists(best_model_path):
                model = evaluator.load_model(best_model_path, model)
            
            # 生成评估报告
            evaluation_start = time.time()
            report_path = evaluator.generate_evaluation_report(model, data_loader)
            evaluation_time = time.time() - evaluation_start
            
            print(f"✅ 评估完成！耗时: {evaluation_time:.2f}秒")
            print(f"📋 评估报告: {report_path}")
            logger.info(f"评估完成，报告路径: {report_path}")
        
        # 5. 保存实验配置和摘要
        print("\n💾 步骤 5/5: 保存实验结果")
        print("-" * 30)
        
        # 保存配置
        config_path = os.path.join(config.experiment.output_dir, 'experiment_config.json')
        save_json(config.to_dict(), config_path)
        
        # 创建实验摘要
        summary_path = create_experiment_summary(config.experiment.output_dir)
        
        print(f"✅ 实验配置已保存: {config_path}")
        print(f"📝 实验摘要已生成: {summary_path}")
        
        # 最终结果
        print("\n🎉 训练流程完成！")
        print("=" * 60)
        print(f"📁 输出目录: {config.experiment.output_dir}")
        print(f"📋 实验摘要: {summary_path}")
        
        if not args.train_only:
            print(f"📊 评估报告: {report_path}")
        
        # 打印关键指标
        if hasattr(trainer, 'training_history') and trainer.training_history['metrics']:
            final_metrics = trainer.training_history['metrics'][-1]
            topo_metrics = final_metrics.get('topology', {})
            
            print("\n📈 最终性能指标:")
            print(f"  F1分数: {topo_metrics.get('f1_score', 0):.3f}")
            print(f"  精确率: {topo_metrics.get('precision', 0):.3f}")
            print(f"  召回率: {topo_metrics.get('recall', 0):.3f}")
            
            composite_score = final_metrics.get('composite_score', 0)
            print(f"  综合得分: {composite_score:.3f}")
        
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        logger.warning("训练被用户中断")
        
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        logger.error(f"训练失败: {e}", exc_info=True)
        raise
    
    finally:
        # 清理资源
        if 'trainer' in locals():
            if hasattr(trainer, 'writer') and trainer.writer:
                trainer.writer.close()


if __name__ == "__main__":
    main()