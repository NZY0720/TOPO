#!/usr/bin/env python3
"""
Updated configuration for actual data format
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from omegaconf import DictConfig
import torch


@dataclass
class DataConfig:
    """数据配置 - 适配实际数据格式"""
    data_path: str = "./data"
    
    # 实际数据文件名 (1MVurban0sw数据集)
    bus_file: str = "1MVurban0sw_bus_with_local_coords.csv"
    line_file: str = "1MVurban0sw_lines_with_coordinates.csv"
    voltage_ts_file: str = "1MVurban0sw_voltage_timeseries.csv"
    loading_ts_file: str = "1MVurban0sw_loading_timeseries.csv"
    loads_file: Optional[str] = "1MVurban0sw_loads_with_coordinates.csv"
    generators_file: Optional[str] = "1MVurban0sw_generators_with_coordinates.csv"
    final_state_file: Optional[str] = "1MVurban0sw_final_state_with_coords.csv"
    summary_stats_file: Optional[str] = "1MVurban0sw_summary_statistics.csv"
    
    # 数据处理参数
    unobservable_ratio: float = 0.25
    candidate_k_neighbors: int = 5
    time_steps: Optional[List[int]] = None
    normalize_features: bool = True
    add_noise: bool = False
    noise_std: float = 0.01
    
    # 数据增强
    spatial_jitter: float = 0.0
    temporal_subsample: float = 1.0
    
    # 实际数据特有参数
    voltage_threshold_pu: float = 0.05  # 电压偏差阈值
    use_final_state: bool = False       # 是否使用最终状态数据
    filter_inactive_components: bool = True  # 过滤非投运设备


@dataclass 
class ModelConfig:
    """模型配置"""
    # 基础架构参数
    d_hidden: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1
    
    # 输入特征维度（自动计算，基于实际数据特征）
    d_input: Optional[int] = None
    
    # 输出参数
    n_edge_params: int = 2  # R, X
    
    # 激活函数
    activation: str = "relu"
    use_layer_norm: bool = True
    use_residual: bool = True
    
    # 模型类型选择
    model_type: str = "standard"  # standard, hybrid


@dataclass
class PhysicsConfig:
    """物理约束配置"""
    # 损失权重
    alpha_kcl: float = 1.0
    alpha_kvl: float = 1.0
    alpha_topology: float = 1.0
    alpha_parameter: float = 1.0
    alpha_sparsity: float = 0.01
    alpha_geographic: float = 0.01
    
    # 物理约束参数 (适配实际电网参数)
    voltage_tolerance: float = 0.05      # 电压偏差容忍度 (5%)
    current_tolerance: float = 0.1       # 电流计算容忍度
    impedance_min: float = 1e-6         # 最小阻抗值
    power_tolerance: float = 0.1         # 功率平衡容忍度
    
    # 实际电网物理约束
    voltage_lower_bound: float = 0.95    # 电压下限 (标幺值)
    voltage_upper_bound: float = 1.05    # 电压上限 (标幺值)
    line_loading_limit: float = 100.0    # 线路负载限制 (%)


@dataclass
class TrainingConfig:
    """训练配置"""
    # 优化器参数
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    betas: tuple = (0.9, 0.999)
    
    # 学习率调度
    scheduler: str = "reduce_on_plateau"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    
    # 训练参数
    epochs: int = 100
    batch_size: int = 1  # 通常为1，处理单个图
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 20
    
    # 验证和保存
    validate_every: int = 5
    save_every: int = 10
    log_every: int = 1
    
    # 设备配置
    device: str = "auto"  # auto, cpu, cuda
    num_workers: int = 4
    pin_memory: bool = True
    
    # 混合精度训练
    use_amp: bool = False
    
    # 梯度累积
    gradient_accumulation_steps: int = 1


@dataclass
class EvaluationConfig:
    """评估配置"""
    # 拓扑重建参数
    edge_threshold: float = 0.5
    ensure_radial: bool = True
    
    # 评估指标
    compute_topology_metrics: bool = True
    compute_parameter_metrics: bool = True
    compute_physics_metrics: bool = True
    compute_statistical_metrics: bool = True
    
    # 可视化
    save_plots: bool = True
    plot_format: str = "png"
    plot_dpi: int = 300
    
    # 阈值分析
    threshold_range: tuple = (0.1, 0.9)
    threshold_step: float = 0.05
    
    # 错误分析
    spatial_error_analysis: bool = True
    temporal_error_analysis: bool = False


@dataclass
class ExperimentConfig:
    """实验配置"""
    # 实验标识
    name: str = "1MVurban0sw_experiment"
    seed: int = 42
    description: str = "1MVurban0sw配电网拓扑重建实验"
    tags: List[str] = field(default_factory=lambda: ["1MVurban0sw", "topology_reconstruction"])
    
    # 输出路径
    output_dir: str = "./outputs"
    log_dir: str = "./outputs/logs"
    checkpoint_dir: str = "./outputs/models"
    figure_dir: str = "./outputs/figures"
    
    # 实验跟踪
    use_wandb: bool = False
    wandb_project: str = "power-grid-topology-1MVurban0sw"
    use_tensorboard: bool = True
    
    # 调试选项
    debug_mode: bool = False
    profile_training: bool = False
    
    # 实验特定配置
    dataset_name: str = "1MVurban0sw"
    network_type: str = "urban_distribution"
    voltage_level: str = "medium_voltage"


@dataclass
class Config:
    """完整配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def __post_init__(self):
        """后处理配置"""
        # 自动设置设备
        if self.training.device == "auto":
            self.training.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 设置默认时间步 (基于实际数据的288个时间步)
        if self.data.time_steps is None:
            self.data.time_steps = list(range(min(50, 288)))  # 使用前50个时间步
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """从字典创建配置"""
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            physics=PhysicsConfig(**config_dict.get('physics', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {})),
            experiment=ExperimentConfig(**config_dict.get('experiment', {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'physics': self.physics.__dict__,
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__,
            'experiment': self.experiment.__dict__
        }


def get_default_config() -> Config:
    """获取默认配置"""
    return Config()


def get_1MVurban0sw_config() -> Config:
    """获取1MVurban0sw数据集专用配置"""
    config = Config()
    
    # 数据配置
    config.data.unobservable_ratio = 0.3  # 30%不可观测
    config.data.candidate_k_neighbors = 6  # 增加候选连接
    config.data.normalize_features = True
    config.data.add_noise = True
    config.data.noise_std = 0.005  # 小噪声
    
    # 模型配置
    config.model.d_hidden = 128
    config.model.n_heads = 6
    config.model.n_layers = 4
    config.model.dropout = 0.15
    
    # 物理约束配置 (适配中压配电网)
    config.physics.alpha_kcl = 1.5
    config.physics.alpha_kvl = 1.2
    config.physics.alpha_topology = 1.0
    config.physics.alpha_parameter = 0.8
    config.physics.voltage_tolerance = 0.06  # 6%容忍度
    
    # 训练配置
    config.training.epochs = 150
    config.training.learning_rate = 8e-4
    config.training.early_stopping_patience = 25
    config.training.validate_every = 3
    
    # 评估配置
    config.evaluation.edge_threshold = 0.6  # 稍高的阈值
    config.evaluation.ensure_radial = True
    config.evaluation.spatial_error_analysis = True
    
    return config


def get_quick_test_config() -> Config:
    """获取快速测试配置"""
    config = Config()
    
    # 减少训练时间
    config.training.epochs = 20
    config.data.time_steps = list(range(10))  # 只用10个时间步
    
    # 减少模型复杂度
    config.model.d_hidden = 64
    config.model.n_heads = 2
    config.model.n_layers = 2
    
    # 快速验证
    config.training.validate_every = 2
    config.training.log_every = 1
    
    return config


def get_production_config() -> Config:
    """获取生产环境配置"""
    config = get_1MVurban0sw_config()
    
    # 更严格的训练
    config.training.epochs = 300
    config.training.early_stopping_patience = 50
    config.training.gradient_clip_norm = 0.5
    
    # 更大的模型
    config.model.d_hidden = 256
    config.model.n_heads = 8
    config.model.n_layers = 6
    
    # 更多的候选连接
    config.data.candidate_k_neighbors = 8
    
    # 启用所有评估指标
    config.evaluation.compute_statistical_metrics = True
    config.evaluation.spatial_error_analysis = True
    config.evaluation.temporal_error_analysis = True
    
    return config


# 预定义配置集合
PREDEFINED_CONFIGS = {
    'default': get_default_config,
    '1MVurban0sw': get_1MVurban0sw_config,
    'quick_test': get_quick_test_config,
    'production': get_production_config
}


def get_config_by_name(config_name: str) -> Config:
    """根据名称获取预定义配置"""
    if config_name not in PREDEFINED_CONFIGS:
        raise ValueError(f"未知配置名称: {config_name}. 可用配置: {list(PREDEFINED_CONFIGS.keys())}")
    
    return PREDEFINED_CONFIGS[config_name]()