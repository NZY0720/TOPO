#!/usr/bin/env python3
"""
Base configuration classes for Power Grid Topology Reconstruction
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from omegaconf import DictConfig
import torch


@dataclass
class DataConfig:
    """数据配置"""
    data_path: str = "./data"
    bus_file: str = "buses.csv"
    line_file: str = "lines.csv"
    voltage_ts_file: str = "voltage_timeseries.csv"
    loading_ts_file: str = "loading_timeseries.csv"
    loads_file: Optional[str] = "loads.csv"
    generators_file: Optional[str] = "generators.csv"
    
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


@dataclass 
class ModelConfig:
    """模型配置"""
    # 基础架构参数
    d_hidden: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1
    
    # 输入特征维度（自动计算）
    d_input: Optional[int] = None
    
    # 输出参数
    n_edge_params: int = 2  # R, X
    
    # 激活函数
    activation: str = "relu"
    use_layer_norm: bool = True
    use_residual: bool = True


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
    
    # 物理约束参数
    voltage_tolerance: float = 0.05
    current_tolerance: float = 0.1
    impedance_min: float = 1e-6


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
    
    # 可视化
    save_plots: bool = True
    plot_format: str = "png"
    plot_dpi: int = 300


@dataclass
class ExperimentConfig:
    """实验配置"""
    # 实验标识
    name: str = "default_experiment"
    seed: int = 42
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # 输出路径
    output_dir: str = "./outputs"
    log_dir: str = "./outputs/logs"
    checkpoint_dir: str = "./outputs/models"
    figure_dir: str = "./outputs/figures"
    
    # 实验跟踪
    use_wandb: bool = False
    wandb_project: str = "power-grid-topology"
    use_tensorboard: bool = True
    
    # 调试选项
    debug_mode: bool = False
    profile_training: bool = False


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
        
        # 设置默认时间步
        if self.data.time_steps is None:
            self.data.time_steps = list(range(20))
    
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


def get_small_test_config() -> Config:
    """获取小规模测试配置"""
    config = Config()
    
    # 减少模型复杂度
    config.model.d_hidden = 64
    config.model.n_heads = 2
    config.model.n_layers = 2
    
    # 减少训练轮数
    config.training.epochs = 20
    config.data.time_steps = list(range(5))
    
    # 快速验证
    config.training.validate_every = 2
    config.training.log_every = 1
    
    return config


def get_large_scale_config() -> Config:
    """获取大规模实验配置"""
    config = Config()
    
    # 增加模型容量
    config.model.d_hidden = 256
    config.model.n_heads = 8
    config.model.n_layers = 6
    
    # 更长训练
    config.training.epochs = 200
    config.training.learning_rate = 5e-4
    config.data.time_steps = list(range(50))
    
    # 更频繁的候选连接
    config.data.candidate_k_neighbors = 10
    
    return config