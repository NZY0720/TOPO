#!/usr/bin/env python3
"""
Logging utilities for Power Grid Topology Reconstruction
"""

import os
import sys
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import numpy as np
import torch


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',     # 青色
        'INFO': '\033[32m',      # 绿色
        'WARNING': '\033[33m',   # 黄色
        'ERROR': '\033[31m',     # 红色
        'CRITICAL': '\033[35m',  # 紫色
    }
    RESET = '\033[0m'
    
    def __init__(self, fmt: str = None, use_color: bool = True):
        super().__init__(fmt)
        self.use_color = use_color and sys.stdout.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        if self.use_color and record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            )
        return super().format(record)


class TensorJSONEncoder(json.JSONEncoder):
    """处理PyTorch张量和NumPy数组的JSON编码器"""
    
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


def setup_logger(name: str, 
                log_file: Optional[str] = None,
                level: Union[str, int] = logging.INFO,
                console: bool = True,
                file_mode: str = 'a',
                use_color: bool = True) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径（可选）
        level: 日志级别
        console: 是否输出到控制台
        file_mode: 文件打开模式 ('a' 追加, 'w' 覆盖)
        use_color: 是否使用彩色输出
        
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    
    # 如果已经配置过，直接返回
    if logger.handlers:
        return logger
    
    # 设置日志级别
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # 日志格式
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    
    # 控制台处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = ColoredFormatter(fmt, use_color=use_color)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode=file_mode, encoding='utf-8')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(fmt, datefmt=datefmt)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # 防止日志传播到父记录器
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """获取已存在的日志记录器"""
    return logging.getLogger(name)


def log_metrics(logger: logging.Logger, 
               metrics: Dict[str, Any], 
               epoch: Optional[int] = None,
               prefix: str = "",
               level: int = logging.INFO) -> None:
    """
    记录评估指标
    
    Args:
        logger: 日志记录器
        metrics: 指标字典
        epoch: 当前轮数（可选）
        prefix: 日志前缀
        level: 日志级别
    """
    if epoch is not None:
        message = f"{prefix}Epoch {epoch} - "
    else:
        message = f"{prefix}"
    
    # 格式化指标
    metric_strs = []
    for key, value in metrics.items():
        if isinstance(value, float):
            metric_strs.append(f"{key}: {value:.4f}")
        elif isinstance(value, dict):
            # 嵌套字典
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, float):
                    metric_strs.append(f"{key}/{sub_key}: {sub_value:.4f}")
                else:
                    metric_strs.append(f"{key}/{sub_key}: {sub_value}")
        else:
            metric_strs.append(f"{key}: {value}")
    
    message += " | ".join(metric_strs)
    logger.log(level, message)


def log_config(logger: logging.Logger, 
              config: Dict[str, Any],
              save_path: Optional[str] = None) -> None:
    """
    记录配置信息
    
    Args:
        logger: 日志记录器
        config: 配置字典
        save_path: 配置保存路径（可选）
    """
    logger.info("=" * 80)
    logger.info("配置信息:")
    logger.info("=" * 80)
    
    # 递归打印配置
    def log_dict(d: Dict[str, Any], indent: int = 0):
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info("  " * indent + f"{key}:")
                log_dict(value, indent + 1)
            else:
                logger.info("  " * indent + f"{key}: {value}")
    
    log_dict(config)
    logger.info("=" * 80)
    
    # 保存配置到文件
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, cls=TensorJSONEncoder, ensure_ascii=False)
        logger.info(f"配置已保存到: {save_path}")


def log_model_info(logger: logging.Logger, 
                  model: torch.nn.Module,
                  input_shape: Optional[tuple] = None) -> None:
    """
    记录模型信息
    
    Args:
        logger: 日志记录器
        model: PyTorch模型
        input_shape: 输入形状（可选）
    """
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("=" * 80)
    logger.info("模型信息:")
    logger.info("=" * 80)
    logger.info(f"模型类型: {model.__class__.__name__}")
    logger.info(f"总参数数: {total_params:,}")
    logger.info(f"可训练参数: {trainable_params:,}")
    logger.info(f"不可训练参数: {total_params - trainable_params:,}")
    
    if input_shape:
        logger.info(f"输入形状: {input_shape}")
    
    # 模型结构摘要
    logger.info("\n模型结构:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子模块
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                logger.info(f"  {name}: {module.__class__.__name__} ({params:,} params)")
    
    logger.info("=" * 80)


def create_experiment_logger(experiment_name: str,
                           log_dir: str = "./logs",
                           console_level: int = logging.INFO,
                           file_level: int = logging.DEBUG) -> Dict[str, logging.Logger]:
    """
    创建实验日志记录器集合
    
    Args:
        experiment_name: 实验名称
        log_dir: 日志目录
        console_level: 控制台日志级别
        file_level: 文件日志级别
        
    Returns:
        包含不同用途日志记录器的字典
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_log_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_log_dir, exist_ok=True)
    
    loggers = {
        'main': setup_logger(
            'main',
            os.path.join(exp_log_dir, 'main.log'),
            level=file_level,
            console=True
        ),
        'train': setup_logger(
            'train',
            os.path.join(exp_log_dir, 'train.log'),
            level=file_level,
            console=False
        ),
        'eval': setup_logger(
            'eval',
            os.path.join(exp_log_dir, 'eval.log'),
            level=file_level,
            console=False
        ),
        'metrics': setup_logger(
            'metrics',
            os.path.join(exp_log_dir, 'metrics.log'),
            level=logging.INFO,
            console=False
        )
    }
    
    # 设置控制台输出级别
    for logger in loggers.values():
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(console_level)
    
    return loggers


class LoggingContext:
    """日志上下文管理器"""
    
    def __init__(self, logger: logging.Logger, 
                level: int = logging.INFO,
                message: str = ""):
        self.logger = logger
        self.level = level
        self.message = message
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        if self.message:
            self.logger.log(self.level, f"{self.message} 开始...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            if self.message:
                self.logger.log(self.level, f"{self.message} 完成 (耗时: {elapsed_time:.2f}秒)")
        else:
            self.logger.error(f"{self.message} 失败: {exc_val}")
        
        return False  # 不抑制异常


def log_gpu_memory(logger: logging.Logger, prefix: str = "") -> None:
    """记录GPU内存使用情况"""
    if not torch.cuda.is_available():
        return
    
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        cached = torch.cuda.memory_reserved(i) / 1024**3  # GB
        
        logger.info(
            f"{prefix}GPU {i} 内存使用: "
            f"已分配={allocated:.2f}GB, 已缓存={cached:.2f}GB"
        )


def log_system_info(logger: logging.Logger) -> None:
    """记录系统信息"""
    import platform
    
    logger.info("=" * 80)
    logger.info("系统信息:")
    logger.info("=" * 80)
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"cuDNN版本: {torch.backends.cudnn.version()}")
        logger.info(f"GPU数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    logger.info(f"操作系统: {platform.system()} {platform.release()}")
    logger.info(f"CPU: {platform.processor()}")
    logger.info("=" * 80)


def setup_file_rotation(logger: logging.Logger, 
                       max_bytes: int = 10*1024*1024,  # 10MB
                       backup_count: int = 5) -> None:
    """设置日志文件轮转"""
    from logging.handlers import RotatingFileHandler
    
    # 移除现有的文件处理器
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
    
    # 添加轮转文件处理器
    for handler in logger.handlers:
        if hasattr(handler, 'baseFilename'):
            rotating_handler = RotatingFileHandler(
                handler.baseFilename,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
            rotating_handler.setFormatter(handler.formatter)
            rotating_handler.setLevel(handler.level)
            logger.addHandler(rotating_handler)


# 便捷函数
def debug(message: str, *args, **kwargs):
    """快速调试日志"""
    logger = get_logger('debug')
    if not logger.handlers:
        setup_logger('debug', level=logging.DEBUG)
    logger.debug(message, *args, **kwargs)


def info(message: str, *args, **kwargs):
    """快速信息日志"""
    logger = get_logger('info')
    if not logger.handlers:
        setup_logger('info', level=logging.INFO)
    logger.info(message, *args, **kwargs)


def warning(message: str, *args, **kwargs):
    """快速警告日志"""
    logger = get_logger('warning')
    if not logger.handlers:
        setup_logger('warning', level=logging.WARNING)
    logger.warning(message, *args, **kwargs)


def error(message: str, *args, **kwargs):
    """快速错误日志"""
    logger = get_logger('error')
    if not logger.handlers:
        setup_logger('error', level=logging.ERROR)
    logger.error(message, *args, **kwargs)