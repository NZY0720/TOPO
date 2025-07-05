#!/usr/bin/env python3
"""
Power Grid Topology Reconstruction Package

Physics-informed Graph Transformer for Power Grid Topology Reconstruction
基于物理信息图变换器的电力网格拓扑重建
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__description__ = "Physics-informed Graph Transformer for Power Grid Topology Reconstruction"

# 导入主要组件
from .config.base_config import Config, get_default_config
from .data.data_loader import PowerGridDataLoader
from .models.graph_transformer import PhysicsGraphTransformer, create_model
from .training.trainer import PowerGridTrainer
from .evaluation.evaluator import ModelEvaluator

# 导入工具函数
from .utils.graph_utils import create_candidate_graph, ensure_radial_topology
from .utils.io_utils import load_json, save_json, load_checkpoint, save_checkpoint
from .utils.logging_utils import setup_logger

# 导入可视化函数
from .visualization.network_viz import plot_topology_comparison
from .visualization.train_viz import plot_training_curves

# 版本信息
__all__ = [
    # 版本信息
    '__version__',
    '__author__',
    '__email__',
    '__description__',
    
    # 核心组件
    'Config',
    'get_default_config',
    'PowerGridDataLoader',
    'PhysicsGraphTransformer',
    'create_model',
    'PowerGridTrainer',
    'ModelEvaluator',
    
    # 工具函数
    'create_candidate_graph',
    'ensure_radial_topology',
    'load_json',
    'save_json',
    'load_checkpoint',
    'save_checkpoint',
    'setup_logger',
    
    # 可视化函数
    'plot_topology_comparison',
    'plot_training_curves',
]

# 包级别配置
import warnings
import logging

# 配置警告过滤
warnings.filterwarnings('ignore', category=UserWarning, module='torch_geometric')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

# 设置默认日志级别
logging.getLogger(__name__).setLevel(logging.INFO)

def get_package_info():
    """获取包信息"""
    return {
        'name': 'power-grid-topology',
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'description': __description__,
        'license': 'MIT'
    }

def check_dependencies():
    """检查依赖是否满足"""
    import sys
    
    dependencies = {
        'torch': '1.12.0',
        'torch_geometric': '2.1.0',
        'pandas': '1.4.0',
        'numpy': '1.21.0',
        'matplotlib': '3.5.0',
        'networkx': '2.8.0',
        'scikit-learn': '1.1.0'
    }
    
    missing_deps = []
    version_issues = []
    
    for package, min_version in dependencies.items():
        try:
            if package == 'torch_geometric':
                import torch_geometric
                installed_version = torch_geometric.__version__
            else:
                module = __import__(package)
                installed_version = getattr(module, '__version__', 'unknown')
            
            # 简单版本检查（实际项目中可能需要更复杂的版本比较）
            if installed_version == 'unknown':
                version_issues.append(f"{package}: version unknown")
            
        except ImportError:
            missing_deps.append(package)
    
    if missing_deps:
        warnings.warn(
            f"Missing dependencies: {', '.join(missing_deps)}. "
            f"Install with: pip install {' '.join(missing_deps)}"
        )
    
    if version_issues:
        warnings.warn(f"Version issues: {', '.join(version_issues)}")
    
    return len(missing_deps) == 0 and len(version_issues) == 0

# 在导入时检查依赖
try:
    _dependencies_ok = check_dependencies()
    if not _dependencies_ok:
        warnings.warn(
            "Some dependencies are missing or have version issues. "
            "The package may not work correctly."
        )
except Exception as e:
    warnings.warn(f"Error checking dependencies: {e}")

# 显示包信息
def _show_welcome_message():
    """显示欢迎信息"""
    import sys
    if hasattr(sys, 'ps1'):  # 交互式环境
        print(f"🚀 Power Grid Topology Reconstruction v{__version__}")
        print(f"📖 Documentation: https://github.com/yourusername/power-grid-topology")
        print(f"🐛 Issues: https://github.com/yourusername/power-grid-topology/issues")

# 在交互式环境中显示欢迎信息
try:
    _show_welcome_message()
except:
    pass