#!/usr/bin/env python3
"""
IO utilities for Power Grid Topology Reconstruction
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')


def load_csv_safe(filepath: str, **kwargs) -> pd.DataFrame:
    """安全地加载CSV文件"""
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件不存在: {filepath}")
        
        # 尝试不同的编码
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding, **kwargs)
                print(f"✅ 成功加载 {filepath} (编码: {encoding})")
                return df
            except UnicodeDecodeError:
                continue
        
        # 如果所有编码都失败，使用默认编码并忽略错误
        df = pd.read_csv(filepath, encoding='utf-8', errors='ignore', **kwargs)
        print(f"⚠️  使用容错模式加载 {filepath}")
        return df
        
    except Exception as e:
        raise IOError(f"加载CSV文件失败 {filepath}: {e}")


def save_csv_safe(df: pd.DataFrame, filepath: str, **kwargs) -> None:
    """安全地保存CSV文件"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存文件
        df.to_csv(filepath, index=False, encoding='utf-8', **kwargs)
        print(f"✅ 成功保存 {filepath}")
        
    except Exception as e:
        raise IOError(f"保存CSV文件失败 {filepath}: {e}")


def load_json(filepath: str) -> Dict[str, Any]:
    """加载JSON文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ 成功加载JSON {filepath}")
        return data
    except Exception as e:
        raise IOError(f"加载JSON文件失败 {filepath}: {e}")


def save_json(data: Dict[str, Any], filepath: str, indent: int = 2) -> None:
    """保存JSON文件"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 处理不可序列化的对象
        serializable_data = make_json_serializable(data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=indent, ensure_ascii=False)
        print(f"✅ 成功保存JSON {filepath}")
        
    except Exception as e:
        raise IOError(f"保存JSON文件失败 {filepath}: {e}")


def make_json_serializable(obj: Any) -> Any:
    """将对象转换为JSON可序列化格式"""
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return list(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif hasattr(obj, '__dict__'):
        # 对于有__dict__属性的对象，转换其属性
        return make_json_serializable(obj.__dict__)
    else:
        return obj


def load_yaml(filepath: str) -> Dict[str, Any]:
    """加载YAML文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        print(f"✅ 成功加载YAML {filepath}")
        return data
    except Exception as e:
        raise IOError(f"加载YAML文件失败 {filepath}: {e}")


def save_yaml(data: Dict[str, Any], filepath: str) -> None:
    """保存YAML文件"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        print(f"✅ 成功保存YAML {filepath}")
        
    except Exception as e:
        raise IOError(f"保存YAML文件失败 {filepath}: {e}")


def load_pickle(filepath: str) -> Any:
    """加载pickle文件"""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ 成功加载pickle {filepath}")
        return data
    except Exception as e:
        raise IOError(f"加载pickle文件失败 {filepath}: {e}")


def save_pickle(data: Any, filepath: str) -> None:
    """保存pickle文件"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"✅ 成功保存pickle {filepath}")
        
    except Exception as e:
        raise IOError(f"保存pickle文件失败 {filepath}: {e}")


def save_checkpoint(state_dict: Dict[str, Any], filepath: str) -> None:
    """保存PyTorch检查点"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save(state_dict, filepath)
        print(f"✅ 成功保存检查点 {filepath}")
        
    except Exception as e:
        raise IOError(f"保存检查点失败 {filepath}: {e}")


def load_checkpoint(filepath: str, device: torch.device) -> Dict[str, Any]:
    """加载PyTorch检查点"""
    try:
        checkpoint = torch.load(filepath, map_location=device)
        print(f"✅ 成功加载检查点 {filepath}")
        return checkpoint
    except Exception as e:
        raise IOError(f"加载检查点失败 {filepath}: {e}")


def create_directory_structure(base_path: str, structure: Dict[str, Any]) -> None:
    """创建目录结构"""
    def _create_dirs(current_path: str, struct: Dict[str, Any]):
        for name, content in struct.items():
            path = os.path.join(current_path, name)
            
            if isinstance(content, dict):
                os.makedirs(path, exist_ok=True)
                _create_dirs(path, content)
            else:
                # 如果是文件，创建父目录
                os.makedirs(os.path.dirname(path), exist_ok=True)
    
    os.makedirs(base_path, exist_ok=True)
    _create_dirs(base_path, structure)
    print(f"✅ 成功创建目录结构: {base_path}")


def get_file_info(filepath: str) -> Dict[str, Any]:
    """获取文件信息"""
    if not os.path.exists(filepath):
        return {'exists': False}
    
    stat = os.stat(filepath)
    return {
        'exists': True,
        'size_bytes': stat.st_size,
        'size_mb': stat.st_size / (1024 * 1024),
        'modified_time': stat.st_mtime,
        'is_file': os.path.isfile(filepath),
        'is_dir': os.path.isdir(filepath),
        'extension': os.path.splitext(filepath)[1]
    }


def find_files(directory: str, pattern: str = "*", recursive: bool = True) -> List[str]:
    """查找文件"""
    try:
        path = Path(directory)
        if recursive:
            files = list(path.rglob(pattern))
        else:
            files = list(path.glob(pattern))
        
        return [str(f) for f in files if f.is_file()]
    except Exception as e:
        print(f"查找文件失败: {e}")
        return []


def copy_file(src: str, dst: str, overwrite: bool = False) -> bool:
    """复制文件"""
    try:
        import shutil
        
        if os.path.exists(dst) and not overwrite:
            print(f"目标文件已存在，跳过复制: {dst}")
            return False
        
        # 确保目标目录存在
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        
        shutil.copy2(src, dst)
        print(f"✅ 成功复制文件: {src} -> {dst}")
        return True
        
    except Exception as e:
        print(f"复制文件失败: {e}")
        return False


def backup_file(filepath: str, backup_dir: Optional[str] = None) -> str:
    """备份文件"""
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件不存在: {filepath}")
        
        # 确定备份目录
        if backup_dir is None:
            backup_dir = os.path.join(os.path.dirname(filepath), 'backup')
        
        os.makedirs(backup_dir, exist_ok=True)
        
        # 生成备份文件名
        import time
        timestamp = int(time.time())
        filename = os.path.basename(filepath)
        name, ext = os.path.splitext(filename)
        backup_filename = f"{name}_{timestamp}{ext}"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        # 复制文件
        copy_file(filepath, backup_path, overwrite=True)
        
        return backup_path
        
    except Exception as e:
        raise IOError(f"备份文件失败: {e}")


def clean_directory(directory: str, pattern: str = "*", keep_recent: int = 5) -> int:
    """清理目录中的旧文件"""
    try:
        files = find_files(directory, pattern, recursive=False)
        
        if len(files) <= keep_recent:
            return 0
        
        # 按修改时间排序
        files_with_time = [(f, os.path.getmtime(f)) for f in files]
        files_with_time.sort(key=lambda x: x[1], reverse=True)  # 新文件在前
        
        # 删除旧文件
        deleted_count = 0
        for filepath, _ in files_with_time[keep_recent:]:
            try:
                os.remove(filepath)
                deleted_count += 1
                print(f"删除旧文件: {filepath}")
            except Exception as e:
                print(f"删除文件失败 {filepath}: {e}")
        
        return deleted_count
        
    except Exception as e:
        print(f"清理目录失败: {e}")
        return 0


def compress_file(filepath: str, output_path: Optional[str] = None) -> str:
    """压缩文件"""
    try:
        import gzip
        import shutil
        
        if output_path is None:
            output_path = filepath + '.gz'
        
        with open(filepath, 'rb') as f_in:
            with gzip.open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        print(f"✅ 成功压缩文件: {filepath} -> {output_path}")
        return output_path
        
    except Exception as e:
        raise IOError(f"压缩文件失败: {e}")


def extract_file(filepath: str, output_path: Optional[str] = None) -> str:
    """解压文件"""
    try:
        import gzip
        
        if not filepath.endswith('.gz'):
            raise ValueError("只支持.gz格式的压缩文件")
        
        if output_path is None:
            output_path = filepath[:-3]  # 移除.gz后缀
        
        with gzip.open(filepath, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                f_out.write(f_in.read())
        
        print(f"✅ 成功解压文件: {filepath} -> {output_path}")
        return output_path
        
    except Exception as e:
        raise IOError(f"解压文件失败: {e}")


def save_experiment_config(config: Dict[str, Any], output_dir: str, 
                         filename: str = "experiment_config.json") -> str:
    """保存实验配置"""
    filepath = os.path.join(output_dir, filename)
    
    # 添加时间戳和其他元信息
    import time
    config_with_meta = {
        'timestamp': time.time(),
        'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': config
    }
    
    save_json(config_with_meta, filepath)
    return filepath


def load_experiment_results(results_dir: str) -> Dict[str, Any]:
    """加载实验结果"""
    results = {}
    
    # 查找常见的结果文件
    result_files = {
        'config': 'experiment_config.json',
        'training_history': 'training_history.json',
        'evaluation_results': 'evaluation_results.json',
        'final_metrics': 'final_metrics.json'
    }
    
    for key, filename in result_files.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            try:
                results[key] = load_json(filepath)
            except Exception as e:
                print(f"加载 {filename} 失败: {e}")
    
    return results


def create_experiment_summary(results_dir: str, output_file: str = "experiment_summary.txt") -> str:
    """创建实验摘要"""
    try:
        results = load_experiment_results(results_dir)
        
        summary_path = os.path.join(results_dir, output_file)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("实验摘要报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 实验配置
            if 'config' in results:
                config = results['config']['config']
                f.write("实验配置:\n")
                f.write(f"  实验名称: {config.get('experiment', {}).get('name', 'N/A')}\n")
                f.write(f"  模型类型: {config.get('model', {}).get('model_type', 'standard')}\n")
                f.write(f"  训练轮数: {config.get('training', {}).get('epochs', 'N/A')}\n")
                f.write(f"  学习率: {config.get('training', {}).get('learning_rate', 'N/A')}\n\n")
            
            # 训练结果
            if 'training_history' in results:
                history = results['training_history']
                if 'train_losses' in history and history['train_losses']:
                    final_loss = history['train_losses'][-1].get('total', 0)
                    f.write(f"最终训练损失: {final_loss:.4f}\n")
                
                if 'metrics' in history and history['metrics']:
                    final_metrics = history['metrics'][-1]
                    topo_metrics = final_metrics.get('topology', {})
                    f.write(f"最终F1分数: {topo_metrics.get('f1_score', 0):.3f}\n")
                    f.write(f"最终精确率: {topo_metrics.get('precision', 0):.3f}\n")
                    f.write(f"最终召回率: {topo_metrics.get('recall', 0):.3f}\n\n")
            
            # 评估结果
            if 'evaluation_results' in results:
                eval_results = results['evaluation_results']
                summary_metrics = eval_results.get('summary_metrics', {})
                
                f.write("评估结果:\n")
                for metric_name, value in summary_metrics.items():
                    if '_mean' in metric_name:
                        f.write(f"  {metric_name}: {value:.4f}\n")
        
        print(f"✅ 成功创建实验摘要: {summary_path}")
        return summary_path
        
    except Exception as e:
        raise IOError(f"创建实验摘要失败: {e}")