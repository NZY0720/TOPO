#!/usr/bin/env python3
"""
SimBench 时序潮流计算
包含结果保存和可视化
"""
import simbench as sb
import pandapower as pp
from pandapower.timeseries import run_timeseries, OutputWriter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 1. 加载网络
print("正在加载SimBench网络...")
net = sb.get_simbench_net("1-complete_data-mixed-all-2-sw")
print(f"网络加载完成: {len(net.bus)}个母线, {len(net.line)}条线路, {len(net.load)}个负荷")

# 2. 获取时间序列数据
print("正在获取时间序列profiles...")
profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
print(f"可用的profile类型: {list(profiles.keys())}")

# 3. 应用控制器
print("正在创建时序控制器...")
sb.apply_const_controllers(net, profiles)
print(f"创建了 {len(net.controller)} 个控制器")

# 4. 设置时序仿真参数
timesteps = range(96)  # 24小时，15分钟间隔
print(f"时间步长: {len(timesteps)} (24小时，15分钟间隔)")

# 5. 设置结果保存
output_dir = Path("time_series_results")
output_dir.mkdir(exist_ok=True)

# 选择要监测的母线和线路
n_buses = len(net.bus)
n_lines = len(net.line)
monitored_buses = [0, n_buses//4, n_buses//2, 3*n_buses//4, n_buses-1]
monitored_lines = [0, 1, min(5, n_lines-1), min(10, n_lines-1)]

# 清理索引确保有效
monitored_buses = [idx for idx in monitored_buses if idx < n_buses][:5]
monitored_lines = [idx for idx in monitored_lines if idx < n_lines][:4]

print(f"监测 {len(monitored_buses)} 个母线和 {len(monitored_lines)} 条线路")

# 6. 创建输出记录器
ow = OutputWriter(net, timesteps, output_path=output_dir, output_file_type=".json")

# 记录关键变量
ow.log_variable('res_bus', 'vm_pu', index=monitored_buses)
ow.log_variable('res_line', 'loading_percent', index=monitored_lines)
ow.log_variable('res_load', 'p_mw', eval_function=np.sum, eval_name="total_load")
ow.log_variable('res_sgen', 'p_mw', eval_function=np.sum, eval_name="total_generation")

# 7. 运行时序潮流
print("正在运行时序潮流计算...")
try:
    run_timeseries(net, time_steps=timesteps, verbose=False, 
                   output_writer=ow, continue_on_divergence=True)
    print("✅ 时序潮流计算完成")
except Exception as e:
    print(f"❌ 时序潮流计算失败: {e}")
    exit(1)

# 8. 读取并处理结果
print("正在读取结果...")
try:
    # 读取结果文件
    bus_voltages = pd.read_json(output_dir / "res_bus" / "vm_pu.json")
    line_loadings = pd.read_json(output_dir / "res_line" / "loading_percent.json")
    total_load = pd.read_json(output_dir / "results" / "total_load.json")
    total_generation = pd.read_json(output_dir / "results" / "total_generation.json")
    
    # 转换为DataFrame便于处理
    total_load_df = pd.DataFrame(total_load.values, columns=['total_load_mw'])
    total_generation_df = pd.DataFrame(total_generation.values, columns=['total_generation_mw'])
    
    print("✅ 结果读取成功")
except Exception as e:
    print(f"❌ 结果读取失败: {e}")
    exit(1)

# 9. 保存处理后的结果到CSV
print("正在保存结果到CSV...")
results_summary = pd.DataFrame({
    'timestep': range(len(total_load_df)),
    'hour': [t/4 for t in range(len(total_load_df))],  # 15分钟间隔转为小时
    'total_load_mw': total_load_df['total_load_mw'],
    'total_generation_mw': total_generation_df['total_generation_mw'],
    'load_generation_diff': total_generation_df['total_generation_mw'] - total_load_df['total_load_mw']
})

# 添加电压统计
if not bus_voltages.empty:
    results_summary['min_voltage_pu'] = bus_voltages.min(axis=1)
    results_summary['max_voltage_pu'] = bus_voltages.max(axis=1)
    results_summary['avg_voltage_pu'] = bus_voltages.mean(axis=1)

# 添加线路负载统计
if not line_loadings.empty:
    results_summary['max_line_loading_pct'] = line_loadings.max(axis=1)
    results_summary['avg_line_loading_pct'] = line_loadings.mean(axis=1)

# 保存汇总结果
results_summary.to_csv(output_dir / "time_series_summary.csv", index=False)
print(f"✅ 结果已保存到: {output_dir / 'time_series_summary.csv'}")

# 10. 创建可视化
print("正在创建可视化...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 子图1: 负荷和发电对比
ax1 = axes[0, 0]
hours = results_summary['hour']
ax1.plot(hours, results_summary['total_load_mw'], 'b-', label='负荷', linewidth=2)
ax1.plot(hours, results_summary['total_generation_mw'], 'g-', label='发电', linewidth=2)
ax1.set_title('负荷与发电时序曲线')
ax1.set_xlabel('时间 (小时)')
ax1.set_ylabel('功率 (MW)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 子图2: 电压变化
ax2 = axes[0, 1]
if 'min_voltage_pu' in results_summary.columns:
    ax2.plot(hours, results_summary['min_voltage_pu'], 'r-', label='最低电压', linewidth=2)
    ax2.plot(hours, results_summary['max_voltage_pu'], 'b-', label='最高电压', linewidth=2)
    ax2.plot(hours, results_summary['avg_voltage_pu'], 'g-', label='平均电压', linewidth=2)
    ax2.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='下限')
    ax2.axhline(y=1.05, color='r', linestyle='--', alpha=0.7, label='上限')
    ax2.set_title('母线电压变化')
    ax2.set_xlabel('时间 (小时)')
    ax2.set_ylabel('电压 (标幺值)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
else:
    ax2.text(0.5, 0.5, '电压数据不可用', ha='center', va='center')
    ax2.set_title('母线电压变化')

# 子图3: 线路负载率
ax3 = axes[1, 0]
if 'max_line_loading_pct' in results_summary.columns:
    ax3.plot(hours, results_summary['max_line_loading_pct'], 'r-', label='最高负载率', linewidth=2)
    ax3.plot(hours, results_summary['avg_line_loading_pct'], 'b-', label='平均负载率', linewidth=2)
    ax3.axhline(y=100, color='r', linestyle='--', alpha=0.7, label='过载线')
    ax3.set_title('线路负载率变化')
    ax3.set_xlabel('时间 (小时)')
    ax3.set_ylabel('负载率 (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
else:
    ax3.text(0.5, 0.5, '线路负载数据不可用', ha='center', va='center')
    ax3.set_title('线路负载率变化')

# 子图4: 功率平衡
ax4 = axes[1, 1]
ax4.plot(hours, results_summary['load_generation_diff'], 'purple', linewidth=2)
ax4.axhline(y=0, color='k', linestyle='-', alpha=0.5)
ax4.set_title('功率平衡 (发电-负荷)')
ax4.set_xlabel('时间 (小时)')
ax4.set_ylabel('功率差 (MW)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "time_series_plots.png", dpi=150, bbox_inches='tight')
plt.show()

# 11. 输出统计摘要
print("\n" + "="*50)
print("时序潮流分析结果摘要")
print("="*50)
print(f"仿真时长: 24小时 ({len(timesteps)}个时间步)")
print(f"负荷范围: {results_summary['total_load_mw'].min():.1f} - {results_summary['total_load_mw'].max():.1f} MW")
print(f"发电范围: {results_summary['total_generation_mw'].min():.1f} - {results_summary['total_generation_mw'].max():.1f} MW")

if 'min_voltage_pu' in results_summary.columns:
    print(f"电压范围: {results_summary['min_voltage_pu'].min():.3f} - {results_summary['max_voltage_pu'].max():.3f} 标幺值")

if 'max_line_loading_pct' in results_summary.columns:
    print(f"最大线路负载率: {results_summary['max_line_loading_pct'].max():.1f}%")

print(f"\n结果文件保存在: {output_dir}")
print("="*50)
