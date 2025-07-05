cd /workspace/topo
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
cd power_grid_topology

# 现在运行
python scripts/train_model.py \
    --data_path ./data/local_coords_results_1_MV_urban__0_sw \
    --epochs 100 \
    --hidden_dim 128 \
    --unobservable_ratio 0.25