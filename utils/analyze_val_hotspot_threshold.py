import os
import argparse
import csv
import json
from typing import Dict, Iterable, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# =====================================================================
# SCI 顶刊全局视觉标准设置
# =====================================================================
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
rcParams['axes.linewidth'] = 1.2
rcParams['xtick.major.width'] = 1.2
rcParams['ytick.major.width'] = 1.2
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['axes.labelweight'] = 'bold'

def get_project_root() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, os.pardir))

def load_train_99p(data_dir: str) -> float:
    params_path = os.path.join(data_dir, "ais_norm_params.npy")
    if not os.path.exists(params_path):
        return 1.0
    params = np.load(params_path, allow_pickle=True).item()
    return float(params.get("train_99p", 1.0))

def load_validation_values(data_dir: str, mask_path: str | None = None) -> np.ndarray:
    val_path = os.path.join(data_dir, "ais_val.npy")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation AIS file not found: {val_path}")
    
    values = np.squeeze(np.load(val_path))
    
    if mask_path is None:
        mask_path = os.path.join(data_dir, "all_vars_train_mask_intersection.npy")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    mask = np.load(mask_path).astype(bool)
    valid_values = values[:, mask]
    valid_values = valid_values[np.isfinite(valid_values)]
    return valid_values.astype(np.float64)

def compute_otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    hist, bin_edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    probabilities = hist.astype(np.float64) / max(hist.sum(), 1)
    omega = np.cumsum(probabilities)
    mu = np.cumsum(probabilities * bin_centers)
    mu_total = mu[-1]
    denominator = omega * (1.0 - omega)
    between_var = np.zeros_like(bin_centers)
    valid = denominator > 0
    between_var[valid] = (mu_total * omega[valid] - mu[valid]) ** 2 / denominator[valid]
    return float(bin_centers[int(np.argmax(between_var[:-1]))])

def build_threshold_curve(values: np.ndarray, train_99p: float) -> List[Dict]:
    thresholds = np.linspace(0.0, 1.0, 401)
    sorted_values = np.sort(values)
    total = sorted_values.size
    rows = []
    for t in thresholds:
        left_count = int(np.searchsorted(sorted_values, t, side="left"))
        rows.append({
            "threshold_norm": float(t),
            "threshold_hours": float(t * train_99p),
            "hotspot_ratio": (total - left_count) / total
        })
    return rows

# =====================================================================
# 终极优化的 SCI 绘图函数
# =====================================================================
def save_threshold_plot(values: np.ndarray, output_path: str, train_99p: float, 
                        current_threshold: float, otsu_threshold: float):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sorted_values = np.sort(values)
    cdf = np.linspace(1.0 / len(sorted_values), 1.0, len(sorted_values))
    curve_data = build_threshold_curve(values, train_99p)
    t_vals = np.array([d["threshold_norm"] for d in curve_data])
    h_ratio = np.array([d["hotspot_ratio"] for d in curve_data])

    fig = plt.figure(figsize=(13, 5.5))
    gs = fig.add_gridspec(1, 2, wspace=0.35)

    # --- Panel (a): Distribution Analysis (双 Y 轴融合) ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1_cdf = ax1.twinx()
    
    c_hist, c_cdf = "#4C72B0", "#333333"
    ax1.hist(values, bins=160, range=(0.0, 1.0), density=True, color=c_hist, alpha=0.5, edgecolor='none')
    ax1.set_xlabel("Normalized AIS Value", fontsize=12)
    ax1.set_ylabel("Probability Density", fontsize=12, color=c_hist)
    ax1.tick_params(axis='y', labelcolor=c_hist, labelsize=10)
    
    ax1_cdf.plot(sorted_values, cdf, color=c_cdf, linewidth=2.0)
    ax1_cdf.set_ylabel("Cumulative Distribution (CDF)", fontsize=12, color=c_cdf)
    ax1_cdf.set_ylim(0, 1.1)  # 顶部留白
    ax1.set_xlim(0, 1.0)
    ax1.text(-0.14, 1.08, "(a)", transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top')

    # --- Panel (b): Threshold Sensitivity (多指标对比) ---
    ax2 = fig.add_subplot(gs[0, 1])
    c_curve = "#C44E52"
    
    # 绘制主要曲线并赋予 label
    line_h = ax2.plot(t_vals, h_ratio, color=c_curve, linewidth=2.5, label="Hotspot Area Ratio")[0]
    
    ax2.set_xlabel("Normalized Threshold", fontsize=12)
    ax2.set_ylabel("Ratio of Identified Hotspots", fontsize=12, color=c_curve)
    ax2.tick_params(axis='y', labelcolor=c_curve, labelsize=10)
    ax2.set_ylim(0, 1.1)  # 顶部留白
    ax2.set_xlim(0, 1.0)

    # 物理单位轴 (Top Axis)
    secax = ax2.secondary_xaxis('top', functions=(lambda x: x * train_99p, lambda x: x / max(train_99p, 1e-8)))
    secax.set_xlabel("Physical Threshold (hours/day)", fontsize=11, labelpad=10)
    secax.tick_params(labelsize=10)

    # 关键参考线
    line_v1 = ax2.axvline(current_threshold, color="#333333", linestyle="--", linewidth=1.8, 
                label=f"Selected Thresh ({current_threshold:.4f})")
    line_v2 = ax2.axvline(otsu_threshold, color="#8172B3", linestyle=":", linewidth=1.8, 
                label=f"Otsu's Method ({otsu_threshold:.4f})")

    # 整合图例：包含曲线和参考线，无边框
    ax2.legend(handles=[line_h, line_v1, line_v2], frameon=False, loc="upper right", fontsize=10)
    ax2.text(-0.14, 1.08, "(b)", transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top')

    # 添加极细辅助网格
    ax2.grid(axis='both', alpha=0.15, linestyle='--', linewidth=0.5)

    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close()

def main():
    # 路径配置（请根据实际情况微调）
    project_root = get_project_root()
    DATA_DIR = r"D:\VsCode Space\test\data\ST_FishNet_Features"
    OUT_DIR = os.path.join(project_root, "model_outcomes", "threshold_analysis")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("[INFO] Loading data and computing thresholds...")
    train_99p = load_train_99p(DATA_DIR)
    values = load_validation_values(DATA_DIR)
    otsu_thresh = compute_otsu_threshold(values)
    
    # 这里设置你最终决定的阈值 (0.3175)
    selected_thresh = 0.3175
    
    print(f"[INFO] Generating SCI Plot with threshold: {selected_thresh}")
    save_threshold_plot(
        values=values, 
        output_path=os.path.join(OUT_DIR, "SCI_Threshold_Analysis_Final.png"), 
        train_99p=train_99p, 
        current_threshold=selected_thresh, 
        otsu_threshold=otsu_thresh
    )
    
    print(f"✅ 任务完成！终极分析图已保存至: {os.path.join(OUT_DIR, 'SCI_Threshold_Analysis_Final.png')}")

if __name__ == "__main__":
    main()