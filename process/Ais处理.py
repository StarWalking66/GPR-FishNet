import os
import glob
import json
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

# ================= 1. 路径配置 =================
AIS_DATA_DIR = r"D:\VsCode Space\2026GPR-FishNet\data\ais"
OUTPUT_DIR = r"./data/ST_FishNet_Features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 2. 研究区域与统一分辨率 =================
MIN_LAT, MAX_LAT = 23.0, 28.0
MIN_LON, MAX_LON = 118.0, 126.0
RES = 0.1

# ================= 3. 模型输入目标尺寸 =================
TARGET_H, TARGET_W = 64, 96

# ================= 4. 时间切分 =================
TRAIN_LEN = 120   # 2012-2021
VAL_LEN = 24      # 2022-2023
TEST_LEN = 12     # 2024
TOTAL_LEN = TRAIN_LEN + VAL_LEN + TEST_LEN  # 156

# ================= 5. AIS 字段配置 =================
LAT_COL = "cell_ll_lat"
LON_COL = "cell_ll_lon"
VAL_COL = "fishing_hours"
FILE_TEMPLATE = "fleet-monthly-csvs-10-v3-{year}-{month:02d}-01.csv"

# ================= 6. 掩膜与统一网格文件（由 thetao 掩膜脚本生成） =================
COMMON_MASK_PATH = os.path.join(OUTPUT_DIR, "thetao_train_mask_unified.npy")
LAND_MASK_0P1_PATH = os.path.join(OUTPUT_DIR, "land_mask_0.1deg.npy")
TARGET_LATS_PATH = os.path.join(OUTPUT_DIR, "target_lats.npy")
TARGET_LONS_PATH = os.path.join(OUTPUT_DIR, "target_lons.npy")

# ================= 7. AIS 参数 =================
SMOOTH_SIGMA = 1.0          
CLIP_MIN = 0.0
CLIP_MAX = 1.0
STRICT_MISSING_MONTHS = True
COORD_SEMANTICS = "lower_left_corner"   
APPARENT_EFFORT_LABEL = True            


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def build_expected_time_axis():
    """
    统一标准月时间轴：2012-01 ~ 2024-12
    保存为 datetime64[M]
    """
    return np.arange(
        np.datetime64("2012-01"),
        np.datetime64("2025-01"),
        np.timedelta64(1, "M")
    ).astype("datetime64[M]")


def days_in_month(year, month):
    return pd.Period(f"{year}-{month:02d}", freq="M").days_in_month


def ensure_common_files():
    missing = []
    for p in [COMMON_MASK_PATH, LAND_MASK_0P1_PATH, TARGET_LATS_PATH, TARGET_LONS_PATH]:
        if not os.path.exists(p):
            missing.append(p)

    if missing:
        raise FileNotFoundError(
            "缺少 thetao 先生成的统一掩膜/网格文件，请先运行 thetao 掩膜脚本。缺少：\n"
            + "\n".join(missing)
        )

    common_mask = np.load(COMMON_MASK_PATH).astype(np.uint8)   # (64,96)
    land_mask_0p1 = np.load(LAND_MASK_0P1_PATH).astype(np.uint8)  # (H,W)
    target_lats = np.load(TARGET_LATS_PATH).astype(np.float64)
    target_lons = np.load(TARGET_LONS_PATH).astype(np.float64)

    if common_mask.shape != (TARGET_H, TARGET_W):
        raise ValueError(
            f"统一训练掩膜尺寸错误：{common_mask.shape}，期望 {(TARGET_H, TARGET_W)}"
        )

    if target_lats.ndim != 1 or target_lons.ndim != 1:
        raise ValueError("target_lats / target_lons 必须是一维数组")

    if len(target_lats) != land_mask_0p1.shape[0] or len(target_lons) != land_mask_0p1.shape[1]:
        raise ValueError(
            f"land_mask_0p1 与 target_lats/target_lons 尺寸不一致: "
            f"mask={land_mask_0p1.shape}, lats={len(target_lats)}, lons={len(target_lons)}"
        )

    if not np.all(np.diff(target_lats) > 0):
        raise ValueError("target_lats 必须严格递增")

    if not np.all(np.diff(target_lons) > 0):
        raise ValueError("target_lons 必须严格递增")

    return common_mask, land_mask_0p1, target_lats, target_lons


def find_month_file(year, month):
    exact_path = os.path.join(AIS_DATA_DIR, FILE_TEMPLATE.format(year=year, month=month))
    if os.path.exists(exact_path):
        return exact_path

    pattern = os.path.join(AIS_DATA_DIR, f"*{year}-{month:02d}-01*.csv")
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches[0]
    return None


def masked_gaussian_smooth(grid_2d, valid_mask, sigma):
    """
    Smooth only within valid ocean pixels so coastal hotspots are not
    artificially attenuated by land zeros.
    """
    valid_mask = valid_mask.astype(np.float32)

    weighted_values = gaussian_filter(
        grid_2d.astype(np.float32) * valid_mask,
        sigma=sigma,
        mode="constant",
        cval=0.0
    )
    weighted_mask = gaussian_filter(
        valid_mask,
        sigma=sigma,
        mode="constant",
        cval=0.0
    )

    smoothed = np.zeros_like(grid_2d, dtype=np.float32)
    np.divide(
        weighted_values,
        np.clip(weighted_mask, 1e-8, None),
        out=smoothed,
        where=weighted_mask > 1e-8
    )
    smoothed[valid_mask == 0] = 0.0
    return smoothed.astype(np.float32)


def _coords_to_grid_index(ll_values, grid_start, grid_len, res):
    """
    将 GFW 提供的左下角坐标映射到统一 0.1° 栅格索引。
    这里按“左下角单元索引语义”处理，而不是点中心语义。
    """
    idx = np.rint((ll_values - grid_start) / res).astype(np.int64)
    return idx


def aggregate_month_to_grid(file_path, target_lats, target_lons, land_mask_0p1, year, month):
    H, W = land_mask_0p1.shape
    grid_total = np.zeros((H, W), dtype=np.float32)

    if file_path is None:
        return grid_total, grid_total.copy()

    df = pd.read_csv(file_path, usecols=[LAT_COL, LON_COL, VAL_COL])
    df = df.dropna(subset=[LAT_COL, LON_COL, VAL_COL]).copy()

    # 只保留正捕鱼时长记录
    df = df[df[VAL_COL] > 0].copy()
    if len(df) == 0:
        return grid_total, grid_total.copy()

    # GFW 文档说明：cell_ll_lat / cell_ll_lon 对应网格左下角
    # 为了和你现有 target_lats/target_lons 数值网格保持一致，
    # 这里按“左下角单元索引”直接落格，而不是强行改成中心点坐标。
    lat_min_supported = target_lats[0]
    lon_min_supported = target_lons[0]
    lat_max_supported = target_lats[-1]
    lon_max_supported = target_lons[-1]

    df = df[
        (df[LAT_COL] >= lat_min_supported - 1e-8) &
        (df[LAT_COL] <= lat_max_supported + 1e-8) &
        (df[LON_COL] >= lon_min_supported - 1e-8) &
        (df[LON_COL] <= lon_max_supported + 1e-8)
    ].copy()

    if len(df) == 0:
        return grid_total, grid_total.copy()

    
    df_agg = df.groupby([LAT_COL, LON_COL], as_index=False)[VAL_COL].sum()

    row_idx = _coords_to_grid_index(df_agg[LAT_COL].values, target_lats[0], H, RES)
    col_idx = _coords_to_grid_index(df_agg[LON_COL].values, target_lons[0], W, RES)

    
    valid = (
        (row_idx >= 0) & (row_idx < H) &
        (col_idx >= 0) & (col_idx < W)
    )

    row_idx = row_idx[valid]
    col_idx = col_idx[valid]
    vals = df_agg[VAL_COL].values[valid].astype(np.float32)

    np.add.at(grid_total, (row_idx, col_idx), vals)

    
    grid_total[land_mask_0p1 == 0] = 0.0

    dim = days_in_month(year, month)
    grid_daymean = grid_total / float(dim)

    return grid_total.astype(np.float32), grid_daymean.astype(np.float32)


def final_time_alignment_check_with_env(output_dir, time_axis):
    """
    如果环境时间文件已经存在，则做最终一致性检查。
    """
    env_vars = ["thetao", "uo", "vo", "so", "zos", "chl", "o2"]
    time_axis = np.array(time_axis).astype("datetime64[M]")

    existing_env_time_files = [
        os.path.join(output_dir, f"{var}_time_all.npy")
        for var in env_vars
        if os.path.exists(os.path.join(output_dir, f"{var}_time_all.npy"))
    ]

    if len(existing_env_time_files) == 0:
        print("ℹ️ 尚未检测到环境变量 *_time_all.npy，跳过 AIS-ENV 时间轴一致性检查。")
        return

    for var in env_vars:
        var_time_path = os.path.join(output_dir, f"{var}_time_all.npy")
        if not os.path.exists(var_time_path):
            continue

        var_time = np.load(var_time_path).astype("datetime64[M]")
        if len(var_time) != len(time_axis):
            raise ValueError(
                f"[FINAL CHECK] {var}_time_all.npy 与 AIS 时间轴长度不一致: "
                f"{len(var_time)} vs {len(time_axis)}"
            )

        if not np.array_equal(var_time, time_axis):
            mismatch_idx = np.where(var_time != time_axis)[0]
            first_bad = int(mismatch_idx[0]) if len(mismatch_idx) > 0 else -1
            raise ValueError(
                f"[FINAL CHECK] {var} 与 AIS 时间轴不一致。\n"
                f"第一个错误位置 idx={first_bad}\n"
                f"{var}_time={var_time[first_bad] if first_bad >= 0 else '未知'}\n"
                f"ais_time={time_axis[first_bad] if first_bad >= 0 else '未知'}"
            )

    print("✅ AIS 与所有已存在的环境变量时间轴完全一致")


def preprocess_ais():
    print("🚀 开始处理 AIS 分支数据（论文对齐稳定版）...")

    common_mask, land_mask_0p1, target_lats, target_lons = ensure_common_files()
    H, W = len(target_lats), len(target_lons)

    print(f"📐 目标 0.1° 网格: H={H}, W={W}")
    print(f"📐 模型目标尺寸: H={TARGET_H}, W={TARGET_W}")
    print(f"🧭 坐标语义: {COORD_SEMANTICS}")

    time_axis = build_expected_time_axis()
    if len(time_axis) != TOTAL_LEN:
        raise ValueError(f"时间轴长度异常：{len(time_axis)}，预期 {TOTAL_LEN}")

    # ================= 1. 构建 monthly total 与 day-mean grid =================
    ais_monthly_total_grid = np.zeros((TOTAL_LEN, H, W), dtype=np.float32)
    ais_monthly_daymean_grid = np.zeros((TOTAL_LEN, H, W), dtype=np.float32)
    missing_months = []

    for t, dt in enumerate(pd.period_range("2012-01", "2024-12", freq="M")):
        year = dt.year
        month = dt.month
        file_path = find_month_file(year, month)

        if file_path is None:
            missing_tag = f"{year}-{month:02d}"
            missing_months.append(missing_tag)
            if STRICT_MISSING_MONTHS:
                raise FileNotFoundError(
                    f"AIS month file missing: {missing_tag}. "
                    "Preprocessing stops to avoid confusing missing data with real zero fishing effort."
                )
            print(f"⚠️ {missing_tag} 未找到文件，该月保持全 0")
            continue

        print(f"⏳ 正在处理 {year}-{month:02d} -> {os.path.basename(file_path)}")
        month_total, month_daymean = aggregate_month_to_grid(
            file_path=file_path,
            target_lats=target_lats,
            target_lons=target_lons,
            land_mask_0p1=land_mask_0p1,
            year=year,
            month=month
        )
        ais_monthly_total_grid[t] = month_total
        ais_monthly_daymean_grid[t] = month_daymean

    print("✅ 所有月份已完成 grid 聚合")
    np.save(os.path.join(OUTPUT_DIR, "ais_monthly_total_grid.npy"), ais_monthly_total_grid)
    np.save(os.path.join(OUTPUT_DIR, "ais_monthly_daymean_grid.npy"), ais_monthly_daymean_grid)

    # ================= 2. 对 day-mean grid 做空间平滑 =================
    print("🌊 正在执行 Gaussian smoothing ...")
    ais_daymean_smoothed = np.zeros_like(ais_monthly_daymean_grid, dtype=np.float32)
    ocean_mask = land_mask_0p1 == 1

    for t in range(TOTAL_LEN):
        sm = masked_gaussian_smooth(ais_monthly_daymean_grid[t], ocean_mask, SMOOTH_SIGMA)
        ais_daymean_smoothed[t] = sm.astype(np.float32)

    np.save(os.path.join(OUTPUT_DIR, "ais_daymean_smoothed_grid.npy"), ais_daymean_smoothed)

    # ================= 3. 99分位数截断 + [0,1] 归一化 =================
    print("📈 正在执行 99分位数截断 + [0,1] 归一化 ...")

    ais_daymean_processed = ais_daymean_smoothed.astype(np.float32)

    # 仅用训练集统计 99 分位数（防泄露）
    train_processed = ais_daymean_processed[:TRAIN_LEN]
    ocean_pixels = land_mask_0p1 == 1
    train_99p = np.quantile(train_processed[:, ocean_pixels], 0.99)

    if train_99p < 1e-8:
        train_99p = 1e-8
        print("⚠️ 训练集99分位数过小，已自动置为 1e-8")

    print(f"   - 训练集99分位数 (hours/day): {train_99p:.6f}")

    # 截断极端值
    ais_daymean_processed = np.clip(ais_daymean_processed, 0.0, train_99p)

    # 归一化到 [0,1]
    ais_norm = ais_daymean_processed / (train_99p + 1e-8)
    ais_norm = np.clip(ais_norm, CLIP_MIN, CLIP_MAX).astype(np.float32)

    np.savez(
        os.path.join(OUTPUT_DIR, "ais_norm_params.npz"),
        train_99p=float(train_99p),
        smoothing_sigma=float(SMOOTH_SIGMA),
        target_unit="hours/day",
        transform="gaussian_smoothing -> 99p_clip -> train_only_[0,1]",
        coord_semantics=COORD_SEMANTICS,
        apparent_effort_label=APPARENT_EFFORT_LABEL
    )

    np.save(
        os.path.join(OUTPUT_DIR, "ais_norm_params.npy"),
        {
            "train_99p": float(train_99p),
            "smoothing_sigma": float(SMOOTH_SIGMA),
            "target_unit": "hours/day",
            "transform": "gaussian_smoothing -> 99p_clip -> train_only_[0,1]",
            "coord_semantics": COORD_SEMANTICS,
            "apparent_effort_label": APPARENT_EFFORT_LABEL
        },
        allow_pickle=True
    )

    # ================= 4. Padding 到 64×96 =================
    pad_h = TARGET_H - H
    pad_w = TARGET_W - W

    if pad_h < 0 or pad_w < 0:
        raise ValueError(
            f"当前 AIS 网格尺寸 ({H}, {W}) 大于目标尺寸 ({TARGET_H}, {TARGET_W})"
        )

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    print(f"📦 正在 Padding: H +{pad_h}, W +{pad_w}")

    ais_norm_padded = np.pad(
        ais_norm,
        pad_width=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0.0
    ).astype(np.float32)
    ais_norm_padded[:, common_mask == 0] = 0.0

    ais_daymean_grid_padded = np.pad(
        ais_monthly_daymean_grid,
        pad_width=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0.0
    ).astype(np.float32)

    ais_daymean_processed_padded = np.pad(
        ais_daymean_processed,
        pad_width=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0.0
    ).astype(np.float32)

    np.save(os.path.join(OUTPUT_DIR, "ais_daymean_grid_padded.npy"), ais_daymean_grid_padded)
    np.save(os.path.join(OUTPUT_DIR, "ais_daymean_processed_grid_padded.npy"), ais_daymean_processed_padded)
    np.save(os.path.join(OUTPUT_DIR, "ais_full_norm_padded.npy"), ais_norm_padded)

    # ================= 5. 增加通道维度 =================
    final_ais_tensor = np.expand_dims(ais_norm_padded, axis=1).astype(np.float32)
    print(f"🗜️ 最终张量形状: {final_ais_tensor.shape}")

    # ================= 6. 严格时间切分 =================
    train_data = final_ais_tensor[:TRAIN_LEN]
    val_data = final_ais_tensor[TRAIN_LEN:TRAIN_LEN + VAL_LEN]
    test_data = final_ais_tensor[TRAIN_LEN + VAL_LEN:TOTAL_LEN]

    train_time = time_axis[:TRAIN_LEN]
    val_time = time_axis[TRAIN_LEN:TRAIN_LEN + VAL_LEN]
    test_time = time_axis[TRAIN_LEN + VAL_LEN:TOTAL_LEN]

    # ================= 7. 保存训练数据 =================
    np.save(os.path.join(OUTPUT_DIR, "ais_train.npy"), train_data)
    np.save(os.path.join(OUTPUT_DIR, "ais_val.npy"), val_data)
    np.save(os.path.join(OUTPUT_DIR, "ais_test.npy"), test_data)

    np.save(os.path.join(OUTPUT_DIR, "ais_time_all.npy"), time_axis.astype("datetime64[M]"))
    np.save(os.path.join(OUTPUT_DIR, "ais_time_train.npy"), train_time.astype("datetime64[M]"))
    np.save(os.path.join(OUTPUT_DIR, "ais_time_val.npy"), val_time.astype("datetime64[M]"))
    np.save(os.path.join(OUTPUT_DIR, "ais_time_test.npy"), test_time.astype("datetime64[M]"))

    np.save(os.path.join(OUTPUT_DIR, "ais_train_mask_unified.npy"), common_mask)

    # ================= 8. 与环境时间轴最终一致性检查 =================
    final_time_alignment_check_with_env(OUTPUT_DIR, time_axis)

    # ================= 9. metadata =================
    metadata = {
        "data_source_dir": AIS_DATA_DIR,
        "file_template": FILE_TEMPLATE,
        "field_names": {
            "lat": LAT_COL,
            "lon": LON_COL,
            "value": VAL_COL
        },
        "label_semantics": {
            "type": "apparent_fishing_effort",
            "description": "AIS-derived apparent fishing effort (hours), not direct fish abundance truth"
        },
        "coordinate_semantics": {
            "source_coordinate_type": COORD_SEMANTICS,
            "description": "cell_ll_lat / cell_ll_lon are interpreted as lower-left corner coordinates of 0.1-degree grid cells"
        },
        "spatial_range": {
            "min_lat": MIN_LAT,
            "max_lat": MAX_LAT,
            "min_lon": MIN_LON,
            "max_lon": MAX_LON,
            "resolution": RES
        },
        "target_grid_shape_before_padding": [int(H), int(W)],
        "target_grid_shape_after_padding": [int(TARGET_H), int(TARGET_W)],
        "padding": {
            "pad_top": int(pad_top),
            "pad_bottom": int(pad_bottom),
            "pad_left": int(pad_left),
            "pad_right": int(pad_right)
        },
        "time_split": {
            "train_len": TRAIN_LEN,
            "val_len": VAL_LEN,
            "test_len": TEST_LEN,
            "expected_total_len": TOTAL_LEN,
            "expected_range": "2012-01 to 2024-12",
            "time_dtype_saved": "datetime64[M]"
        },
        "target_definition": {
            "monthly_total_grid": "hours/month per grid",
            "monthly_daymean_grid": "hours/day per grid",
            "training_target": "hours/day after smoothing, 99p clip and train-only [0,1] normalization"
        },
        "processing": {
            "aggregation_logic": "sum fishing_hours within each 0.1-degree lower-left indexed grid cell",
            "smoothing": {
                "method": "mask_aware_gaussian_filter",
                "sigma": float(SMOOTH_SIGMA)
            },
            "log_transform": "None",
            "normalization": "clip by train-only 99th percentile then scale to [0,1]"
        },
        "missing_month_policy": "raise_error" if STRICT_MISSING_MONTHS else "fill_zero_and_record",
        "mask": {
            "common_mask_source": COMMON_MASK_PATH,
            "mask_semantics": "1=ocean, 0=land_or_padding"
        },
        "saved_files": {
            "monthly_total": "ais_monthly_total_grid.npy",
            "monthly_daymean": "ais_monthly_daymean_grid.npy",
            "smoothed_daymean": "ais_daymean_smoothed_grid.npy",
            "processed_daymean_padded": "ais_daymean_processed_grid_padded.npy",
            "full_norm_padded": "ais_full_norm_padded.npy",
            "train": "ais_train.npy",
            "val": "ais_val.npy",
            "test": "ais_test.npy"
        },
        "missing_months": missing_months
    }

    save_json(metadata, os.path.join(OUTPUT_DIR, "ais_metadata.json"))

    print("\n✅ AIS 分支预处理完成！")
    print(f"   Train: {train_data.shape}")
    print(f"   Val:   {val_data.shape}")
    print(f"   Test:  {test_data.shape}")
    print(f"📁 输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    preprocess_ais()