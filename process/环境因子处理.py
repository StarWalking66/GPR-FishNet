import xarray as xr
import numpy as np
import os
import json

# ================= 1. 路径配置 =================
DATA_DIR = r"D:\VsCode Space\2026GPR-FishNet\data\env"
OUTPUT_DIR = r"./data/ST_FishNet_Features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 2. 研究区域与统一分辨率 =================
MIN_LAT, MAX_LAT = 23.0, 28.0
MIN_LON, MAX_LON = 118.0, 126.0
RES = 0.1

# ================= 3. 模型输入目标尺寸 =================
TARGET_H, TARGET_W = 64, 96

# ================= 4. 时间切分 =================
TRAIN_LEN = 120  # 2012-2021
VAL_LEN = 24     # 2022-2023
TEST_LEN = 12    # 2024
TOTAL_LEN = TRAIN_LEN + VAL_LEN + TEST_LEN  # 156

# ================= 5. 统一 mask 文件（由独立掩膜脚本生成） =================
COMMON_MASK_PATH = os.path.join(OUTPUT_DIR, "thetao_train_mask_unified.npy")
LAND_MASK_0P1_PATH = os.path.join(OUTPUT_DIR, "land_mask_0.1deg.npy")
LAND_MASK_PADDED_PATH = os.path.join(OUTPUT_DIR, "land_mask_padded.npy")
TARGET_LATS_PATH = os.path.join(OUTPUT_DIR, "target_lats.npy")
TARGET_LONS_PATH = os.path.join(OUTPUT_DIR, "target_lons.npy")

# ================= 6. 变量配置 =================
VAR_CONFIGS = {
    "thetao": {
        "filename": "thetao_2012_2024_0.083.nc",
        "var_candidates": ["thetao"],
        "use_log": False,
        "surface_depth": 0.5
    },
    "uo": {
        "filename": "uo_2012_2024_0.083.nc",
        "var_candidates": ["uo"],
        "use_log": False,
        "surface_depth": 0.5
    },
    "vo": {
        "filename": "vo_2012_2024_0.083.nc",
        "var_candidates": ["vo"],
        "use_log": False,
        "surface_depth": 0.5
    },
    "so": {
        "filename": "so_2012_2024_0.083.nc",
        "var_candidates": ["so"],
        "use_log": False,
        "surface_depth": 0.5
    },
    "zos": {
        "filename": "zos_2012_2024_0.083.nc",
        "var_candidates": ["zos"],
        "use_log": False,
        "surface_depth": None
    },
    "chl": {
        "filename": "chl_2012_2024_0.25.nc",
        "var_candidates": ["chl", "ch1", "chs", "CHS", "chlorophyll", "chla", "chlor_a"],
        "use_log": True,
        "surface_depth": 0.51
    },
    "o2": {
        "filename": "o2_2012_2024_0.25.nc",
        "var_candidates": ["o2", "O2", "dissolved_oxygen", "oxygen"],
        "use_log": False,
        "surface_depth": 0.51
    }
}

CHL_EPS = 1e-6


def infer_coord_name(ds, candidates):
    for name in candidates:
        if name in ds.coords or name in ds.dims or name in ds.variables:
            return name
    return None


def infer_var_name(ds, candidates):
    for name in candidates:
        if name in ds.data_vars:
            return name
    return None


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def build_expected_time_axis():
    """
    统一的月尺度标准时间轴：2012-01 ~ 2024-12
    返回 datetime64[M]
    """
    return np.arange(
        np.datetime64("2012-01"),
        np.datetime64("2025-01"),
        np.timedelta64(1, "M")
    ).astype("datetime64[M]")


def normalize_time_axis_to_month(time_axis):
    """
    把 xarray 读出来的时间轴统一转成 datetime64[M]，
    避免 ns / us / pandas Timestamp 格式不一致导致假不相等。
    """
    return np.array(time_axis).astype("datetime64[M]")


def safe_open_dataset(input_nc):
    """
    统一使用 h5netcdf，避免默认 netcdf4 在 Windows/中文路径下不稳定。
    """
    print(f"📂 读取文件: {repr(input_nc)}")
    print(f"📌 文件是否存在: {os.path.exists(input_nc)}")

    if not os.path.exists(input_nc):
        raise FileNotFoundError(f"找不到输入文件：{input_nc}")

    return xr.open_dataset(input_nc, engine="h5netcdf")


def ensure_common_mask_files():
    missing = []
    for p in [
        COMMON_MASK_PATH,
        LAND_MASK_0P1_PATH,
        LAND_MASK_PADDED_PATH,
        TARGET_LATS_PATH,
        TARGET_LONS_PATH
    ]:
        if not os.path.exists(p):
            missing.append(p)

    if missing:
        raise FileNotFoundError(
            "缺少统一掩膜相关文件，请先运行独立掩膜生成脚本。缺少文件如下：\n"
            + "\n".join(missing)
        )

    common_mask = np.load(COMMON_MASK_PATH).astype(np.uint8)
    land_mask_0p1 = np.load(LAND_MASK_0P1_PATH).astype(np.uint8)
    land_mask_padded = np.load(LAND_MASK_PADDED_PATH).astype(np.uint8)
    target_lats = np.load(TARGET_LATS_PATH)
    target_lons = np.load(TARGET_LONS_PATH)

    if common_mask.shape != (TARGET_H, TARGET_W):
        raise ValueError(
            f"统一训练 mask 尺寸错误：{common_mask.shape}，期望 {(TARGET_H, TARGET_W)}"
        )

    return common_mask, land_mask_0p1, land_mask_padded, target_lats, target_lons


def compute_train_stats_on_ocean(train_data_padded, common_mask):
    """
    train_data_padded: (T, H, W)
    common_mask: (H, W), 1=ocean, 0=land_or_padding
    """
    ocean_mask_3d = np.broadcast_to(common_mask[None, :, :], train_data_padded.shape)
    ocean_values = train_data_padded[ocean_mask_3d == 1]
    ocean_values = ocean_values[~np.isnan(ocean_values)]

    if ocean_values.size == 0:
        raise ValueError("训练集海洋区域没有有效值，无法计算均值和标准差")

    mean_val = float(np.mean(ocean_values))
    std_val = float(np.std(ocean_values))

    if std_val < 1e-8:
        std_val = 1e-8
        print("⚠️ 标准差过小，已自动置为 1e-8")

    return mean_val, std_val


def preprocess_single_variable(var_key, cfg, common_mask, target_lats, target_lons):
    print(f"\n{'=' * 72}")
    print(f"🚀 开始处理环境因子: {var_key}")
    print(f"{'=' * 72}")

    input_nc = os.path.join(DATA_DIR, cfg["filename"])
    if not os.path.exists(input_nc):
        raise FileNotFoundError(f"[{var_key}] 找不到输入文件：{input_nc}")

    with safe_open_dataset(input_nc) as ds:
        print(f"📌 原始变量: {list(ds.data_vars)}")
        print(f"📌 原始维度: {dict(ds.sizes)}")

        # ===== 识别变量与坐标 =====
        var_name = infer_var_name(ds, cfg["var_candidates"])
        if var_name is None:
            raise ValueError(
                f"[{var_key}] 未找到候选变量 {cfg['var_candidates']}，当前变量有：{list(ds.data_vars)}"
            )

        lat_name = infer_coord_name(ds, ["latitude", "lat"])
        lon_name = infer_coord_name(ds, ["longitude", "lon"])
        time_name = infer_coord_name(ds, ["time"])
        depth_name = infer_coord_name(ds, ["depth", "deptho", "lev", "level"])

        if lat_name is None or lon_name is None or time_name is None:
            raise ValueError(
                f"[{var_key}] 无法识别坐标名。lat={lat_name}, lon={lon_name}, time={time_name}"
            )

        print(f"🧭 坐标识别 -> lat: {lat_name}, lon: {lon_name}, time: {time_name}, depth: {depth_name}")
        print(f"🌊 变量识别 -> {var_name}")

        da = ds[var_name]

        # ===== 显式处理 depth =====
        if cfg["surface_depth"] is not None and depth_name is not None and depth_name in da.dims:
            depth_values = ds[depth_name].values
            print(f"🌊 检测到 depth 维度，深度值示例: {depth_values[:5] if len(depth_values) > 5 else depth_values}")

            if ds[depth_name].size == 1:
                da = da.squeeze(depth_name, drop=True)
                print("✅ depth 只有一层，已 squeeze")
            else:
                da = da.sel({depth_name: cfg["surface_depth"]}, method="nearest")
                da = da.squeeze(drop=True)
                print(f"✅ 已选择最接近 {cfg['surface_depth']} m 的表层数据")
        elif depth_name is not None and depth_name in da.dims and ds[depth_name].size == 1:
            da = da.squeeze(depth_name, drop=True)
            print("✅ depth 只有一层，已 squeeze")

        # ===== 处理纬度/经度顺序 =====
        lat_values = da[lat_name].values
        if lat_values[0] > lat_values[-1]:
            da = da.sortby(lat_name)
            print("🔄 纬度为降序，已改为升序")

        lon_values = da[lon_name].values
        if lon_values[0] > lon_values[-1]:
            da = da.sortby(lon_name)
            print("🔄 经度为降序，已改为升序")

        # ===== 粗裁剪 =====
        print("✂️ 正在粗裁剪...")
        da_cropped = da.sel(
            {
                lat_name: slice(MIN_LAT - 0.2, MAX_LAT + 0.2),
                lon_name: slice(MIN_LON - 0.2, MAX_LON + 0.2)
            }
        )

        # ===== 插值到统一 0.1° 网格 =====
        print(f"🌊 正在插值到统一 0.1° 网格: ({len(target_lats)}, {len(target_lons)})")
        da_interp = da_cropped.interp(
            {
                lat_name: target_lats,
                lon_name: target_lons
            },
            method="linear"
        ).load()

        data_matrix = da_interp.values
        time_axis = normalize_time_axis_to_month(da_interp[time_name].values)
        expected_time_axis = build_expected_time_axis()

    # ===== 离开 with 后文件已关闭，下面只处理 numpy =====
    if data_matrix.ndim != 3:
        raise ValueError(
            f"[{var_key}] 插值后维度异常，期望 (time, lat, lon)，实际 shape = {data_matrix.shape}"
        )

    print(f"📦 插值后张量形状: {data_matrix.shape}")

    # ===== 检查时间长度 =====
    if data_matrix.shape[0] != TOTAL_LEN:
        actual_start = str(time_axis[0]) if len(time_axis) > 0 else "未知"
        actual_end = str(time_axis[-1]) if len(time_axis) > 0 else "未知"
        raise ValueError(
            f"[{var_key}] 时间维度长度不是 {TOTAL_LEN}，而是 {data_matrix.shape[0]}。\n"
            f"预期时间范围：2012-01 到 2024-12（共156个月）。\n"
            f"当前文件实际时间范围：{actual_start} 到 {actual_end}"
        )

    if len(time_axis) != len(expected_time_axis):
        raise ValueError(
            f"[{var_key}] 时间轴长度与标准时间轴不一致："
            f"{len(time_axis)} vs {len(expected_time_axis)}"
        )

    if not np.array_equal(time_axis, expected_time_axis):
        mismatch_idx = np.where(time_axis != expected_time_axis)[0]
        first_bad = int(mismatch_idx[0]) if len(mismatch_idx) > 0 else -1
        raise ValueError(
            f"[{var_key}] 时间轴与标准月序列不一致。\n"
            f"第一个错误位置: idx={first_bad}\n"
            f"文件时间={time_axis[first_bad] if first_bad >= 0 else '未知'}\n"
            f"标准时间={expected_time_axis[first_bad] if first_bad >= 0 else '未知'}"
        )

    # ===== 数据预变换 =====
    if cfg["use_log"]:
        print("🧪 正在执行 chl 专属 log 变换...")
        data_prepared = np.full_like(data_matrix, np.nan, dtype=np.float32)
        positive_mask = data_matrix > 0
        data_prepared[positive_mask] = np.log(data_matrix[positive_mask] + CHL_EPS)
    else:
        print("🧪 正在执行普通变量预处理（不做 log）...")
        data_prepared = data_matrix.astype(np.float32)

    # ===== Padding =====
    T, H, W = data_prepared.shape
    if H > TARGET_H or W > TARGET_W:
        raise ValueError(
            f"[{var_key}] 当前空间尺寸 ({H}, {W}) 大于目标尺寸 ({TARGET_H}, {TARGET_W})，请检查区域设置。"
        )

    pad_h = TARGET_H - H
    pad_w = TARGET_W - W

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    print(f"📦 正在 Padding: H +{pad_h}, W +{pad_w}")
    data_padded = np.pad(
        data_prepared,
        pad_width=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=np.nan
    )

    print(f"📏 Padding 后形状: {data_padded.shape}")

    # ===== 仅用训练集海洋区域计算 mean/std =====
    print("🧮 正在用训练集海洋区域计算 Z-score 参数...")
    train_prepared = data_padded[:TRAIN_LEN]
    mean_val, std_val = compute_train_stats_on_ocean(train_prepared, common_mask)

    print(f"   - 训练集均值 (Z-score): {mean_val:.6f}")
    print(f"   - 训练集标准差 (Z-score): {std_val:.6f}")

    # ===== 第一步：Z-score 标准化 =====
    print("📊 正在执行第一步：Z-score 标准化...")
    data_normalized = (data_padded - mean_val) / std_val
    ocean_mask_3d = np.broadcast_to(common_mask[None, :, :], data_normalized.shape)
    valid_mask_3d = np.isfinite(data_padded) & (ocean_mask_3d == 1)
    data_normalized = np.where(valid_mask_3d, data_normalized, np.nan)

    # ===== 第二步：Min-Max 归一化到 [0, 1] =====
    print("📊 正在执行第二步：Min-Max 归一化到 [0, 1]...")
    train_normalized = data_normalized[:TRAIN_LEN]
    train_valid_mask_3d = valid_mask_3d[:TRAIN_LEN]
    train_ocean_values = train_normalized[train_valid_mask_3d]

    if train_ocean_values.size == 0:
        raise ValueError(f"[{var_key}] no valid ocean pixels in train set for Min/Max normalization")

    min_val = float(np.min(train_ocean_values))
    max_val = float(np.max(train_ocean_values))

    if max_val - min_val < 1e-8:
        max_val = min_val + 1e-8
        print(f"⚠️ [{var_key}] Min/Max range too small, adjusted to 1e-8")

    print(f"   - 训练集最小值 ([0,1]归一化): {min_val:.6f}")
    print(f"   - 训练集最大值 ([0,1]归一化): {max_val:.6f}")

    # 保存归一化参数
    np.savez(
        os.path.join(OUTPUT_DIR, f"{var_key}_norm_params.npz"),
        mean=mean_val,
        std=std_val,
        min_01=min_val,
        max_01=max_val
    )
    np.save(
        os.path.join(OUTPUT_DIR, f"{var_key}_norm_params.npy"),
        {"mean": float(mean_val), "std": float(std_val), "min_01": min_val, "max_01": max_val},
        allow_pickle=True
    )

    # 执行 [0,1] 归一化
    data_normalized = (data_normalized - min_val) / (max_val - min_val + 1e-8)
    data_normalized = np.clip(data_normalized, 0.0, 1.0)
    data_normalized = np.where(valid_mask_3d, data_normalized, np.nan)
    data_normalized = np.nan_to_num(data_normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # ===== 当前变量自身的 padded mask（只做记录）=====
    train_matrix_raw = data_matrix[:TRAIN_LEN]
    if cfg["use_log"]:
        local_mask_0p1 = np.any(train_matrix_raw > 0, axis=0).astype(np.uint8)
    else:
        valid_count = np.sum(~np.isnan(train_matrix_raw), axis=0)
        local_mask_0p1 = (valid_count > 0).astype(np.uint8)

    local_mask_padded = np.pad(
        local_mask_0p1,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0
    ).astype(np.uint8)

    np.save(os.path.join(OUTPUT_DIR, f"{var_key}_land_mask_0.1deg.npy"), local_mask_0p1)
    np.save(os.path.join(OUTPUT_DIR, f"{var_key}_land_mask_padded.npy"), local_mask_padded)

    # ===== 记录训练期有效像素并与 common_mask 取交集 =====
    valid_mask_train_padded = np.any(train_valid_mask_3d, axis=0).astype(np.uint8)
    strict_train_mask = (common_mask.astype(np.uint8) * valid_mask_train_padded).astype(np.uint8)

    np.save(os.path.join(OUTPUT_DIR, f"{var_key}_valid_mask_train.npy"), valid_mask_train_padded)
    np.save(os.path.join(OUTPUT_DIR, f"{var_key}_train_mask_unified.npy"), strict_train_mask)
    print(f"💾 已保存统一训练 mask: {var_key}_train_mask_unified.npy")

    # ===== 增加通道维度 =====
    data_final = np.expand_dims(data_normalized, axis=1).astype(np.float32)
    print(f"🗜️ 最终张量形状: {data_final.shape}")

    # ===== 严格时间切分 =====
    train_data = data_final[:TRAIN_LEN]
    val_data = data_final[TRAIN_LEN:TRAIN_LEN + VAL_LEN]
    test_data = data_final[TRAIN_LEN + VAL_LEN:TOTAL_LEN]

    train_time = time_axis[:TRAIN_LEN]
    val_time = time_axis[TRAIN_LEN:TRAIN_LEN + VAL_LEN]
    test_time = time_axis[TRAIN_LEN + VAL_LEN:TOTAL_LEN]

    # ===== 保存张量 =====
    np.save(os.path.join(OUTPUT_DIR, f"{var_key}_train.npy"), train_data)
    np.save(os.path.join(OUTPUT_DIR, f"{var_key}_val.npy"), val_data)
    np.save(os.path.join(OUTPUT_DIR, f"{var_key}_test.npy"), test_data)

    # ===== 保存时间（统一保存为 month 精度）=====
    np.save(os.path.join(OUTPUT_DIR, f"{var_key}_time_all.npy"), time_axis.astype("datetime64[M]"))
    np.save(os.path.join(OUTPUT_DIR, f"{var_key}_time_train.npy"), train_time.astype("datetime64[M]"))
    np.save(os.path.join(OUTPUT_DIR, f"{var_key}_time_val.npy"), val_time.astype("datetime64[M]"))
    np.save(os.path.join(OUTPUT_DIR, f"{var_key}_time_test.npy"), test_time.astype("datetime64[M]"))

    metadata = {
        "input_file": input_nc,
        "variable_key": var_key,
        "variable_name_in_nc": var_name,
        "lat_name": lat_name,
        "lon_name": lon_name,
        "time_name": time_name,
        "depth_name": depth_name,
        "surface_depth_target": cfg["surface_depth"],
        "use_log": cfg["use_log"],
        "spatial_range": {
            "min_lat": MIN_LAT,
            "max_lat": MAX_LAT,
            "min_lon": MIN_LON,
            "max_lon": MAX_LON,
            "resolution": RES
        },
        "target_grid_shape_before_padding": [int(len(target_lats)), int(len(target_lons))],
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
        "normalization": {
            "method": "z-score -> min-max_[0,1]",
            "mean_from": "train_only_ocean_pixels",
            "std_from": "train_only_ocean_pixels",
            "min_01_from": "train_only_ocean_pixels_after_zscore",
            "max_01_from": "train_only_ocean_pixels_after_zscore",
            "nan_fill_value": 0.0
        },
        "mask": {
            "local_land_mask_0.1deg": f"{var_key}_land_mask_0.1deg.npy",
            "local_land_mask_padded": f"{var_key}_land_mask_padded.npy",
            "valid_mask_train_padded": f"{var_key}_valid_mask_train.npy",
            "train_mask_unified": f"{var_key}_train_mask_unified.npy",
            "mask_semantics": "1=valid_ocean, 0=land_or_padding_or_missing"
        }
    }

    save_json(metadata, os.path.join(OUTPUT_DIR, f"{var_key}_metadata.json"))

    print(f"✅ {var_key} 处理完成")
    print(f"   Train: {train_data.shape}")
    print(f"   Val:   {val_data.shape}")
    print(f"   Test:  {test_data.shape}")


def main():
    common_mask, _, _, target_lats, target_lons = ensure_common_mask_files()

    global_config = {
        "data_dir": DATA_DIR,
        "output_dir": OUTPUT_DIR,
        "spatial_range": {
            "min_lat": MIN_LAT,
            "max_lat": MAX_LAT,
            "min_lon": MIN_LON,
            "max_lon": MAX_LON,
            "resolution": RES
        },
        "target_shape": {
            "height": TARGET_H,
            "width": TARGET_W
        },
        "time_split": {
            "train_len": TRAIN_LEN,
            "val_len": VAL_LEN,
            "test_len": TEST_LEN,
            "total_len": TOTAL_LEN
        },
        "variables": VAR_CONFIGS,
        "common_mask_source": COMMON_MASK_PATH,
        "expected_time_axis_start": "2012-01",
        "expected_time_axis_end": "2024-12",
        "expected_time_axis_dtype": "datetime64[M]"
    }
    save_json(global_config, os.path.join(OUTPUT_DIR, "global_preprocess_config.json"))

    intersection_mask = np.ones((TARGET_H, TARGET_W), dtype=np.uint8)

    for var_key, cfg in VAR_CONFIGS.items():
        preprocess_single_variable(var_key, cfg, common_mask, target_lats, target_lons)
        var_mask_path = os.path.join(OUTPUT_DIR, f"{var_key}_train_mask_unified.npy")
        var_mask = np.load(var_mask_path).astype(np.uint8)
        intersection_mask = (intersection_mask * (var_mask > 0).astype(np.uint8)).astype(np.uint8)

    intersection_path = os.path.join(OUTPUT_DIR, "all_vars_train_mask_intersection.npy")
    np.save(intersection_path, intersection_mask)
    print(f"💾 已保存多变量交集训练掩膜: {intersection_path}")

    # ================= 时间轴最终一致性校验 =================
    print("🕒 正在进行最终时间轴一致性校验...")
    expected_time_axis = build_expected_time_axis()

    for var_key in VAR_CONFIGS.keys():
        var_time = np.load(os.path.join(OUTPUT_DIR, f"{var_key}_time_all.npy")).astype("datetime64[M]")
        if not np.array_equal(var_time, expected_time_axis):
            raise ValueError(f"[FINAL CHECK] {var_key}_time_all.npy 与标准时间轴不一致")

    ais_time_path = os.path.join(OUTPUT_DIR, "ais_time_all.npy")
    if os.path.exists(ais_time_path):
        ais_time = np.load(ais_time_path).astype("datetime64[M]")

        if not np.array_equal(ais_time, expected_time_axis):
            raise ValueError("[FINAL CHECK] ais_time_all.npy 与标准时间轴不一致")

        for var_key in VAR_CONFIGS.keys():
            var_time = np.load(os.path.join(OUTPUT_DIR, f"{var_key}_time_all.npy")).astype("datetime64[M]")
            if not np.array_equal(var_time, ais_time):
                raise ValueError(f"[FINAL CHECK] {var_key} 与 AIS 时间轴不一致")

        print("✅ 所有环境因子与 AIS 时间轴完全一致")
    else:
        print("ℹ️ 尚未检测到 ais_time_all.npy，已完成环境变量内部时间轴一致性校验。")

    print(f"\n🎉 全部环境因子处理完成，输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()