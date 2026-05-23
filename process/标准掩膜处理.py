import xarray as xr
import numpy as np
import os
import json
import glob

# ================= 1. 路径配置 =================
DATA_DIR = r"D:\VsCode Space\2026GPR-FishNet\data\env"

# 优先使用你现在改好的新文件名
PREFERRED_FILES = [
    "thetao_2012_2024_0.083.nc",
]

OUTPUT_DIR = r"./data/ST_FishNet_Features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 2. 空间范围与目标分辨率 =================
MIN_LAT, MAX_LAT = 23.0, 28.0
MIN_LON, MAX_LON = 118.0, 126.0
RES = 0.1

# ================= 3. 模型输入目标尺寸 =================
TARGET_H, TARGET_W = 64, 96

# ================= 4. 时间配置 =================
TRAIN_LEN = 120   # 2012-01 ~ 2021-12
TOTAL_LEN = 156   # 2012-01 ~ 2024-12


def infer_coord_name(ds, candidates):
    for name in candidates:
        if name in ds.coords or name in ds.dims or name in ds.variables:
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


def find_input_nc(data_dir):
    """
    优先精确匹配新文件名；如果没有，再回退到 *thetao*.nc
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"数据目录不存在：{data_dir}")

    # 1) 先按精确文件名找
    for fname in PREFERRED_FILES:
        full_path = os.path.join(data_dir, fname)
        if os.path.isfile(full_path):
            print(f"✅ 使用精确匹配文件：{repr(full_path)}")
            return full_path

    # 2) 再模糊匹配
    matches = glob.glob(os.path.join(data_dir, "*thetao*.nc"))
    matches = [f for f in matches if os.path.isfile(f)]

    print("目录下匹配到的 thetao 文件：")
    for f in matches:
        print(" -", repr(f))

    if not matches:
        raise FileNotFoundError(
            f"在目录 {data_dir} 中未找到 thetao 的 nc 文件。\n"
            f"请确认文件名是否为：{PREFERRED_FILES[0]}"
        )

    # 优先选择名字里包含 0.083 的文件
    preferred = [f for f in matches if "0.083" in os.path.basename(f)]
    if preferred:
        print(f"✅ 使用优先候选文件：{repr(preferred[0])}")
        return preferred[0]

    print(f"✅ 使用第一个匹配文件：{repr(matches[0])}")
    return matches[0]


def build_land_mask():
    print("🚀 开始生成统一海陆掩膜...")

    input_nc = find_input_nc(DATA_DIR)

    print("实际读取文件：", repr(input_nc))
    print("文件是否存在：", os.path.exists(input_nc))

    if not os.path.isfile(input_nc):
        raise FileNotFoundError(f"找不到输入文件：{input_nc}")

    # 强制使用 h5netcdf
    with xr.open_dataset(input_nc, engine="h5netcdf") as ds:
        print("数据集打开成功")
        print(ds)

        if "thetao" not in ds.data_vars:
            raise ValueError(f"未找到变量 'thetao'，当前变量有：{list(ds.data_vars)}")

        lat_name = infer_coord_name(ds, ["latitude", "lat"])
        lon_name = infer_coord_name(ds, ["longitude", "lon"])
        time_name = infer_coord_name(ds, ["time"])
        depth_name = infer_coord_name(ds, ["depth", "deptho", "lev", "level"])

        if lat_name is None or lon_name is None or time_name is None:
            raise ValueError(
                f"无法识别坐标名。lat={lat_name}, lon={lon_name}, time={time_name}"
            )

        da = ds["thetao"]

        # 显式处理 depth
        if depth_name is not None and depth_name in da.dims:
            if ds[depth_name].size == 1:
                da = da.squeeze(depth_name, drop=True)
                print("✅ depth 只有一层，已 squeeze")
            else:
                da = da.sel({depth_name: 0.5}, method="nearest").squeeze(drop=True)
                print("✅ 已选择最接近 0.5 m 的表层数据")

        # 保证纬度/经度升序
        if da[lat_name].values[0] > da[lat_name].values[-1]:
            da = da.sortby(lat_name)
            print("🔄 纬度为降序，已改为升序")

        if da[lon_name].values[0] > da[lon_name].values[-1]:
            da = da.sortby(lon_name)
            print("🔄 经度为降序，已改为升序")

        # 粗裁剪
        da_cropped = da.sel(
            {
                lat_name: slice(MIN_LAT - 0.2, MAX_LAT + 0.2),
                lon_name: slice(MIN_LON - 0.2, MAX_LON + 0.2)
            }
        )

        # 插值到统一 0.1°
        target_lats = np.arange(MIN_LAT, MAX_LAT + RES / 2, RES)
        target_lons = np.arange(MIN_LON, MAX_LON + RES / 2, RES)

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

    # ===== 这里已经离开 with，文件已安全关闭 =====

    if data_matrix.ndim != 3:
        raise ValueError(f"期望 (time, lat, lon)，实际 shape = {data_matrix.shape}")

    if data_matrix.shape[0] != TOTAL_LEN:
        actual_start = str(time_axis[0]) if len(time_axis) > 0 else "未知"
        actual_end = str(time_axis[-1]) if len(time_axis) > 0 else "未知"
        raise ValueError(
            f"时间维度不是 {TOTAL_LEN}，而是 {data_matrix.shape[0]}。\n"
            f"预期时间范围：2012-01 到 2024-12。\n"
            f"当前实际时间范围：{actual_start} 到 {actual_end}"
        )

    if len(time_axis) != len(expected_time_axis):
        raise ValueError(
            f"时间轴长度与标准时间轴不一致："
            f"{len(time_axis)} vs {len(expected_time_axis)}"
        )

    if not np.array_equal(time_axis, expected_time_axis):
        mismatch_idx = np.where(time_axis != expected_time_axis)[0]
        first_bad = int(mismatch_idx[0]) if len(mismatch_idx) > 0 else -1
        raise ValueError(
            f"thetao 时间轴与标准月序列不一致。\n"
            f"第一个错误位置: idx={first_bad}\n"
            f"文件时间={time_axis[first_bad] if first_bad >= 0 else '未知'}\n"
            f"标准时间={expected_time_axis[first_bad] if first_bad >= 0 else '未知'}"
        )

    # 只基于训练期生成掩膜
    train_matrix = data_matrix[:TRAIN_LEN]

    # 训练期只要任一时刻有有效值，就认为是海洋
    valid_count = np.sum(~np.isnan(train_matrix), axis=0)
    land_mask_0p1deg = (valid_count > 0).astype(np.uint8)
    # 约定：1=ocean, 0=land

    np.save(os.path.join(OUTPUT_DIR, "land_mask_0.1deg.npy"), land_mask_0p1deg)

    # padding 到 64x96
    H, W = land_mask_0p1deg.shape
    if H > TARGET_H or W > TARGET_W:
        raise ValueError(
            f"当前 mask 尺寸 ({H}, {W}) 大于目标尺寸 ({TARGET_H}, {TARGET_W})"
        )

    pad_h = TARGET_H - H
    pad_w = TARGET_W - W

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    land_mask_padded = np.pad(
        land_mask_0p1deg,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0
    ).astype(np.uint8)

    np.save(os.path.join(OUTPUT_DIR, "land_mask_padded.npy"), land_mask_padded)
    np.save(os.path.join(OUTPUT_DIR, "thetao_train_mask_unified.npy"), land_mask_padded)

    # 保存坐标和时间轴
    np.save(os.path.join(OUTPUT_DIR, "target_lats.npy"), target_lats)
    np.save(os.path.join(OUTPUT_DIR, "target_lons.npy"), target_lons)
    np.save(os.path.join(OUTPUT_DIR, "thetao_mask_time_all.npy"), time_axis.astype("datetime64[M]"))

    metadata = {
        "source_file": input_nc,
        "source_variable": "thetao",
        "time_range_used_for_mask": "2012-01 to 2021-12",
        "expected_total_len": TOTAL_LEN,
        "expected_time_range": "2012-01 to 2024-12",
        "time_dtype_saved": "datetime64[M]",
        "mask_semantics": "1=ocean, 0=land_or_padding",
        "grid_shape_before_padding": [int(H), int(W)],
        "grid_shape_after_padding": [int(TARGET_H), int(TARGET_W)],
        "padding": {
            "pad_top": int(pad_top),
            "pad_bottom": int(pad_bottom),
            "pad_left": int(pad_left),
            "pad_right": int(pad_right)
        },
        "spatial_range": {
            "min_lat": MIN_LAT,
            "max_lat": MAX_LAT,
            "min_lon": MIN_LON,
            "max_lon": MAX_LON,
            "resolution": RES
        },
        "saved_files": {
            "land_mask_0.1deg": "land_mask_0.1deg.npy",
            "land_mask_padded": "land_mask_padded.npy",
            "thetao_train_mask_unified": "thetao_train_mask_unified.npy",
            "target_lats": "target_lats.npy",
            "target_lons": "target_lons.npy",
            "thetao_mask_time_all": "thetao_mask_time_all.npy"
        }
    }
    save_json(metadata, os.path.join(OUTPUT_DIR, "mask_metadata.json"))

    print("✅ 掩膜生成完成")
    print(f"land_mask_0.1deg.npy shape: {land_mask_0p1deg.shape}")
    print(f"land_mask_padded.npy shape: {land_mask_padded.shape}")
    print(f"thetao_mask_time_all.npy shape: {time_axis.shape}")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    build_land_mask()