import os
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class STFishNetUltimateDataset(Dataset):
    """
    通用版数据集

    1) 基线 SwinLSTM
       return_future_env=False 时返回：
           inputs  : (seq_len, 8, H, W)   -> 7个环境因子 + 1个AIS
           targets : (pred_len, 1, H, W)  -> 未来AIS

    2) 主模型 / 双分支模型
       return_future_env=True 时返回：
           inputs     : (seq_len, 8, H, W)
           future_env : (pred_len, 7, H, W)
           targets    : (pred_len, 1, H, W)

    依赖的预处理输出文件命名：
        环境因子:
            {var}_train.npy
            {var}_val.npy
            {var}_test.npy
            {var}_time_all.npy

        AIS:
            ais_train.npy
            ais_val.npy
            ais_test.npy
            ais_time_all.npy
    """

    def __init__(
        self,
        data_dir: str,
        env_vars: List[str],
        split: str = "train",
        seq_len: int = 12,
        pred_len: int = 1,
        return_future_env: bool = False,
    ):
        super().__init__()

        if split not in ["train", "val", "test"]:
            raise ValueError(f"split 必须是 train/val/test，当前收到: {split}")

        if not isinstance(seq_len, int) or seq_len <= 0:
            raise ValueError(f"seq_len 必须是正整数，当前收到: {seq_len}")

        if not isinstance(pred_len, int) or pred_len <= 0:
            raise ValueError(f"pred_len 必须是正整数，当前收到: {pred_len}")

        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"data_dir 不存在: {data_dir}")

        if not isinstance(env_vars, (list, tuple)) or len(env_vars) == 0:
            raise ValueError("env_vars 必须是非空列表，例如 ['thetao','uo','vo','so','zos','chl','o2']")

        self.data_dir = data_dir
        self.env_vars = list(env_vars)
        self.split = split
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.return_future_env = return_future_env

        print(f"[{split.upper()}] Loading STFishNetUltimateDataset...")
        print(f"   data_dir={data_dir}")
        print(f"   env_vars={self.env_vars}")
        print(f"   seq_len={seq_len}, pred_len={pred_len}, return_future_env={return_future_env}")

        # ================= 0. 时间轴一致性检查 =================
        self._validate_time_alignment()

        # ================= 1. 读取环境因子 =================
        env_data_list = []
        base_hw = None

        for var in self.env_vars:
            data = self._load_split_array(var, split)  # 期望 (T, 1, H, W)
            self._validate_single_var_array(data, var_name=var)

            if base_hw is None:
                base_hw = data.shape[2:]
            elif data.shape[2:] != base_hw:
                raise ValueError(
                    f"环境因子空间尺寸不一致: {var} 的 shape={data.shape}, "
                    f"而前面变量空间尺寸={base_hw}"
                )

            env_data_list.append(data.astype(np.float32, copy=False))

        
        self.env_tensor = np.concatenate(env_data_list, axis=1).astype(np.float32, copy=False)

        
        self.ais_tensor = self._load_split_array("ais", split).astype(np.float32, copy=False)
        self._validate_ais_array(self.ais_tensor)

        
        if self.env_tensor.shape[0] != self.ais_tensor.shape[0]:
            raise ValueError(
                f"时间长度未对齐: env={self.env_tensor.shape[0]}, ais={self.ais_tensor.shape[0]}"
            )

        if self.env_tensor.shape[2:] != self.ais_tensor.shape[2:]:
            raise ValueError(
                f"空间尺寸未对齐: env={self.env_tensor.shape[2:]}, ais={self.ais_tensor.shape[2:]}"
            )

        if self.env_tensor.shape[1] != len(self.env_vars):
            raise ValueError(
                f"环境因子通道数与 env_vars 数量不一致: "
                f"env_tensor.shape[1]={self.env_tensor.shape[1]}, len(env_vars)={len(self.env_vars)}"
            )

        self.total_frames = int(self.env_tensor.shape[0])
        self.total_samples = self.total_frames - self.seq_len - self.pred_len + 1

        if self.total_samples <= 0:
            raise ValueError(
                f"样本数 <= 0，请检查 seq_len={self.seq_len}, pred_len={self.pred_len}, "
                f"total_frames={self.total_frames}"
            )

        print(f"[OK] {split} dataset loaded")
        print(f"   env_tensor shape: {self.env_tensor.shape}")
        print(f"   ais_tensor shape: {self.ais_tensor.shape}")
        print(f"   total_frames    : {self.total_frames}")
        print(f"   total_samples   : {self.total_samples}")

    
    def _load_npy(self, filename: str) -> np.ndarray:
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到文件: {path}")

        arr = np.load(path)

        if not isinstance(arr, np.ndarray):
            raise ValueError(f"读取到的不是 numpy.ndarray: {path}")

        return arr

    def _load_time_all(self, var_name: str) -> np.ndarray:
        path = os.path.join(self.data_dir, f"{var_name}_time_all.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到时间轴文件: {path}")

        time_arr = np.load(path)
        return np.array(time_arr).astype("datetime64[M]")

    def _load_split_array(self, var_name: str, split: str) -> np.ndarray:
        """
        统一读取 train/val/test，并在 val/test 前拼接历史 seq_len 帧，
        保证滑动窗口连续。

        train: 直接返回 train
        val  : train最后seq_len帧 + val
        test : val最后seq_len帧 + test
        """
        train_arr = self._load_npy(f"{var_name}_train.npy")

        if split == "train":
            return train_arr

        val_arr = self._load_npy(f"{var_name}_val.npy")

        if split == "val":
            if len(train_arr) < self.seq_len:
                raise ValueError(
                    f"{var_name}_train.npy 长度不足 seq_len={self.seq_len}，"
                    f"当前长度={len(train_arr)}"
                )
            return np.concatenate([train_arr[-self.seq_len:], val_arr], axis=0)

        # split == "test"
        test_arr = self._load_npy(f"{var_name}_test.npy")

        if len(val_arr) < self.seq_len:
            raise ValueError(
                f"{var_name}_val.npy 长度不足 seq_len={self.seq_len}，"
                f"当前长度={len(val_arr)}"
            )

        return np.concatenate([val_arr[-self.seq_len:], test_arr], axis=0)

    
    def _validate_single_var_array(self, arr: np.ndarray, var_name: str) -> None:
        if arr.ndim != 4:
            raise ValueError(
                f"{var_name} 数据维度应为 (T, 1, H, W)，当前: {arr.shape}"
            )
        if arr.shape[1] != 1:
            raise ValueError(
                f"{var_name} 通道数应为 1，当前: {arr.shape[1]}，shape={arr.shape}"
            )

    def _validate_ais_array(self, arr: np.ndarray) -> None:
        if arr.ndim != 4:
            raise ValueError(f"AIS 数据维度应为 (T, 1, H, W)，当前: {arr.shape}")
        if arr.shape[1] != 1:
            raise ValueError(f"AIS 通道数应为 1，当前: {arr.shape[1]}，shape={arr.shape}")

    def _validate_time_alignment(self) -> None:
        """
        检查所有环境因子与 AIS 的时间轴是否一致。
        统一使用 *_time_all.npy 做检查。
        """
        ais_time = self._load_time_all("ais")

        if ais_time.ndim != 1:
            raise ValueError(f"ais_time_all.npy 维度应为 1，当前: {ais_time.shape}")

        if len(ais_time) == 0:
            raise ValueError("ais_time_all.npy 为空")

        for var in self.env_vars:
            var_time = self._load_time_all(var)

            if var_time.ndim != 1:
                raise ValueError(f"{var}_time_all.npy 维度应为 1，当前: {var_time.shape}")

            if len(var_time) != len(ais_time):
                raise ValueError(
                    f"{var} 与 AIS 时间轴长度不一致: {len(var_time)} vs {len(ais_time)}"
                )

            if not np.array_equal(var_time, ais_time):
                mismatch_idx = np.where(var_time != ais_time)[0]
                first_bad = int(mismatch_idx[0]) if len(mismatch_idx) > 0 else -1

                raise ValueError(
                    f"{var} 与 AIS 时间轴不一致。\n"
                    f"第一个错误位置 idx={first_bad}\n"
                    f"{var}_time={var_time[first_bad] if first_bad >= 0 else '未知'}\n"
                    f"ais_time={ais_time[first_bad] if first_bad >= 0 else '未知'}"
                )

        print("[OK] Time alignment check passed: all environmental variables match AIS timestamps.")

    # ------------------------------------------------------------------
    # Dataset 接口
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"idx 越界: idx={idx}, total_samples={self.total_samples}")

        # ================= 历史输入 =================
        x_env = self.env_tensor[idx: idx + self.seq_len]   # (seq_len, C_env, H, W)
        x_ais = self.ais_tensor[idx: idx + self.seq_len]   # (seq_len, 1, H, W)
        inputs = np.concatenate([x_env, x_ais], axis=1)    # (seq_len, C_env+1, H, W)

        # ================= 未来标签 =================
        target_slice = slice(idx + self.seq_len, idx + self.seq_len + self.pred_len)
        targets = self.ais_tensor[target_slice]            # (pred_len, 1, H, W)

        # 转 Tensor
        inputs = torch.from_numpy(np.ascontiguousarray(inputs)).float()
        targets = torch.from_numpy(np.ascontiguousarray(targets)).float()

        # ================= 主模型可选 future_env =================
        if self.return_future_env:
            future_env = self.env_tensor[target_slice]     # (pred_len, C_env, H, W)
            future_env = torch.from_numpy(np.ascontiguousarray(future_env)).float()
            return inputs, future_env, targets

        
        return inputs, targets


if __name__ == "__main__":
    env_vars = ["thetao", "uo", "vo", "so", "zos", "chl", "o2"]

    ds_train = STFishNetUltimateDataset(
        data_dir=r"/ST_FishNet_Features",
        env_vars=env_vars,
        split="train",
        seq_len=12,
        pred_len=1,
        return_future_env=False
    )

    print("len(ds_train) =", len(ds_train))
    x, y = ds_train[0]
    print("x.shape =", x.shape)
    print("y.shape =", y.shape)
