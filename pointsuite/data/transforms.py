"""
3D 点云数据增强变换模块
"""

import random
import numbers
import scipy
import scipy.ndimage
import scipy.interpolate
import scipy.stats
import numpy as np
import torch
import copy
from collections.abc import Sequence, Mapping


# ———— 通用操作 ————
# 组合多个变换
class Compose:
    def __init__(self, transforms=None):
        if transforms is None:
            transforms = []
        self.transforms = [t for t in transforms if t is not None]

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
            if data_dict is None:
                return None
        return data_dict


# 索引操作
def index_operator(data_dict, index, duplicate=False):
    # 对 "index_valid_keys" 中的键执行索引选择操作
    # 可在配置中通过 "Update" 变换自定义这些键
    # 对 data_dict 中的键进行索引操作
    if "index_valid_keys" not in data_dict:
        data_dict["index_valid_keys"] = [
            "coord",
            "intensity",
            "echo",
            "h_norm",
            "color",
            "normal",
            "class",
            "instance",
        ]
    if not duplicate:
        for key in data_dict["index_valid_keys"]:
            if key in data_dict:
                data_dict[key] = data_dict[key][index]
        return data_dict
    else:
        data_dict_ = dict()
        for key in data_dict.keys():
            if key in data_dict["index_valid_keys"]:
                data_dict_[key] = data_dict[key][index]
            else:
                data_dict_[key] = data_dict[key]
        return data_dict_


# 收集指定 key 的数据，支持 offset 和特征拼接
class Collect(object):
    def __init__(self, keys, offset_key=None, feat_keys=None):
        """
        e.g. Collect(keys=[coord], feat_keys=[coord, color])
        """
        if offset_key is None:
            offset_key = dict(offset="coord")
        if feat_keys is None:
            feat_keys = dict(feat="coord")
        self.keys = keys
        self.offset_key = offset_key
        self.feat_keys = feat_keys

    def __call__(self, data_dict):
        data = dict()
        if isinstance(self.keys, str):
            self.keys = [self.keys]
        for key in self.keys:
            # 直接传递数据（可能是 numpy 或 tensor）
            data[key] = data_dict[key]
        for key, value in self.offset_key.items():
            # offset 创建为 Tensor
            data[key] = torch.tensor([data_dict[value].shape[0]])
        for name, keys in self.feat_keys.items():
            name = name.replace("_keys", "")
            assert isinstance(keys, Sequence)
            # data[name] = torch.cat([data_dict[key].float() for key in keys], dim=1)
            tensors = []
            for key in keys:
                # 先转换为 Tensor（如果还不是）
                if isinstance(data_dict[key], np.ndarray):
                    tensor = torch.from_numpy(data_dict[key]).float()
                else:
                    tensor = data_dict[key].float()
                
                if tensor.dim() == 1:  # 如果是 [n]，扩展成 [n, 1]
                    tensor = tensor.unsqueeze(1)
                tensors.append(tensor)
            data[name] = torch.cat(tensors, dim=1)  # [n, c + m]（m 是额外拼接的 1D 张量数量）
        return data
    

# 更新指定的键
class Update(object):
    def __init__(self, keys_dict=None):
        if keys_dict is None:
            keys_dict = dict()
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            data_dict[key] = value
        return data_dict
    
     
# 将数据转换为张量
class ToTensor(object):
    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, str):
            # note that str is also a kind of sequence, judgement should before sequence
            return data
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, bool):
            return torch.from_numpy(data)
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
            return torch.from_numpy(data).long()
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            return torch.from_numpy(data).float()
        elif isinstance(data, Mapping):
            result = {sub_key: self(item) for sub_key, item in data.items()}
            return result
        elif isinstance(data, Sequence):
            result = [self(item) for item in data]
            return result
        else:
            raise TypeError(f"type {type(data)} cannot be converted to tensor.")
        

# ———— 坐标变换 ————
# 坐标标准化，减去质心并缩放到单位球
class NormalizeCoord(object):
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            # modified from pointnet2
            centroid = np.mean(data_dict["coord"], axis=0)
            data_dict["coord"] -= centroid
            m = np.max(np.sqrt(np.sum(data_dict["coord"] ** 2, axis=1)))
            data_dict["coord"] = data_dict["coord"] / m
        return data_dict
    

# 减去均值除以标准差（标准化）
class StandardNormalize(object):
    def __init__(self, apply_z=True):
        self.apply_z = apply_z

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            mean = data_dict["coord"].mean(axis=0)
            std = data_dict["coord"].std(axis=0)
            if not self.apply_z:
                mean[2] = 0
                std[2] = 1
            # 避免除以0
            std[std == 0] = 1
            data_dict["coord"] = (data_dict["coord"] - mean) / std
        return data_dict


# 减去最小值除以最大最小值之差（MinMax归一化）
class MinMaxNormalize(object):
    def __init__(self, apply_z=True):
        self.apply_z = apply_z

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            min_vals = data_dict["coord"].min(axis=0)
            max_vals = data_dict["coord"].max(axis=0)
            if not self.apply_z:
                min_vals[2] = 0
                max_vals[2] = 1
            # 计算范围，避免除以0
            ranges = max_vals - min_vals
            ranges[ranges == 0] = 1
            data_dict["coord"] = (data_dict["coord"] - min_vals) / ranges
        return data_dict


# 坐标偏移（最小值）
class PositiveShift(object):
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            coord_min = np.min(data_dict["coord"], 0)
            data_dict["coord"] -= coord_min
        return data_dict


# 坐标偏移（中心）
class CenterShift(object):
    def __init__(self, apply_z=True):
        self.apply_z = apply_z

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            x_min, y_min, z_min = data_dict["coord"].min(axis=0)
            x_max, y_max, _ = data_dict["coord"].max(axis=0)
            if self.apply_z:
                shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]
            else:
                shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, 0]
            data_dict["coord"] -= shift
        return data_dict


# 随机偏移
class RandomShift(object):
    def __init__(self, shift=((-0.2, 0.2), (-0.2, 0.2), (0, 0))):
        self.shift = shift

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            shift_x = np.random.uniform(self.shift[0][0], self.shift[0][1])
            shift_y = np.random.uniform(self.shift[1][0], self.shift[1][1])
            shift_z = np.random.uniform(self.shift[2][0], self.shift[2][1])
            data_dict["coord"] += [shift_x, shift_y, shift_z]
        return data_dict


# 随机丢弃
class RandomDropout(object):
    def __init__(self, dropout_ratio=0.2, p=0.5):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = p

    def __call__(self, data_dict):
        if random.random() < self.dropout_application_ratio:
            n = len(data_dict["coord"])
            idx = np.random.choice(n, int(n * (1 - self.dropout_ratio)), replace=False)
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx = np.unique(np.append(idx, data_dict["sampled_index"]))
                mask = np.zeros_like(data_dict["class"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx])[0]
            data_dict = index_operator(data_dict, idx)
        return data_dict


# 随机旋转
class RandomRotate(object):
    def __init__(self, angle=None, center=None, axis="z", always_apply=False, p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict


# 随机旋转到特定角度
class RandomRotateTargetAngle(object):
    def __init__(
        self, angle=(1 / 2, 1, 3 / 2), center=None, axis="z", always_apply=False, p=0.75
    ):
        self.angle = angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.choice(self.angle) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict


# 随机缩放
class RandomScale(object):
    def __init__(self, scale=None, anisotropic=False):
        self.scale = scale if scale is not None else [0.95, 1.05]
        self.anisotropic = anisotropic

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            scale = np.random.uniform(
                self.scale[0], self.scale[1], 3 if self.anisotropic else 1
            )
            data_dict["coord"] *= scale
        return data_dict


# 随机翻转
class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            if "coord" in data_dict.keys():
                data_dict["coord"][:, 0] = -data_dict["coord"][:, 0]
            if "normal" in data_dict.keys():
                data_dict["normal"][:, 0] = -data_dict["normal"][:, 0]
        if np.random.rand() < self.p:
            if "coord" in data_dict.keys():
                data_dict["coord"][:, 1] = -data_dict["coord"][:, 1]
            if "normal" in data_dict.keys():
                data_dict["normal"][:, 1] = -data_dict["normal"][:, 1]
        return data_dict


# 随机抖动
class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            jitter = np.clip(
                self.sigma * np.random.randn(data_dict["coord"].shape[0], 3),
                -self.clip,
                self.clip,
            )
            data_dict["coord"] += jitter
        return data_dict


# 高斯抖动
class ClipGaussianJitter(object):
    def __init__(self, scalar=0.02, store_jitter=False):
        self.scalar = scalar
        self.mean = np.mean(3)
        self.cov = np.identity(3)
        self.quantile = 1.96
        self.store_jitter = store_jitter

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            jitter = np.random.multivariate_normal(
                self.mean, self.cov, data_dict["coord"].shape[0]
            )
            jitter = self.scalar * np.clip(jitter / 1.96, -1, 1)
            data_dict["coord"] += jitter
            if self.store_jitter:
                data_dict["jitter"] = jitter
        return data_dict
    

# 顺序打乱
class ShufflePoint(object):
    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        shuffle_index = np.arange(data_dict["coord"].shape[0])
        np.random.shuffle(shuffle_index)
        data_dict = index_operator(data_dict, shuffle_index)
        return data_dict


# ———— 强度变换 ————
# Intensity 自动检测并归一化
class AutoNormalizeIntensity(object):
    def __init__(self, target_range=(0, 1)):
        """
        自动检测 intensity 位数并归一化到目标范围
        
        检测逻辑：
        - 如果 max <= 1.0: 认为已归一化，不处理
        - 如果 max <= 255: 认为是 8 位，除以 255
        - 如果 max <= 65535: 认为是 16 位，除以 65535
        - 否则: 使用实际的 max-min 范围归一化
        
        Args:
            target_range: 目标范围 (min, max)，默认 (0, 1)
        """
        self.target_range = target_range

    def __call__(self, data_dict):
        if "intensity" in data_dict.keys():
            intensity = data_dict["intensity"].astype(np.float32)
            
            # 检测当前范围
            i_min = intensity.min()
            i_max = intensity.max()
            
            # 自动检测位数并归一化
            if i_max <= 1.0:
                # 已经归一化，可能需要调整范围
                if self.target_range != (0, 1):
                    # 从 [0, 1] 映射到 target_range
                    target_min, target_max = self.target_range
                    intensity = intensity * (target_max - target_min) + target_min
            elif i_max <= 255:
                # 8 位
                intensity = intensity / 255.0
                if self.target_range != (0, 1):
                    target_min, target_max = self.target_range
                    intensity = intensity * (target_max - target_min) + target_min
            elif i_max <= 65535:
                # 16 位
                intensity = intensity / 65535.0
                if self.target_range != (0, 1):
                    target_min, target_max = self.target_range
                    intensity = intensity * (target_max - target_min) + target_min
            else:
                # 未知范围，使用 min-max 归一化
                if i_max > i_min:
                    intensity = (intensity - i_min) / (i_max - i_min)
                    if self.target_range != (0, 1):
                        target_min, target_max = self.target_range
                        intensity = intensity * (target_max - target_min) + target_min
            
            data_dict["intensity"] = intensity
        return data_dict


# Intensity 归一化（指定位数）
class NormalizeIntensity(object):
    def __init__(self, max_value=65535.0):
        """
        使用指定的最大值归一化 intensity 到 [0, 1]
        
        Args:
            max_value: 最大可能的强度值（如 65535 表示 16 位，255 表示 8 位）
        """
        self.max_value = max_value

    def __call__(self, data_dict):
        if "intensity" in data_dict.keys():
            data_dict["intensity"] = data_dict["intensity"].astype(np.float32) / self.max_value
        return data_dict


# Intensity 随机缩放
class RandomIntensityScale(object):
    def __init__(self, scale=(0.8, 1.2), p=0.95):
        """
        Randomly scale intensity values.
        
        Args:
            scale: (min_scale, max_scale) tuple
            p: Probability of applying the transform
        """
        self.scale = scale
        self.p = p

    def __call__(self, data_dict):
        if "intensity" in data_dict.keys() and np.random.rand() < self.p:
            scale_factor = np.random.uniform(self.scale[0], self.scale[1])
            data_dict["intensity"] = np.clip(
                data_dict["intensity"] * scale_factor, 0, 1
            ).astype(data_dict["intensity"].dtype)
        return data_dict


# Intensity 随机偏移
class RandomIntensityShift(object):
    def __init__(self, shift=(-0.1, 0.1), p=0.95):
        """
        Randomly shift intensity values.
        
        Args:
            shift: (min_shift, max_shift) tuple for additive shift
            p: Probability of applying the transform
        """
        self.shift = shift
        self.p = p

    def __call__(self, data_dict):
        if "intensity" in data_dict.keys() and np.random.rand() < self.p:
            shift_value = np.random.uniform(self.shift[0], self.shift[1])
            if data_dict["intensity"].dtype in [np.float32, np.float64]:
                # Normalized intensity [0, 1]
                data_dict["intensity"] = np.clip(
                    data_dict["intensity"] + shift_value, 0, 1
                )
            else:
                # Raw intensity (e.g., uint16)
                max_val = np.iinfo(data_dict["intensity"].dtype).max
                shift_value_scaled = shift_value * max_val
                data_dict["intensity"] = np.clip(
                    data_dict["intensity"] + shift_value_scaled, 0, max_val
                ).astype(data_dict["intensity"].dtype)
        return data_dict


# Intensity 随机噪声
class RandomIntensityNoise(object):
    def __init__(self, sigma=0.01, p=0.5):
        """
        Add random Gaussian noise to intensity.
        
        Args:
            sigma: Standard deviation of Gaussian noise (for normalized intensity)
            p: Probability of applying the transform
        """
        self.sigma = sigma
        self.p = p

    def __call__(self, data_dict):
        if "intensity" in data_dict.keys() and np.random.rand() < self.p:
            noise = np.random.normal(0, self.sigma, data_dict["intensity"].shape)
            if data_dict["intensity"].dtype in [np.float32, np.float64]:
                # Normalized intensity
                data_dict["intensity"] = np.clip(
                    data_dict["intensity"] + noise, 0, 1
                ).astype(data_dict["intensity"].dtype)
            else:
                # Raw intensity
                max_val = np.iinfo(data_dict["intensity"].dtype).max
                noise_scaled = noise * max_val
                data_dict["intensity"] = np.clip(
                    data_dict["intensity"] + noise_scaled, 0, max_val
                ).astype(data_dict["intensity"].dtype)
        return data_dict


# Intensity 随机丢弃（置为0）
class RandomIntensityDrop(object):
    def __init__(self, drop_ratio=0.1, p=0.2):
        """
        Randomly drop (set to 0) a portion of intensity values.
        
        Args:
            drop_ratio: Ratio of points whose intensity will be set to 0
            p: Probability of applying the transform
        """
        self.drop_ratio = drop_ratio
        self.p = p

    def __call__(self, data_dict):
        if "intensity" in data_dict.keys() and np.random.rand() < self.p:
            n = len(data_dict["intensity"])
            drop_mask = np.random.rand(n) < self.drop_ratio
            data_dict["intensity"][drop_mask] = 0
        return data_dict


# Intensity Gamma 变换
class RandomIntensityGamma(object):
    def __init__(self, gamma_range=(0.8, 1.2), p=0.5):
        """
        Apply random gamma correction to intensity.
        
        Args:
            gamma_range: (min_gamma, max_gamma) range for gamma values
            p: Probability of applying the transform
        """
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, data_dict):
        if "intensity" in data_dict.keys() and np.random.rand() < self.p:
            gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
            
            if data_dict["intensity"].dtype in [np.float32, np.float64]:
                # Normalized intensity [0, 1]
                data_dict["intensity"] = np.power(
                    data_dict["intensity"], gamma
                ).astype(data_dict["intensity"].dtype)
            else:
                # Raw intensity - normalize first, apply gamma, then denormalize
                max_val = np.iinfo(data_dict["intensity"].dtype).max
                normalized = data_dict["intensity"].astype(np.float32) / max_val
                gamma_corrected = np.power(normalized, gamma)
                data_dict["intensity"] = (gamma_corrected * max_val).astype(
                    data_dict["intensity"].dtype
                )
        return data_dict


# Intensity 标准化（减均值除方差）
class StandardNormalizeIntensity(object):
    def __init__(self, mean=None, std=None):
        """
        Standardize intensity by subtracting mean and dividing by std.
        
        Args:
            mean: Mean value for normalization. If None, computed from data.
            std: Std value for normalization. If None, computed from data.
        """
        self.mean = mean
        self.std = std

    def __call__(self, data_dict):
        if "intensity" in data_dict.keys():
            intensity = data_dict["intensity"].astype(np.float32)
            
            # Compute mean and std if not provided
            mean = self.mean if self.mean is not None else intensity.mean()
            std = self.std if self.std is not None else intensity.std()
            
            # Avoid division by zero
            if std == 0:
                std = 1.0
            
            data_dict["intensity"] = ((intensity - mean) / std).astype(np.float32)
        return data_dict


# Intensity MinMax 归一化
class MinMaxNormalizeIntensity(object):
    def __init__(self, min_val=None, max_val=None, target_range=(0, 1)):
        """
        MinMax normalize intensity to target range.
        
        Args:
            min_val: Minimum value for normalization. If None, computed from data.
            max_val: Maximum value for normalization. If None, computed from data.
            target_range: Target range (min, max) for normalized values.
        """
        self.min_val = min_val
        self.max_val = max_val
        self.target_range = target_range

    def __call__(self, data_dict):
        if "intensity" in data_dict.keys():
            intensity = data_dict["intensity"].astype(np.float32)
            
            # Compute min and max if not provided
            min_val = self.min_val if self.min_val is not None else intensity.min()
            max_val = self.max_val if self.max_val is not None else intensity.max()
            
            # Avoid division by zero
            if max_val == min_val:
                data_dict["intensity"] = np.full_like(
                    intensity, 
                    (self.target_range[0] + self.target_range[1]) / 2,
                    dtype=np.float32
                )
            else:
                # Normalize to [0, 1] first
                normalized = (intensity - min_val) / (max_val - min_val)
                # Scale to target range
                target_min, target_max = self.target_range
                data_dict["intensity"] = (
                    normalized * (target_max - target_min) + target_min
                ).astype(np.float32)
        return data_dict

    
# ———— 颜色变换 ————
# 颜色自动检测并归一化
class AutoNormalizeColor(object):
    def __init__(self, target_range=(0, 255)):
        """
        自动检测 color 位数并归一化到目标范围
        
        检测逻辑：
        - 如果 max <= 1.0: 认为已归一化到 [0, 1]，映射到 target_range
        - 如果 max <= 255: 认为是 8 位，已在正确范围
        - 如果 max <= 65535: 认为是 16 位，转换到 8 位 [0, 255]
        - 否则: 使用实际的 max-min 范围归一化
        
        注意：大部分颜色增强（ChromaticJitter 等）期望 [0, 255] 范围
        
        Args:
            target_range: 目标范围，默认 (0, 255) 用于颜色增强
        """
        self.target_range = target_range

    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            color = data_dict["color"].astype(np.float32)
            
            # 检测当前范围（使用所有通道的最大值）
            c_min = color.min()
            c_max = color.max()
            
            target_min, target_max = self.target_range
            
            # 自动检测位数并归一化
            if c_max <= 1.0:
                # 已经归一化到 [0, 1]，映射到 target_range
                color = color * (target_max - target_min) + target_min
            elif c_max <= 255:
                # 8 位，已在 [0, 255] 范围
                if self.target_range != (0, 255):
                    # 需要映射到其他范围
                    color = (color / 255.0) * (target_max - target_min) + target_min
            elif c_max <= 65535:
                # 16 位，转换到目标范围
                color = (color / 65535.0) * (target_max - target_min) + target_min
            else:
                # 未知范围，使用 min-max 归一化
                if c_max > c_min:
                    color = (color - c_min) / (c_max - c_min)
                    color = color * (target_max - target_min) + target_min
            
            data_dict["color"] = color
        return data_dict


# 颜色归一化（指定位数）
class NormalizeColor(object):
    def __init__(self, source_bits=16, target_range=(0, 255)):
        """
        使用指定的位数归一化 color
        
        Args:
            source_bits: 源数据位数（8 或 16）
            target_range: 目标范围，默认 (0, 255)
        """
        self.source_bits = source_bits
        self.target_range = target_range
        self.source_max = (2 ** source_bits) - 1

    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            color = data_dict["color"].astype(np.float32)
            
            # 归一化到 [0, 1]
            color = color / self.source_max
            
            # 映射到目标范围
            target_min, target_max = self.target_range
            color = color * (target_max - target_min) + target_min
            
            data_dict["color"] = color
        return data_dict


# 颜色对比度增强
class ChromaticAutoContrast(object):
    def __init__(self, p=0.2, blend_factor=None):
        self.p = p
        self.blend_factor = blend_factor

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            lo = np.min(data_dict["color"], 0, keepdims=True)
            hi = np.max(data_dict["color"], 0, keepdims=True)
            scale = 255 / (hi - lo)
            contrast_feat = (data_dict["color"][:, :3] - lo) * scale
            blend_factor = (
                np.random.rand() if self.blend_factor is None else self.blend_factor
            )
            data_dict["color"][:, :3] = (1 - blend_factor) * data_dict["color"][
                :, :3
            ] + blend_factor * contrast_feat
        return data_dict


# 颜色随机平移
class ChromaticTranslation(object):
    def __init__(self, p=0.95, ratio=0.05):
        self.p = p
        self.ratio = ratio

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.ratio
            data_dict["color"][:, :3] = np.clip(tr + data_dict["color"][:, :3], 0, 255)
        return data_dict


# 颜色随机抖动
class ChromaticJitter(object):
    def __init__(self, p=0.95, std=0.005):
        self.p = p
        self.std = std

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            noise = np.random.randn(data_dict["color"].shape[0], 3)
            noise *= self.std * 255
            data_dict["color"][:, :3] = np.clip(
                noise + data_dict["color"][:, :3], 0, 255
            )
        return data_dict


# 随机颜色灰度化
class RandomColorGrayScale(object):
    def __init__(self, p):
        self.p = p

    @staticmethod
    def rgb_to_grayscale(color, num_output_channels=1):
        if color.shape[-1] < 3:
            raise TypeError(
                "Input color should have at least 3 dimensions, but found {}".format(
                    color.shape[-1]
                )
            )

        if num_output_channels not in (1, 3):
            raise ValueError("num_output_channels should be either 1 or 3")

        r, g, b = color[..., 0], color[..., 1], color[..., 2]
        gray = (0.2989 * r + 0.587 * g + 0.114 * b).astype(color.dtype)
        gray = np.expand_dims(gray, axis=-1)

        if num_output_channels == 3:
            gray = np.broadcast_to(gray, color.shape)

        return gray

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            data_dict["color"] = self.rgb_to_grayscale(data_dict["color"], 3)
        return data_dict


# 随机颜色抖动
class RandomColorJitter(object):
    """
    Random Color Jitter for 3D point cloud (refer torchvision)
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.95):
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )
        self.p = p

    @staticmethod
    def _check_input(
        value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".format(name)
                )
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with length 2.".format(
                    name
                )
            )

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def blend(color1, color2, ratio):
        ratio = float(ratio)
        bound = 255.0
        return (
            (ratio * color1 + (1.0 - ratio) * color2)
            .clip(0, bound)
            .astype(color1.dtype)
        )

    @staticmethod
    def rgb2hsv(rgb):
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb, axis=-1)
        minc = np.min(rgb, axis=-1)
        eqc = maxc == minc
        cr = maxc - minc
        s = cr / (np.ones_like(maxc) * eqc + maxc * (1 - eqc))
        cr_divisor = np.ones_like(maxc) * eqc + cr * (1 - eqc)
        rc = (maxc - r) / cr_divisor
        gc = (maxc - g) / cr_divisor
        bc = (maxc - b) / cr_divisor

        hr = (maxc == r) * (bc - gc)
        hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
        hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
        h = hr + hg + hb
        h = (h / 6.0 + 1.0) % 1.0
        return np.stack((h, s, maxc), axis=-1)

    @staticmethod
    def hsv2rgb(hsv):
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = np.floor(h * 6.0)
        f = (h * 6.0) - i
        i = i.astype(np.int32)

        p = np.clip((v * (1.0 - s)), 0.0, 1.0)
        q = np.clip((v * (1.0 - s * f)), 0.0, 1.0)
        t = np.clip((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
        i = i % 6
        mask = np.expand_dims(i, axis=-1) == np.arange(6)

        a1 = np.stack((v, q, p, p, t, v), axis=-1)
        a2 = np.stack((t, v, v, q, p, p), axis=-1)
        a3 = np.stack((p, p, t, v, v, q), axis=-1)
        a4 = np.stack((a1, a2, a3), axis=-1)

        return np.einsum("...na, ...nab -> ...nb", mask.astype(hsv.dtype), a4)

    def adjust_brightness(self, color, brightness_factor):
        if brightness_factor < 0:
            raise ValueError(
                "brightness_factor ({}) is not non-negative.".format(brightness_factor)
            )

        return self.blend(color, np.zeros_like(color), brightness_factor)

    def adjust_contrast(self, color, contrast_factor):
        if contrast_factor < 0:
            raise ValueError(
                "contrast_factor ({}) is not non-negative.".format(contrast_factor)
            )
        mean = np.mean(RandomColorGrayScale.rgb_to_grayscale(color))
        return self.blend(color, mean, contrast_factor)

    def adjust_saturation(self, color, saturation_factor):
        if saturation_factor < 0:
            raise ValueError(
                "saturation_factor ({}) is not non-negative.".format(saturation_factor)
            )
        gray = RandomColorGrayScale.rgb_to_grayscale(color)
        return self.blend(color, gray, saturation_factor)

    def adjust_hue(self, color, hue_factor):
        if not (-0.5 <= hue_factor <= 0.5):
            raise ValueError(
                "hue_factor ({}) is not in [-0.5, 0.5].".format(hue_factor)
            )
        orig_dtype = color.dtype
        hsv = self.rgb2hsv(color / 255.0)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        h = (h + hue_factor) % 1.0
        hsv = np.stack((h, s, v), axis=-1)
        color_hue_adj = (self.hsv2rgb(hsv) * 255.0).astype(orig_dtype)
        return color_hue_adj

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        fn_idx = torch.randperm(4)
        b = (
            None
            if brightness is None
            else np.random.uniform(brightness[0], brightness[1])
        )
        c = None if contrast is None else np.random.uniform(contrast[0], contrast[1])
        s = (
            None
            if saturation is None
            else np.random.uniform(saturation[0], saturation[1])
        )
        h = None if hue is None else np.random.uniform(hue[0], hue[1])
        return fn_idx, b, c, s, h

    def __call__(self, data_dict):
        (
            fn_idx,
            brightness_factor,
            contrast_factor,
            saturation_factor,
            hue_factor,
        ) = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        for fn_id in fn_idx:
            if (
                fn_id == 0
                and brightness_factor is not None
                and np.random.rand() < self.p
            ):
                data_dict["color"] = self.adjust_brightness(
                    data_dict["color"], brightness_factor
                )
            elif (
                fn_id == 1 and contrast_factor is not None and np.random.rand() < self.p
            ):
                data_dict["color"] = self.adjust_contrast(
                    data_dict["color"], contrast_factor
                )
            elif (
                fn_id == 2
                and saturation_factor is not None
                and np.random.rand() < self.p
            ):
                data_dict["color"] = self.adjust_saturation(
                    data_dict["color"], saturation_factor
                )
            elif fn_id == 3 and hue_factor is not None and np.random.rand() < self.p:
                data_dict["color"] = self.adjust_hue(data_dict["color"], hue_factor)
        return data_dict


# 随机颜色饱和度
class HueSaturationTranslation(object):
    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype("float")
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select(
            [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc
        )
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype("uint8")
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype("uint8")

    def __init__(self, hue_max=0.5, saturation_max=0.2):
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            # Assume color[:, :3] is rgb
            hsv = HueSaturationTranslation.rgb_to_hsv(data_dict["color"][:, :3])
            hue_val = (np.random.rand() - 0.5) * 2 * self.hue_max
            sat_ratio = 1 + (np.random.rand() - 0.5) * 2 * self.saturation_max
            hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
            hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
            data_dict["color"][:, :3] = np.clip(
                HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255
            )
        return data_dict


# 随机颜色丢弃
class RandomColorDrop(object):
    def __init__(self, p=0.2, color_augment=0.0):
        self.p = p
        self.color_augment = color_augment

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            data_dict["color"] *= self.color_augment
        return data_dict

    def __repr__(self):
        return "RandomColorDrop(color_augment: {}, p: {})".format(
            self.color_augment, self.p
        )


# 弹性失真，模拟自然变形
class ElasticDistortion(object):
    def __init__(self, distortion_params=None):
        self.distortion_params = (
            [[0.2, 0.4], [0.8, 1.6]] if distortion_params is None else distortion_params
        )

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        """
        Apply elastic distortion on sparse coordinate space.
        pointcloud: numpy array of (number of points, at least 3 spatial dims)
        granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
        magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(
                noise, blurx, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blury, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blurz, mode="constant", cval=0
            )

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(
                coords_min - granularity,
                coords_min + granularity * (noise_dim - 2),
                noise_dim,
            )
        ]
        interp = scipy.interpolate.RegularGridInterpolator(
            ax, noise, bounds_error=False, fill_value=0
        )
        coords += interp(coords) * magnitude
        return coords

    def __call__(self, data_dict):
        if "coord" in data_dict.keys() and self.distortion_params is not None:
            if random.random() < 0.95:
                for granularity, magnitude in self.distortion_params:
                    data_dict["coord"] = self.elastic_distortion(
                        data_dict["coord"], granularity, magnitude
                    )
        return data_dict


# ———— 归一化高程（h_norm）变换 ————
# 归一化高程自动归一化（可选裁剪）
class AutoNormalizeHNorm(object):
    def __init__(self, clip_range=None):
        """
        自动归一化 h_norm（可选裁剪异常值）
        
        默认行为（clip_range=None）：
        - ✅ 不裁剪任何值，保留负值和极大值
        - ✅ 负值可能代表地下结构（地下室、隧道、坑洞）
        - ✅ 极大值可能代表真实高层建筑或噪声
        - ✅ 让模型学习识别和处理异常值，增强鲁棒性
        
        可选裁剪（clip_range=(min, max)）：
        - 如 (0, 50) 将高程限制在 0-50m（排除明显异常值）
        - 如 (-5, 100) 保留合理的地下和高空范围
        
        Args:
            clip_range: 裁剪范围 (min, max)，默认 None（不裁剪）
                       None: 不裁剪，保留所有值（推荐）
                       (min, max): 裁剪到指定范围
                       (None, max): 只裁剪上界
                       (min, None): 只裁剪下界
        """
        self.clip_range = clip_range

    def __call__(self, data_dict):
        if "h_norm" in data_dict.keys():
            h_norm = data_dict["h_norm"].astype(np.float32)
            
            # 可选裁剪异常值
            if self.clip_range is not None:
                if self.clip_range[0] is not None:
                    h_norm = np.maximum(h_norm, self.clip_range[0])
                if self.clip_range[1] is not None:
                    h_norm = np.minimum(h_norm, self.clip_range[1])
            
            data_dict["h_norm"] = h_norm
        return data_dict


# 归一化高程标准化
class StandardNormalizeHNorm(object):
    def __init__(self, mean=None, std=None):
        """
        标准化 h_norm（Z-score 归一化）
        
        适用于需要零均值、单位方差输入的模型
        
        Args:
            mean: 均值，如果为 None 则从数据计算
            std: 标准差，如果为 None 则从数据计算
        """
        self.mean = mean
        self.std = std

    def __call__(self, data_dict):
        if "h_norm" in data_dict.keys():
            h_norm = data_dict["h_norm"].astype(np.float32)
            
            mean = self.mean if self.mean is not None else h_norm.mean()
            std = self.std if self.std is not None else h_norm.std()
            
            # 避免除零
            if std == 0:
                std = 1.0
            
            data_dict["h_norm"] = ((h_norm - mean) / std).astype(np.float32)
        return data_dict


# 归一化高程随机缩放
class RandomHNormScale(object):
    def __init__(self, scale=(0.9, 1.1), p=0.5):
        """
        随机缩放 h_norm
        
        模拟不同的地面识别精度或高程测量误差
        
        Args:
            scale: 缩放范围 (min_scale, max_scale)
            p: 应用概率
        """
        self.scale = scale
        self.p = p

    def __call__(self, data_dict):
        if "h_norm" in data_dict.keys() and np.random.rand() < self.p:
            scale_factor = np.random.uniform(self.scale[0], self.scale[1])
            data_dict["h_norm"] = (data_dict["h_norm"] * scale_factor).astype(
                data_dict["h_norm"].dtype
            )
        return data_dict


# 归一化高程随机噪声
class RandomHNormNoise(object):
    def __init__(self, sigma=0.1, p=0.5):
        """
        为 h_norm 添加随机高斯噪声
        
        模拟地面高程估计的局部误差
        
        Args:
            sigma: 高斯噪声的标准差（单位与 h_norm 相同，通常是米）
            p: 应用概率
        """
        self.sigma = sigma
        self.p = p

    def __call__(self, data_dict):
        if "h_norm" in data_dict.keys() and np.random.rand() < self.p:
            noise = np.random.normal(0, self.sigma, data_dict["h_norm"].shape)
            data_dict["h_norm"] = (data_dict["h_norm"] + noise).astype(
                data_dict["h_norm"].dtype
            )
        return data_dict


# 归一化高程对数变换
class LogTransformHNorm(object):
    def __init__(self, epsilon=1e-6):
        """
        对 h_norm 进行对数变换
        
        用于处理高度范围很大的场景（如建筑物和地面）
        使模型对不同高度尺度更敏感
        
        Args:
            epsilon: 避免 log(0) 的小常数
        """
        self.epsilon = epsilon

    def __call__(self, data_dict):
        if "h_norm" in data_dict.keys():
            h_norm = data_dict["h_norm"].astype(np.float32)
            # 确保非负
            h_norm = np.maximum(h_norm, 0)
            # 对数变换
            data_dict["h_norm"] = np.log(h_norm + self.epsilon).astype(np.float32)
        return data_dict


# 归一化高程分桶编码
class BinHNorm(object):
    def __init__(self, bins=10, range=(0, 20)):
        """
        将 h_norm 离散化为桶（bins）
        
        将连续的高度值转换为离散的高度等级
        
        Args:
            bins: 桶的数量
            range: 高度范围 (min, max)
        """
        self.bins = bins
        self.range = range

    def __call__(self, data_dict):
        if "h_norm" in data_dict.keys():
            h_norm = data_dict["h_norm"].astype(np.float32)
            
            # 使用 numpy 的 digitize 进行分桶
            bin_edges = np.linspace(self.range[0], self.range[1], self.bins + 1)
            binned = np.digitize(h_norm, bin_edges) - 1
            
            # 裁剪到 [0, bins-1]
            binned = np.clip(binned, 0, self.bins - 1)
            
            # 转换为 float（归一化到 [0, 1]）
            data_dict["h_norm"] = (binned / (self.bins - 1)).astype(np.float32)
        return data_dict


# ———— 噪点注入增强 ————
# 添加极端高度噪点
class AddExtremeOutliers(object):
    def __init__(self, 
                 num_outliers=None, 
                 ratio=0.01,
                 height_range=(-10, 100),
                 height_mode='uniform',
                 intensity_range=(0, 1),
                 color_value=(128, 128, 128),
                 class_label=None,
                 p=0.5):
        """
        添加极端高度噪点（模拟大气噪声、多路径反射等）
        
        噪点来源模拟：
        - 🌩️ 大气噪声：飞鸟、云、灰尘（高空噪点）
        - 🔻 地面反射：水面、玻璃反射（低空/地下噪点）
        - 📡 多路径反射：建筑物、金属表面反射（随机高度）
        - 🌳 植被遮挡：树叶间隙的伪点（中等高度）
        
        噪点属性设置策略：
        - coord: 在现有点云的 XY 范围内随机分布，Z 为极端值
        - intensity: 通常较弱（大气噪声）或很强（反射）
        - color: 灰色（未知）或随机色
        - h_norm: 根据 Z 和地面高程计算（或设为极端值）
        - class: 噪声类别（可配置，如 0=未分类）
        
        Args:
            num_outliers: 固定噪点数量，如果指定则忽略 ratio
            ratio: 噪点数量占总点数的比例，默认 0.01（1%）
            height_range: 噪点高度范围 (z_min, z_max)，默认 (-10, 100) 米
                         相对于原始 Z 坐标，不是 h_norm
            height_mode: 高度分布模式
                - 'uniform': 均匀分布在 height_range
                - 'bimodal': 双峰分布（高空+低空）
                - 'high': 只在高空（模拟飞鸟、云）
                - 'low': 只在低空/地下（模拟反射）
            intensity_range: 噪点强度范围 (min, max)，默认 (0, 1)
            color_value: 噪点颜色
                - tuple (R, G, B): 固定颜色，如 (128, 128, 128) 灰色
                - 'random': 随机颜色
                - 'inherit': 从最近的真实点继承颜色
            class_label: 噪点的分类标签
                - None: 从最近的真实点继承
                - int: 固定标签（如 0=未分类, -1=噪声）
                - 'ignore': 使用 ignore_label（通常是 -1）
            p: 应用概率
        """
        self.num_outliers = num_outliers
        self.ratio = ratio
        self.height_range = height_range
        self.height_mode = height_mode
        self.intensity_range = intensity_range
        self.color_value = color_value
        self.class_label = class_label
        self.p = p

    def __call__(self, data_dict):
        if np.random.rand() > self.p:
            return data_dict
        
        if "coord" not in data_dict:
            return data_dict
        
        coord = data_dict["coord"]
        n_points = len(coord)
        
        # 计算噪点数量
        if self.num_outliers is not None:
            n_outliers = self.num_outliers
        else:
            n_outliers = max(1, int(n_points * self.ratio))
        
        # 获取原始点云的 XY 范围
        x_min, y_min, z_min = coord.min(axis=0)
        x_max, y_max, z_max = coord.max(axis=0)
        
        # 生成噪点坐标
        outlier_xy = np.random.rand(n_outliers, 2)
        outlier_xy[:, 0] = outlier_xy[:, 0] * (x_max - x_min) + x_min
        outlier_xy[:, 1] = outlier_xy[:, 1] * (y_max - y_min) + y_min
        
        # 根据模式生成高度
        if self.height_mode == 'uniform':
            # 均匀分布
            outlier_z = np.random.uniform(
                self.height_range[0], self.height_range[1], n_outliers
            )
        elif self.height_mode == 'bimodal':
            # 双峰分布：50% 高空，50% 低空
            n_high = n_outliers // 2
            n_low = n_outliers - n_high
            z_high = np.random.uniform(
                max(self.height_range[0], z_max), self.height_range[1], n_high
            )
            z_low = np.random.uniform(
                self.height_range[0], min(self.height_range[1], z_min), n_low
            )
            outlier_z = np.concatenate([z_high, z_low])
            np.random.shuffle(outlier_z)
        elif self.height_mode == 'high':
            # 只在高空
            outlier_z = np.random.uniform(
                max(self.height_range[0], z_max), self.height_range[1], n_outliers
            )
        elif self.height_mode == 'low':
            # 只在低空/地下
            outlier_z = np.random.uniform(
                self.height_range[0], min(self.height_range[1], z_min), n_outliers
            )
        else:
            raise ValueError(f"Unknown height_mode: {self.height_mode}")
        
        # 组合噪点坐标
        outlier_coord = np.column_stack([outlier_xy, outlier_z]).astype(coord.dtype)
        
        # 添加噪点到坐标
        data_dict["coord"] = np.vstack([coord, outlier_coord])
        
        # 处理其他属性
        # 1. Intensity
        if "intensity" in data_dict:
            outlier_intensity = np.random.uniform(
                self.intensity_range[0], self.intensity_range[1], n_outliers
            ).astype(data_dict["intensity"].dtype)
            data_dict["intensity"] = np.concatenate([
                data_dict["intensity"], outlier_intensity
            ])
        
        # 2. Color
        if "color" in data_dict:
            if self.color_value == 'random':
                # 随机颜色
                outlier_color = np.random.uniform(
                    0, 255, (n_outliers, 3)
                ).astype(data_dict["color"].dtype)
            elif self.color_value == 'inherit':
                # 从最近的真实点继承（使用简单的随机采样）
                random_indices = np.random.choice(n_points, n_outliers)
                outlier_color = data_dict["color"][random_indices].copy()
            else:
                # 固定颜色
                outlier_color = np.tile(
                    np.array(self.color_value, dtype=data_dict["color"].dtype),
                    (n_outliers, 1)
                )
            data_dict["color"] = np.vstack([data_dict["color"], outlier_color])
        
        # 3. h_norm
        if "h_norm" in data_dict:
            # 计算噪点的 h_norm
            # 简化：假设地面高程为原始点云的最小 Z
            ground_z = z_min
            outlier_h_norm = (outlier_z - ground_z).astype(data_dict["h_norm"].dtype)
            data_dict["h_norm"] = np.concatenate([
                data_dict["h_norm"], outlier_h_norm
            ])
        
        # 4. Normal
        if "normal" in data_dict:
            # 噪点的法向量：随机方向（模拟噪声）
            outlier_normal = np.random.randn(n_outliers, 3).astype(
                data_dict["normal"].dtype
            )
            # 归一化
            norms = np.linalg.norm(outlier_normal, axis=1, keepdims=True)
            outlier_normal = outlier_normal / (norms + 1e-8)
            data_dict["normal"] = np.vstack([data_dict["normal"], outlier_normal])
        
        # 5. Echo
        if "echo" in data_dict:
            # 噪点通常是单次回波
            outlier_echo = np.ones((n_outliers, 2), dtype=data_dict["echo"].dtype)
            # 设为首次且末次回波（单次回波的特征）
            data_dict["echo"] = np.vstack([data_dict["echo"], outlier_echo])
        
        # 6. Classification
        if "class" in data_dict:
            if self.class_label is None:
                # 从最近的真实点继承（随机采样）
                random_indices = np.random.choice(n_points, n_outliers)
                outlier_class = data_dict["class"][random_indices].copy()
            elif self.class_label == 'ignore':
                # 使用 ignore_label（通常在 dataset 中定义）
                outlier_class = np.full(n_outliers, -1, dtype=data_dict["class"].dtype)
            else:
                # 固定标签
                outlier_class = np.full(
                    n_outliers, self.class_label, dtype=data_dict["class"].dtype
                )
            data_dict["class"] = np.concatenate([data_dict["class"], outlier_class])
        
        return data_dict


# 添加局部噪点簇
class AddLocalNoiseClusters(object):
    def __init__(self,
                 num_clusters=3,
                 points_per_cluster=(5, 20),
                 cluster_radius=2.0,
                 height_offset=(-5, 5),
                 intensity_range=(0, 1),
                 color_value='random',
                 class_label='ignore',
                 p=0.3):
        """
        添加局部噪点簇（模拟局部测量误差、多路径反射等）
        
        与 AddExtremeOutliers 的区别：
        - AddExtremeOutliers: 全局随机分布的极端噪点
        - AddLocalNoiseClusters: 局部聚集的噪点簇（更真实）
        
        应用场景：
        - 🏢 建筑物玻璃反射：产生局部聚集的假点
        - 🌲 植被遮挡：树叶间隙产生的噪点簇
        - 📡 多路径干扰：特定位置的系统误差
        - 💧 水面反射：水体附近的噪点
        
        Args:
            num_clusters: 噪点簇的数量
            points_per_cluster: 每个簇的点数范围 (min, max)
            cluster_radius: 簇的半径（米）
            height_offset: 噪点相对于簇中心的高度偏移范围 (min, max)
            intensity_range: 噪点强度范围
            color_value: 噪点颜色（'random', 'inherit', 或 RGB tuple）
            class_label: 噪点分类标签（None, int, 'ignore'）
            p: 应用概率
        """
        self.num_clusters = num_clusters
        self.points_per_cluster = points_per_cluster
        self.cluster_radius = cluster_radius
        self.height_offset = height_offset
        self.intensity_range = intensity_range
        self.color_value = color_value
        self.class_label = class_label
        self.p = p

    def __call__(self, data_dict):
        if np.random.rand() > self.p:
            return data_dict
        
        if "coord" not in data_dict:
            return data_dict
        
        coord = data_dict["coord"]
        n_points = len(coord)
        
        if n_points < 10:
            return data_dict
        
        # 随机选择簇中心（从现有点中选择）
        cluster_centers = coord[
            np.random.choice(n_points, min(self.num_clusters, n_points), replace=False)
        ]
        
        all_outlier_coords = []
        
        for center in cluster_centers:
            # 每个簇的点数
            n_cluster = np.random.randint(
                self.points_per_cluster[0], self.points_per_cluster[1] + 1
            )
            
            # 在球形区域内生成点
            # 使用球坐标系：均匀分布
            theta = np.random.uniform(0, 2 * np.pi, n_cluster)
            phi = np.random.uniform(0, np.pi, n_cluster)
            r = np.random.uniform(0, self.cluster_radius, n_cluster)
            
            # 转换为笛卡尔坐标
            x = center[0] + r * np.sin(phi) * np.cos(theta)
            y = center[1] + r * np.sin(phi) * np.sin(theta)
            z_base = center[2] + r * np.cos(phi)
            
            # 添加高度偏移
            z_offset = np.random.uniform(
                self.height_offset[0], self.height_offset[1], n_cluster
            )
            z = z_base + z_offset
            
            cluster_coords = np.column_stack([x, y, z])
            all_outlier_coords.append(cluster_coords)
        
        if len(all_outlier_coords) == 0:
            return data_dict
        
        outlier_coord = np.vstack(all_outlier_coords).astype(coord.dtype)
        n_outliers = len(outlier_coord)
        
        # 添加噪点到坐标
        data_dict["coord"] = np.vstack([coord, outlier_coord])
        
        # 处理其他属性（与 AddExtremeOutliers 类似）
        if "intensity" in data_dict:
            outlier_intensity = np.random.uniform(
                self.intensity_range[0], self.intensity_range[1], n_outliers
            ).astype(data_dict["intensity"].dtype)
            data_dict["intensity"] = np.concatenate([
                data_dict["intensity"], outlier_intensity
            ])
        
        if "color" in data_dict:
            if self.color_value == 'random':
                outlier_color = np.random.uniform(
                    0, 255, (n_outliers, 3)
                ).astype(data_dict["color"].dtype)
            elif self.color_value == 'inherit':
                random_indices = np.random.choice(n_points, n_outliers)
                outlier_color = data_dict["color"][random_indices].copy()
            else:
                outlier_color = np.tile(
                    np.array(self.color_value, dtype=data_dict["color"].dtype),
                    (n_outliers, 1)
                )
            data_dict["color"] = np.vstack([data_dict["color"], outlier_color])
        
        if "h_norm" in data_dict:
            # 简化计算：使用原始点云最小 Z 作为地面
            ground_z = coord[:, 2].min()
            outlier_h_norm = (outlier_coord[:, 2] - ground_z).astype(
                data_dict["h_norm"].dtype
            )
            data_dict["h_norm"] = np.concatenate([
                data_dict["h_norm"], outlier_h_norm
            ])
        
        if "normal" in data_dict:
            outlier_normal = np.random.randn(n_outliers, 3).astype(
                data_dict["normal"].dtype
            )
            norms = np.linalg.norm(outlier_normal, axis=1, keepdims=True)
            outlier_normal = outlier_normal / (norms + 1e-8)
            data_dict["normal"] = np.vstack([data_dict["normal"], outlier_normal])
        
        if "echo" in data_dict:
            outlier_echo = np.ones((n_outliers, 2), dtype=data_dict["echo"].dtype)
            data_dict["echo"] = np.vstack([data_dict["echo"], outlier_echo])
        
        if "class" in data_dict:
            if self.class_label is None:
                random_indices = np.random.choice(n_points, n_outliers)
                outlier_class = data_dict["class"][random_indices].copy()
            elif self.class_label == 'ignore':
                outlier_class = np.full(n_outliers, -1, dtype=data_dict["class"].dtype)
            else:
                outlier_class = np.full(
                    n_outliers, self.class_label, dtype=data_dict["class"].dtype
                )
            data_dict["class"] = np.concatenate([data_dict["class"], outlier_class])
        
        return data_dict