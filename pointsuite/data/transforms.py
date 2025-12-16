"""
3D ç¹äºæ°æ®å¢å¼ºåæ¢æ¨¡å
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


# ââââ éç¨æä½ ââââ
# ç»åå¤ä¸ªåæ¢
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


# ç´¢å¼æä½
def index_operator(data_dict, index, duplicate=False):
    # å¯¹ "index_valid_keys" ä¸­çé®æ§è¡ç´¢å¼éæ©æä½
    # å¯å¨éç½®ä¸­éè¿ "Update" åæ¢èªå®ä¹è¿äºé®
    # å¯¹ data_dict ä¸­çé®è¿è¡ç´¢å¼æä½
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
            "indices",  # ð¥ ä¿®å¤ï¼å¨test/predictæ¨¡å¼ä¸ï¼indiceséè¦ä¸å¶ä»å­æ®µåæ­¥è¿æ»¤
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


# æ¶éæå® key çæ°æ®ï¼æ¯æ offset åç¹å¾æ¼æ¥
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
        if isinstance(data_dict, Sequence):
            return [self(item) for item in data_dict]
            
        data = dict()
        if isinstance(self.keys, str):
            self.keys = [self.keys]
        for key in self.keys:
            # 直接传递数据（可能是 numpy 或 tensor）
            data[key] = data_dict[key]
        for key, value in self.offset_key.items():
            # offset 创建为 Tensor
            data[key] = torch.tensor([data_dict[value].shape[0]])
            
        # 🔥 兼容 list 类型的 feat_keys (自动转换为 {'feat': list})
        feat_keys = self.feat_keys
        if isinstance(feat_keys, list):
            feat_keys = {'feat': feat_keys}
            
        for name, keys in feat_keys.items():
            name = name.replace("_keys", "")
            assert isinstance(keys, Sequence)
            # data[name] = torch.cat([data_dict[key].float() for key in keys], dim=1)
            tensors = []
            for key in keys:
                # åè½¬æ¢ä¸º Tensorï¼å¦æè¿ä¸æ¯ï¼
                if isinstance(data_dict[key], np.ndarray):
                    tensor = torch.from_numpy(data_dict[key]).float()
                else:
                    tensor = data_dict[key].float()
                
                if tensor.dim() == 1:  # å¦ææ¯ [n]ï¼æ©å±æ [n, 1]
                    tensor = tensor.unsqueeze(1)
                tensors.append(tensor)
            data[name] = torch.cat(tensors, dim=1)  # [n, c + m]ï¼m æ¯é¢å¤æ¼æ¥ç 1D å¼ éæ°éï¼
        return data
    

# æ´æ°æå®çé®
class Update(object):
    def __init__(self, keys_dict=None):
        if keys_dict is None:
            keys_dict = dict()
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            data_dict[key] = value
        return data_dict
    
     
# å°æ°æ®è½¬æ¢ä¸ºå¼ é
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
        

# ââââ åæ åæ¢ ââââ
# åæ æ ååï¼åå»è´¨å¿å¹¶ç¼©æ¾å°åä½ç
class NormalizeCoord(object):
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            # modified from pointnet2
            centroid = np.mean(data_dict["coord"], axis=0)
            data_dict["coord"] -= centroid
            m = np.max(np.sqrt(np.sum(data_dict["coord"] ** 2, axis=1)))
            data_dict["coord"] = data_dict["coord"] / m
        return data_dict
    

# åå»åå¼é¤ä»¥æ åå·®ï¼æ ååï¼
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
            # é¿åé¤ä»¥0
            std[std == 0] = 1
            data_dict["coord"] = (data_dict["coord"] - mean) / std
        return data_dict


# åå»æå°å¼é¤ä»¥æå¤§æå°å¼ä¹å·®ï¼MinMaxå½ä¸åï¼
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
            # è®¡ç®èå´ï¼é¿åé¤ä»¥0
            ranges = max_vals - min_vals
            ranges[ranges == 0] = 1
            data_dict["coord"] = (data_dict["coord"] - min_vals) / ranges
        return data_dict


# åæ åç§»ï¼æå°å¼ï¼
class PositiveShift(object):
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            coord_min = np.min(data_dict["coord"], 0)
            data_dict["coord"] -= coord_min
        return data_dict


# åæ åç§»ï¼ä¸­å¿ï¼
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
    

# åæ åç§»ï¼è´¨å¿ï¼
class CentroidShift(object):
    def __init__(self, apply_z=True):
        self.apply_z = apply_z

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            centroid = np.mean(data_dict["coord"], axis=0)
            if not self.apply_z:
                centroid[2] = 0
            data_dict["coord"] -= centroid
        return data_dict


# éæºåç§»
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


# éæºä¸¢å¼
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


# éæºæè½¬
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


# éæºæè½¬å°ç¹å®è§åº¦
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


# éæºç¼©æ¾
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


# éæºç¿»è½¬
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


# éæºæå¨
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


# é«æ¯æå¨
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
    

# é¡ºåºæä¹±
class ShufflePoint(object):
    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        shuffle_index = np.arange(data_dict["coord"].shape[0])
        np.random.shuffle(shuffle_index)
        data_dict = index_operator(data_dict, shuffle_index)
        return data_dict


# ââââ å¼ºåº¦åæ¢ ââââ
# Intensity èªå¨æ£æµå¹¶å½ä¸å
class AutoNormalizeIntensity(object):
    def __init__(self, target_range=(0, 1)):
        """
        èªå¨æ£æµ intensity ä½æ°å¹¶å½ä¸åå°ç®æ èå´
        
        æ£æµé»è¾ï¼
        - å¦æ max <= 1.0: è®¤ä¸ºå·²å½ä¸åï¼ä¸å¤ç
        - å¦æ max <= 255: è®¤ä¸ºæ¯ 8 ä½ï¼é¤ä»¥ 255
        - å¦æ max <= 65535: è®¤ä¸ºæ¯ 16 ä½ï¼é¤ä»¥ 65535
        - å¦å: ä½¿ç¨å®éç max-min èå´å½ä¸å
        
        Args:
            target_range: ç®æ èå´ (min, max)ï¼é»è®¤ (0, 1)
        """
        self.target_range = target_range

    def __call__(self, data_dict):
        if "intensity" in data_dict.keys():
            intensity = data_dict["intensity"].astype(np.float32)
            
            # æ£æµå½åèå´
            i_min = intensity.min()
            i_max = intensity.max()
            
            # èªå¨æ£æµä½æ°å¹¶å½ä¸å
            if i_max <= 1.0:
                # å·²ç»å½ä¸åï¼å¯è½éè¦è°æ´èå´
                if self.target_range != (0, 1):
                    # ä» [0, 1] æ å°å° target_range
                    target_min, target_max = self.target_range
                    intensity = intensity * (target_max - target_min) + target_min
            elif i_max <= 255:
                # 8 ä½
                intensity = intensity / 255.0
                if self.target_range != (0, 1):
                    target_min, target_max = self.target_range
                    intensity = intensity * (target_max - target_min) + target_min
            elif i_max <= 65535:
                # 16 ä½
                intensity = intensity / 65535.0
                if self.target_range != (0, 1):
                    target_min, target_max = self.target_range
                    intensity = intensity * (target_max - target_min) + target_min
            else:
                # æªç¥èå´ï¼ä½¿ç¨ min-max å½ä¸å
                if i_max > i_min:
                    intensity = (intensity - i_min) / (i_max - i_min)
                    if self.target_range != (0, 1):
                        target_min, target_max = self.target_range
                        intensity = intensity * (target_max - target_min) + target_min
            
            data_dict["intensity"] = intensity
        return data_dict


# Intensity å½ä¸åï¼æå®ä½æ°ï¼
class NormalizeIntensity(object):
    def __init__(self, max_value=65535.0):
        """
        ä½¿ç¨æå®çæå¤§å¼å½ä¸å intensity å° [0, 1]
        
        Args:
            max_value: æå¤§å¯è½çå¼ºåº¦å¼ï¼å¦ 65535 è¡¨ç¤º 16 ä½ï¼255 è¡¨ç¤º 8 ä½ï¼
        """
        self.max_value = max_value

    def __call__(self, data_dict):
        if "intensity" in data_dict.keys():
            data_dict["intensity"] = data_dict["intensity"].astype(np.float32) / self.max_value
        return data_dict


# Intensity éæºç¼©æ¾
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


# Intensity éæºåç§»
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


# Intensity éæºåªå£°
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


# Intensity éæºä¸¢å¼ï¼ç½®ä¸º0ï¼
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


# Intensity Gamma åæ¢
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


# Intensity æ ååï¼ååå¼é¤æ¹å·®ï¼
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


# Intensity MinMax å½ä¸å
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

    
# ââââ é¢è²åæ¢ ââââ
# é¢è²èªå¨æ£æµå¹¶å½ä¸å
class AutoNormalizeColor(object):
    def __init__(self, target_range=(0, 255)):
        """
        èªå¨æ£æµ color ä½æ°å¹¶å½ä¸åå°ç®æ èå´
        
        æ£æµé»è¾ï¼
        - å¦æ max <= 1.0: è®¤ä¸ºå·²å½ä¸åå° [0, 1]ï¼æ å°å° target_range
        - å¦æ max <= 255: è®¤ä¸ºæ¯ 8 ä½ï¼å·²å¨æ­£ç¡®èå´
        - å¦æ max <= 65535: è®¤ä¸ºæ¯ 16 ä½ï¼è½¬æ¢å° 8 ä½ [0, 255]
        - å¦å: ä½¿ç¨å®éç max-min èå´å½ä¸å
        
        æ³¨æï¼å¤§é¨åé¢è²å¢å¼ºï¼ChromaticJitter ç­ï¼ææ [0, 255] èå´
        
        Args:
            target_range: ç®æ èå´ï¼é»è®¤ (0, 255) ç¨äºé¢è²å¢å¼º
        """
        self.target_range = target_range

    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            color = data_dict["color"].astype(np.float32)
            
            # æ£æµå½åèå´ï¼ä½¿ç¨ææééçæå¤§å¼ï¼
            c_min = color.min()
            c_max = color.max()
            
            target_min, target_max = self.target_range
            
            # èªå¨æ£æµä½æ°å¹¶å½ä¸å
            if c_max <= 1.0:
                # å·²ç»å½ä¸åå° [0, 1]ï¼æ å°å° target_range
                color = color * (target_max - target_min) + target_min
            elif c_max <= 255:
                # 8 ä½ï¼å·²å¨ [0, 255] èå´
                if self.target_range != (0, 255):
                    # éè¦æ å°å°å¶ä»èå´
                    color = (color / 255.0) * (target_max - target_min) + target_min
            elif c_max <= 65535:
                # 16 ä½ï¼è½¬æ¢å°ç®æ èå´
                color = (color / 65535.0) * (target_max - target_min) + target_min
            else:
                # æªç¥èå´ï¼ä½¿ç¨ min-max å½ä¸å
                if c_max > c_min:
                    color = (color - c_min) / (c_max - c_min)
                    color = color * (target_max - target_min) + target_min
            
            data_dict["color"] = color
        return data_dict


# é¢è²å½ä¸åï¼æå®ä½æ°ï¼
class NormalizeColor(object):
    def __init__(self, source_bits=16, target_range=(0, 255)):
        """
        ä½¿ç¨æå®çä½æ°å½ä¸å color
        
        Args:
            source_bits: æºæ°æ®ä½æ°ï¼8 æ 16ï¼
            target_range: ç®æ èå´ï¼é»è®¤ (0, 255)
        """
        self.source_bits = source_bits
        self.target_range = target_range
        self.source_max = (2 ** source_bits) - 1

    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            color = data_dict["color"].astype(np.float32)
            
            # å½ä¸åå° [0, 1]
            color = color / self.source_max
            
            # æ å°å°ç®æ èå´
            target_min, target_max = self.target_range
            color = color * (target_max - target_min) + target_min
            
            data_dict["color"] = color
        return data_dict


# é¢è²å¯¹æ¯åº¦å¢å¼º
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


# é¢è²éæºå¹³ç§»
class ChromaticTranslation(object):
    def __init__(self, p=0.95, ratio=0.05):
        self.p = p
        self.ratio = ratio

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.ratio
            data_dict["color"][:, :3] = np.clip(tr + data_dict["color"][:, :3], 0, 255)
        return data_dict


# é¢è²éæºæå¨
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


# éæºé¢è²ç°åº¦å
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


# éæºé¢è²æå¨
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


# éæºé¢è²é¥±ååº¦
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


# éæºé¢è²ä¸¢å¼
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


# å¼¹æ§å¤±çï¼æ¨¡æèªç¶åå½¢
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


# ââââ å½ä¸åé«ç¨ï¼h_normï¼åæ¢ ââââ
# å½ä¸åé«ç¨èªå¨å½ä¸åï¼å¯éè£åªï¼
class AutoNormalizeHNorm(object):
    def __init__(self, clip_range=None):
        """
        èªå¨å½ä¸å h_normï¼å¯éè£åªå¼å¸¸å¼ï¼
        
        é»è®¤è¡ä¸ºï¼clip_range=Noneï¼ï¼
        - ä¸è£åªä»»ä½å¼ï¼ä¿çè´å¼åæå¤§å¼
        - è´å¼å¯è½ä»£è¡¨å°ä¸ç»æï¼å°ä¸å®¤ãé§éãåæ´ï¼
        - æå¤§å¼å¯è½ä»£è¡¨çå®é«å±å»ºç­æåªå£°
        - è®©æ¨¡åå­¦ä¹ è¯å«åå¤çå¼å¸¸å¼ï¼å¢å¼ºé²æ£æ§
        
        å¯éè£åªï¼clip_range=(min, max)ï¼ï¼
        - å¦ (0, 50) å°é«ç¨éå¶å¨ 0-50mï¼æé¤ææ¾å¼å¸¸å¼ï¼
        - å¦ (-5, 100) ä¿çåççå°ä¸åé«ç©ºèå´
        
        Args:
            clip_range: è£åªèå´ (min, max)ï¼é»è®¤ Noneï¼ä¸è£åªï¼
                       None: ä¸è£åªï¼ä¿çææå¼ï¼æ¨èï¼
                       (min, max): è£åªå°æå®èå´
                       (None, max): åªè£åªä¸ç
                       (min, None): åªè£åªä¸ç
        """
        self.clip_range = clip_range

    def __call__(self, data_dict):
        if "h_norm" in data_dict.keys():
            h_norm = data_dict["h_norm"].astype(np.float32)
            
            # å¯éè£åªå¼å¸¸å¼
            if self.clip_range is not None:
                if self.clip_range[0] is not None:
                    h_norm = np.maximum(h_norm, self.clip_range[0])
                if self.clip_range[1] is not None:
                    h_norm = np.minimum(h_norm, self.clip_range[1])
            
            data_dict["h_norm"] = h_norm
        return data_dict


# å½ä¸åé«ç¨æ åå
class StandardNormalizeHNorm(object):
    def __init__(self, mean=None, std=None):
        """
        æ åå h_normï¼Z-score å½ä¸åï¼
        
        éç¨äºéè¦é¶åå¼ãåä½æ¹å·®è¾å¥çæ¨¡å
        
        Args:
            mean: åå¼ï¼å¦æä¸º None åä»æ°æ®è®¡ç®
            std: æ åå·®ï¼å¦æä¸º None åä»æ°æ®è®¡ç®
        """
        self.mean = mean
        self.std = std

    def __call__(self, data_dict):
        if "h_norm" in data_dict.keys():
            h_norm = data_dict["h_norm"].astype(np.float32)
            
            mean = self.mean if self.mean is not None else h_norm.mean()
            std = self.std if self.std is not None else h_norm.std()
            
            # é¿åé¤é¶
            if std == 0:
                std = 1.0
            
            data_dict["h_norm"] = ((h_norm - mean) / std).astype(np.float32)
        return data_dict


# å½ä¸åé«ç¨éæºç¼©æ¾
class RandomHNormScale(object):
    def __init__(self, scale=(0.9, 1.1), p=0.5):
        """
        éæºç¼©æ¾ h_norm
        
        æ¨¡æä¸åçå°é¢è¯å«ç²¾åº¦æé«ç¨æµéè¯¯å·®
        
        Args:
            scale: ç¼©æ¾èå´ (min_scale, max_scale)
            p: åºç¨æ¦ç
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


# å½ä¸åé«ç¨éæºåªå£°
class RandomHNormNoise(object):
    def __init__(self, sigma=0.1, p=0.5):
        """
        ä¸º h_norm æ·»å éæºé«æ¯åªå£°
        
        æ¨¡æå°é¢é«ç¨ä¼°è®¡çå±é¨è¯¯å·®
        
        Args:
            sigma: é«æ¯åªå£°çæ åå·®ï¼åä½ä¸ h_norm ç¸åï¼éå¸¸æ¯ç±³ï¼
            p: åºç¨æ¦ç
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


# å½ä¸åé«ç¨å¯¹æ°åæ¢
class LogTransformHNorm(object):
    def __init__(self, epsilon=1e-6):
        """
        å¯¹ h_norm è¿è¡å¯¹æ°åæ¢
        
        ç¨äºå¤çé«åº¦èå´å¾å¤§çåºæ¯ï¼å¦å»ºç­ç©åå°é¢ï¼
        ä½¿æ¨¡åå¯¹ä¸åé«åº¦å°ºåº¦æ´ææ
        
        Args:
            epsilon: é¿å log(0) çå°å¸¸æ°
        """
        self.epsilon = epsilon

    def __call__(self, data_dict):
        if "h_norm" in data_dict.keys():
            h_norm = data_dict["h_norm"].astype(np.float32)
            # ç¡®ä¿éè´
            h_norm = np.maximum(h_norm, 0)
            # å¯¹æ°åæ¢
            data_dict["h_norm"] = np.log(h_norm + self.epsilon).astype(np.float32)
        return data_dict


# å½ä¸åé«ç¨åæ¡¶ç¼ç 
class BinHNorm(object):
    def __init__(self, bins=10, range=(0, 20)):
        """
        å° h_norm ç¦»æ£åä¸ºæ¡¶ï¼binsï¼
        
        å°è¿ç»­çé«åº¦å¼è½¬æ¢ä¸ºç¦»æ£çé«åº¦ç­çº§
        
        Args:
            bins: æ¡¶çæ°é
            range: é«åº¦èå´ (min, max)
        """
        self.bins = bins
        self.range = range

    def __call__(self, data_dict):
        if "h_norm" in data_dict.keys():
            h_norm = data_dict["h_norm"].astype(np.float32)
            
            # ä½¿ç¨ numpy ç digitize è¿è¡åæ¡¶
            bin_edges = np.linspace(self.range[0], self.range[1], self.bins + 1)
            binned = np.digitize(h_norm, bin_edges) - 1
            
            # è£åªå° [0, bins-1]
            binned = np.clip(binned, 0, self.bins - 1)
            
            # è½¬æ¢ä¸º floatï¼å½ä¸åå° [0, 1]ï¼
            data_dict["h_norm"] = (binned / (self.bins - 1)).astype(np.float32)
        return data_dict


# ââââ åªç¹æ³¨å¥å¢å¼º ââââ
# æ·»å æç«¯é«åº¦åªç¹
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
        æ·»å æç«¯é«åº¦åªç¹ï¼æ¨¡æå¤§æ°åªå£°ãå¤è·¯å¾åå°ç­ï¼
        
        åªç¹æ¥æºæ¨¡æï¼
        - ð©ï¸ å¤§æ°åªå£°ï¼é£é¸ãäºãç°å°ï¼é«ç©ºåªç¹ï¼
        - ð» å°é¢åå°ï¼æ°´é¢ãç»çåå°ï¼ä½ç©º/å°ä¸åªç¹ï¼
        - ð¡ å¤è·¯å¾åå°ï¼å»ºç­ç©ãéå±è¡¨é¢åå°ï¼éæºé«åº¦ï¼
        - ð³ æ¤è¢«é®æ¡ï¼æ å¶é´éçä¼ªç¹ï¼ä¸­ç­é«åº¦ï¼
        
        åªç¹å±æ§è®¾ç½®ç­ç¥ï¼
        - coord: å¨ç°æç¹äºç XY èå´åéæºåå¸ï¼Z ä¸ºæç«¯å¼
        - intensity: éå¸¸è¾å¼±ï¼å¤§æ°åªå£°ï¼æå¾å¼ºï¼åå°ï¼
        - color: ç°è²ï¼æªç¥ï¼æéæºè²
        - h_norm: æ ¹æ® Z åå°é¢é«ç¨è®¡ç®ï¼æè®¾ä¸ºæç«¯å¼ï¼
        - class: åªå£°ç±»å«ï¼å¯éç½®ï¼å¦ 0=æªåç±»ï¼
        
        Args:
            num_outliers: åºå®åªç¹æ°éï¼å¦ææå®åå¿½ç¥ ratio
            ratio: åªç¹æ°éå æ»ç¹æ°çæ¯ä¾ï¼é»è®¤ 0.01ï¼1%ï¼
            height_range: åªç¹é«åº¦èå´ (z_min, z_max)ï¼é»è®¤ (-10, 100) ç±³
                         ç¸å¯¹äºåå§ Z åæ ï¼ä¸æ¯ h_norm
            height_mode: é«åº¦åå¸æ¨¡å¼
                - 'uniform': åååå¸å¨ height_range
                - 'bimodal': åå³°åå¸ï¼é«ç©º+ä½ç©ºï¼
                - 'high': åªå¨é«ç©ºï¼æ¨¡æé£é¸ãäºï¼
                - 'low': åªå¨ä½ç©º/å°ä¸ï¼æ¨¡æåå°ï¼
            intensity_range: åªç¹å¼ºåº¦èå´ (min, max)ï¼é»è®¤ (0, 1)
            color_value: åªç¹é¢è²
                - tuple (R, G, B): åºå®é¢è²ï¼å¦ (128, 128, 128) ç°è²
                - 'random': éæºé¢è²
                - 'inherit': ä»æè¿ççå®ç¹ç»§æ¿é¢è²
            class_label: åªç¹çåç±»æ ç­¾
                - None: ä»æè¿ççå®ç¹ç»§æ¿
                - int: åºå®æ ç­¾ï¼å¦ 0=æªåç±», -1=åªå£°ï¼
                - 'ignore': ä½¿ç¨ ignore_labelï¼éå¸¸æ¯ -1ï¼
            p: åºç¨æ¦ç
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
        
        # è®¡ç®åªç¹æ°é
        if self.num_outliers is not None:
            n_outliers = self.num_outliers
        else:
            n_outliers = max(1, int(n_points * self.ratio))
        
        # è·ååå§ç¹äºç XY èå´
        x_min, y_min, z_min = coord.min(axis=0)
        x_max, y_max, z_max = coord.max(axis=0)
        
        # çæåªç¹åæ 
        outlier_xy = np.random.rand(n_outliers, 2)
        outlier_xy[:, 0] = outlier_xy[:, 0] * (x_max - x_min) + x_min
        outlier_xy[:, 1] = outlier_xy[:, 1] * (y_max - y_min) + y_min
        
        # æ ¹æ®æ¨¡å¼çæé«åº¦
        if self.height_mode == 'uniform':
            # åååå¸
            outlier_z = np.random.uniform(
                self.height_range[0], self.height_range[1], n_outliers
            )
        elif self.height_mode == 'bimodal':
            # åå³°åå¸ï¼50% é«ç©ºï¼50% ä½ç©º
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
            # åªå¨é«ç©º
            outlier_z = np.random.uniform(
                max(self.height_range[0], z_max), self.height_range[1], n_outliers
            )
        elif self.height_mode == 'low':
            # åªå¨ä½ç©º/å°ä¸
            outlier_z = np.random.uniform(
                self.height_range[0], min(self.height_range[1], z_min), n_outliers
            )
        else:
            raise ValueError(f"Unknown height_mode: {self.height_mode}")
        
        # ç»ååªç¹åæ 
        outlier_coord = np.column_stack([outlier_xy, outlier_z]).astype(coord.dtype)
        
        # æ·»å åªç¹å°åæ 
        data_dict["coord"] = np.vstack([coord, outlier_coord])
        
        # å¤çå¶ä»å±æ§
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
                # éæºé¢è²
                outlier_color = np.random.uniform(
                    0, 255, (n_outliers, 3)
                ).astype(data_dict["color"].dtype)
            elif self.color_value == 'inherit':
                # ä»æè¿ççå®ç¹ç»§æ¿ï¼ä½¿ç¨ç®åçéæºéæ ·ï¼
                random_indices = np.random.choice(n_points, n_outliers)
                outlier_color = data_dict["color"][random_indices].copy()
            else:
                # åºå®é¢è²
                outlier_color = np.tile(
                    np.array(self.color_value, dtype=data_dict["color"].dtype),
                    (n_outliers, 1)
                )
            data_dict["color"] = np.vstack([data_dict["color"], outlier_color])
        
        # 3. h_norm
        if "h_norm" in data_dict:
            # è®¡ç®åªç¹ç h_norm
            # ç®åï¼åè®¾å°é¢é«ç¨ä¸ºåå§ç¹äºçæå° Z
            ground_z = z_min
            outlier_h_norm = (outlier_z - ground_z).astype(data_dict["h_norm"].dtype)
            data_dict["h_norm"] = np.concatenate([
                data_dict["h_norm"], outlier_h_norm
            ])
        
        # 4. Normal
        if "normal" in data_dict:
            # åªç¹çæ³åéï¼éæºæ¹åï¼æ¨¡æåªå£°ï¼
            outlier_normal = np.random.randn(n_outliers, 3).astype(
                data_dict["normal"].dtype
            )
            # å½ä¸å
            norms = np.linalg.norm(outlier_normal, axis=1, keepdims=True)
            outlier_normal = outlier_normal / (norms + 1e-8)
            data_dict["normal"] = np.vstack([data_dict["normal"], outlier_normal])
        
        # 5. Echo
        if "echo" in data_dict:
            # åªç¹éå¸¸æ¯åæ¬¡åæ³¢
            outlier_echo = np.ones((n_outliers, 2), dtype=data_dict["echo"].dtype)
            # è®¾ä¸ºé¦æ¬¡ä¸æ«æ¬¡åæ³¢ï¼åæ¬¡åæ³¢çç¹å¾ï¼
            data_dict["echo"] = np.vstack([data_dict["echo"], outlier_echo])
        
        # 6. Classification
        if "class" in data_dict:
            if self.class_label is None:
                # ä»æè¿ççå®ç¹ç»§æ¿ï¼éæºéæ ·ï¼
                random_indices = np.random.choice(n_points, n_outliers)
                outlier_class = data_dict["class"][random_indices].copy()
            elif self.class_label == 'ignore':
                # ä½¿ç¨ ignore_labelï¼éå¸¸å¨ dataset ä¸­å®ä¹ï¼
                outlier_class = np.full(n_outliers, -1, dtype=data_dict["class"].dtype)
            else:
                # åºå®æ ç­¾
                outlier_class = np.full(
                    n_outliers, self.class_label, dtype=data_dict["class"].dtype
                )
            data_dict["class"] = np.concatenate([data_dict["class"], outlier_class])
        
        return data_dict


# æ·»å å±é¨åªç¹ç°
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
        æ·»å å±é¨åªç¹ç°ï¼æ¨¡æå±é¨æµéè¯¯å·®ãå¤è·¯å¾åå°ç­ï¼
        
        ä¸ AddExtremeOutliers çåºå«ï¼
        - AddExtremeOutliers: å¨å±éæºåå¸çæç«¯åªç¹
        - AddLocalNoiseClusters: å±é¨èéçåªç¹ç°ï¼æ´çå®ï¼
        
        åºç¨åºæ¯ï¼
        - ð¢ å»ºç­ç©ç»çåå°ï¼äº§çå±é¨èéçåç¹
        - ð² æ¤è¢«é®æ¡ï¼æ å¶é´éäº§ççåªç¹ç°
        - ð¡ å¤è·¯å¾å¹²æ°ï¼ç¹å®ä½ç½®çç³»ç»è¯¯å·®
        - ð§ æ°´é¢åå°ï¼æ°´ä½éè¿çåªç¹
        
        Args:
            num_clusters: åªç¹ç°çæ°é
            points_per_cluster: æ¯ä¸ªç°çç¹æ°èå´ (min, max)
            cluster_radius: ç°çåå¾ï¼ç±³ï¼
            height_offset: åªç¹ç¸å¯¹äºç°ä¸­å¿çé«åº¦åç§»èå´ (min, max)
            intensity_range: åªç¹å¼ºåº¦èå´
            color_value: åªç¹é¢è²ï¼'random', 'inherit', æ RGB tupleï¼
            class_label: åªç¹åç±»æ ç­¾ï¼None, int, 'ignore'ï¼
            p: åºç¨æ¦ç
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
        
        # éæºéæ©ç°ä¸­å¿ï¼ä»ç°æç¹ä¸­éæ©ï¼
        cluster_centers = coord[
            np.random.choice(n_points, min(self.num_clusters, n_points), replace=False)
        ]
        
        all_outlier_coords = []
        
        for center in cluster_centers:
            # æ¯ä¸ªç°çç¹æ°
            n_cluster = np.random.randint(
                self.points_per_cluster[0], self.points_per_cluster[1] + 1
            )
            
            # å¨çå½¢åºååçæç¹
            # ä½¿ç¨çåæ ç³»ï¼åååå¸
            theta = np.random.uniform(0, 2 * np.pi, n_cluster)
            phi = np.random.uniform(0, np.pi, n_cluster)
            r = np.random.uniform(0, self.cluster_radius, n_cluster)
            
            # è½¬æ¢ä¸ºç¬å¡å°åæ 
            x = center[0] + r * np.sin(phi) * np.cos(theta)
            y = center[1] + r * np.sin(phi) * np.sin(theta)
            z_base = center[2] + r * np.cos(phi)
            
            # æ·»å é«åº¦åç§»
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
        
        # æ·»å åªç¹å°åæ 
        data_dict["coord"] = np.vstack([coord, outlier_coord])
        
        # å¤çå¶ä»å±æ§ï¼ä¸ AddExtremeOutliers ç±»ä¼¼ï¼
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
            # ç®åè®¡ç®ï¼ä½¿ç¨åå§ç¹äºæå° Z ä½ä¸ºå°é¢
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
    

# ââââ Grid Sample ââââ
class GridSample(object):
    def __init__(
        self,
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        return_inverse=False,
        return_grid_coord=False,
        return_min_coord=False,
        return_displacement=False,
        project_displacement=False,
        max_test_loops=30,
    ):
        self.grid_size = grid_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.project_displacement = project_displacement

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        # è®¡ç®è§åååæ 
        self.grid_size=self.grid_size
        scaled_coord = data_dict["coord"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        # è®¡ç®æå°ç½æ ¼åæ ï¼å½ä¸å
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * np.array(self.grid_size)
        # è·åè§ååæ åå¸å¼å¹¶æåº
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        # è®¡ç®ç½æ ¼ç´¢å¼åç¹æ°ç»è®¡
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        if self.mode == "train":  # train mode
            # æ ¼ç½ä¸­éæºéæ ·
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx_unique = np.unique(
                    np.append(idx_unique, data_dict["sampled_index"])
                )
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx_unique])[0]
            data_dict = index_operator(data_dict, idx_unique)
            # è¥éè¿åéç´¢å¼ return_inverseï¼è®°å½æ¯ä¸ªç¹å¨åå§æ°æ®ä¸­çå½å±
            if self.return_inverse:
                data_dict["inverse"] = np.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
            # è®°å½ç½æ ¼åæ åæå°åæ 
            if self.return_grid_coord:
                data_dict["grid_coord"] = grid_coord[idx_unique]
                data_dict["index_valid_keys"].append("grid_coord")
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            # ç¹å¨ç½æ ¼åçä½ç½®åæ³çº¿ä¸çè·ç¦»
            if self.return_displacement:
                displacement = (
                    scaled_coord - grid_coord - 0.5
                )  # [0, 1] -> [-0.5, 0.5] displacement to center
                if self.project_displacement:
                    displacement = np.sum(
                        displacement * data_dict["normal"], axis=-1, keepdims=True
                    )
                data_dict["displacement"] = displacement[idx_unique]
                data_dict["index_valid_keys"].append("displacement")
            return data_dict

        elif self.mode == "test":  # test mode
            data_part_list = []
            # å¾ªç¯éæ ·ï¼é¿åéæ¼
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                data_part = index_operator(data_dict, idx_part, duplicate=True)
                data_part["index"] = idx_part
                if self.return_inverse:
                    data_part["inverse"] = np.zeros_like(inverse)
                    data_part["inverse"][idx_sort] = inverse
                if self.return_grid_coord:
                    data_part["grid_coord"] = grid_coord[idx_part]
                    data_dict["index_valid_keys"].append("grid_coord")
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.reshape([1, 3])
                if self.return_displacement:
                    displacement = (
                        scaled_coord - grid_coord - 0.5
                    )  # [0, 1] -> [-0.5, 0.5] displacement to center
                    if self.project_displacement:
                        displacement = np.sum(
                            displacement * data_dict["normal"], axis=-1, keepdims=True
                        )
                    data_dict["displacement"] = displacement[idx_part]
                    data_dict["index_valid_keys"].append("displacement")
                data_part_list.append(data_part)
            return data_part_list
        
        else:
            raise NotImplementedError

    @staticmethod
    # éç¨äºèå´å·²ç¥ï¼å¯éæ ¼ç½
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    # éç¨äºèå´æªç¥æè¾å¤§ï¼ç¨çæ ¼ç½
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr
