import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import utils  # 引用项目中的工具类

class REBlurVoxelH5Dataset(Dataset):
    """
    针对已处理好体素网格的 H5 文件的数据加载器
    结构: /images/image000000000, /voxels/voxel000000000 ...
    """
    def __init__(self, data_dir, args, is_train=True):
        self.args = args
        self.is_train = is_train
        self.h5_files = sorted(glob.glob(os.path.join(data_dir, '*.h5')))
        self.samples = []
        
        # 建立全局索引 (file_path, frame_index)
        for h5_path in self.h5_files:
            with h5py.File(h5_path, 'r') as f:
                if 'images' not in f: continue
                num_frames = len(f['images'].keys())
                for i in range(num_frames):
                    self.samples.append((h5_path, i))
        
        print(f"[{'Train' if is_train else 'Val'}] 已加载 {len(self.h5_files)} 个H5文件，共 {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        h5_path, frame_idx = self.samples[idx]
        
        with h5py.File(h5_path, 'r') as f:
            # 匹配 9 位数字命名规则
            img_key = f"image{frame_idx:09d}"
            voxel_key = f"voxel{frame_idx:09d}"
            
            # 读取数据并转换为 float32
            blur_img = np.array(f['images'][img_key]).astype(np.float32)
            sharp_img = np.array(f['sharp_images'][img_key]).astype(np.float32)
            voxel = np.array(f['voxels'][voxel_key]).astype(np.float32)
            
            # 如果图像是 0-255 范围，进行归一化 (参考 dataset_REBlur.py)
            if blur_img.max() > 1.0: blur_img /= 255.0
            if sharp_img.max() > 1.0: sharp_img /= 255.0

        if self.is_train:
            # 使用项目中统一的增强处理（裁剪、翻转等）
            # 注意：此函数会处理维度对齐并返回 Tensor
            blur_img, voxel, sharp_img = utils.image_proess(
                blur_img, voxel, sharp_img, self.args.TRAINING.TRAIN_PS, self.args
            )
        else:
            # 验证模式：直接转 Tensor
            blur_img = torch.from_numpy(blur_img)
            voxel = torch.from_numpy(voxel)
            sharp_img = torch.from_numpy(sharp_img)
            
        return blur_img, voxel, sharp_img