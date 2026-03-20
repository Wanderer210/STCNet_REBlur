import os
import cv2
import h5py
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

from config import Config
import utils
from U_model import unet
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR

# 1. 配置文件路径
CONFIG_YAML = 'REBlur_funtine_SCER.yml'
opt = Config(CONFIG_YAML)

# 强制指向你那组处理好的 H5 测试集目录
test_h5_dir = '/home/zy/data/zy/zhaoyue/Datasets/REBlur_EFNet/test'

# GPU 配置
gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
torch.backends.cudnn.benchmark = True

# 2. 专门读取预处理 Voxel H5 的数据集类
class DataLoaderTest_VoxelH5(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(self.h5_path, 'r') as f:
            self.keys = sorted(f['images'].keys())
            self.num_frames = len(self.keys)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        # 计算三帧的索引：过去(p), 当前(c), 未来(f)
        idx_p = max(0, idx - 1)
        idx_c = idx
        idx_f = min(self.num_frames - 1, idx + 1)
        
        indices = [idx_p, idx_c, idx_f]
        
        blur_imgs = []
        voxels = []
        
        with h5py.File(self.h5_path, 'r') as f:
            for i in indices:
                img_key = f"image{i:09d}"
                voxel_key = f"voxel{i:09d}"
                
                img = np.array(f['images'][img_key]).astype(np.float32)
                voxel = np.array(f['voxels'][voxel_key]).astype(np.float32)
                
                if img.max() > 1.0: img /= 255.0
                blur_imgs.append(img)
                voxels.append(voxel)
            
            # 读取当前帧对应的 GT (用于指标计算)
            gt_key = f"image{idx_c:09d}"
            sharp_img = np.array(f['sharp_images'][gt_key]).astype(np.float32)
            if sharp_img.max() > 1.0: sharp_img /= 255.0

        # 将列表转换为 numpy 数组，维度变为 (3, C, H, W)
        blur_imgs = np.stack(blur_imgs, axis=0)
        voxels = np.stack(voxels, axis=0)

        return torch.from_numpy(blur_imgs), torch.from_numpy(voxels), torch.from_numpy(sharp_img)

def main():
    # 结果保存路径
    result_dir = os.path.join(opt.TRAINING.SAVE_DIR, 'results', 'GoPro_on_REBlur_SCER')
    utils.mkdir(result_dir)

    # 【修改】强制指向 GoPro 权重
    model_dir = './checkpoints/models/STCNet' 
    path_chk_rest = os.path.join(model_dir, 'model_best.pth')
    
    ######### Model ###########
    # 根据你的 H5 维度，inChannels_event 设为 6
    model_restoration = unet.Restoration(3, 6, 3, opt)
    model_restoration.cuda()

    print('==> Loading GoPro checkpoint:', path_chk_rest)
    utils.load_checkpoint(model_restoration, path_chk_rest)

    # 多 GPU 支持
    device_ids = [i for i in range(torch.cuda.device_count())]
    if len(device_ids) > 1:
        model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

    model_restoration.eval()

    # 获取所有 H5 测试文件
    h5_files = sorted(glob.glob(os.path.join(test_h5_dir, '*.h5')))
    
    psnr_all = []
    ssim_all = []

    for h5_path in h5_files:
        seq_name = os.path.basename(h5_path).replace('.h5', '')
        out_path = os.path.join(result_dir, seq_name)
        utils.mkdir(out_path)

        test_dataset = DataLoaderTest_VoxelH5(h5_path)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)

        print(f"Testing H5 Sequence: {seq_name}")
        
        psnr_seq = []
        ssim_seq = []

        # main_test_REBlur_SCER.py 中的循环部分
        for ii, (img, voxel, gt) in enumerate(tqdm(test_loader)):
            # 此时 img 维度为 (1, 3, 3, H, W), voxel 维度为 (1, 3, 6, H, W)
            img, voxel, gt = img.cuda(), voxel.cuda(), gt.cuda()

            with torch.no_grad():
                # 模型内部会通过 img.shape 获取 frames=3 并进行循环处理
                restored = model_restoration(img, voxel)

            # 后处理：转回 numpy 并对齐维度 (HWC)
            res = torch.clamp(restored, 0, 1)[0].cpu().numpy().transpose([1, 2, 0])
            tar = gt[0].cpu().numpy().transpose([1, 2, 0])

            # 指标计算
            psnr_val = PSNR(res, tar, data_range=1.0)
            
            # 【修复】兼容性 SSIM 计算
            try:
                # 针对新版本 skimage (0.19+)
                ssim_val = SSIM(res, tar, multichannel=True, data_range=1.0)
            except TypeError:
                # 针对旧版本 skimage (如报错提示设置 multichannel=True)
                ssim_val = SSIM(res, tar, multichannel=True, data_range=1.0)

            psnr_seq.append(psnr_val)
            ssim_seq.append(ssim_val)

            # 保存图像 (RGB -> BGR)
            output = (res * 255.0).astype(np.uint8)
            output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            fname = f"{ii:04d}_psnr{psnr_val:.2f}_ssim{ssim_val:.4f}.png"
            cv2.imwrite(os.path.join(out_path, fname), output_bgr)

        avg_psnr = np.mean(psnr_seq)
        avg_ssim = np.mean(ssim_seq)
        psnr_all.append(avg_psnr)
        ssim_all.append(avg_ssim)
        print(f"Done: {seq_name} | PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f}")

    print('\n' + '='*50)
    print(f"Final Average PSNR: {np.mean(psnr_all):.4f}")
    print(f"Final Average SSIM: {np.mean(ssim_all):.4f}")
    print('='*50)

if __name__ == '__main__':
    main()