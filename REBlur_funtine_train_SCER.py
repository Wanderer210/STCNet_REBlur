import os
import torch
import h5py
import torch
import numpy as np
import os
import glob
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler
from config import Config
from U_model import unet
import utils
import losses


class REBlurVoxelH5Dataset(Dataset):
    def __init__(self, data_dir, opt, is_train=True):
        self.opt = opt
        self.is_train = is_train
        # 获取目录下所有的 .h5 文件
        self.h5_files = sorted(glob.glob(os.path.join(data_dir, '*.h5')))
        
        # 建立全局索引映射：(h5_file_path, frame_idx)
        self.samples = []
        for h5_path in self.h5_files:
            with h5py.File(h5_path, 'r') as f:
                num_frames = len(f['images'].keys())
                for i in range(num_frames):
                    self.samples.append((h5_path, i, num_frames))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        h5_path, img_idx, max_frames = self.samples[idx]
        
        # 计算三帧的索引：过去(p), 当前(c), 未来(f)
        idx_p = max(0, img_idx - 1)
        idx_c = img_idx
        idx_f = min(max_frames - 1, img_idx + 1)
        
        indices = [idx_p, idx_c, idx_f]
        
        blur_imgs = []
        voxels = []
        
        with h5py.File(h5_path, 'r') as f:
            for i in indices:
                img_key = f"image{i:09d}"
                voxel_key = f"voxel{i:09d}"
                
                img = np.array(f['images'][img_key]).astype(np.float32)
                voxel = np.array(f['voxels'][voxel_key]).astype(np.float32)
                
                # 归一化
                if img.max() > 1.0: img /= 255.0
                blur_imgs.append(img)
                voxels.append(voxel)
            
            # 读取当前帧对应的 Ground Truth
            gt_key = f"image{idx_c:09d}"
            sharp_img = np.array(f['sharp_images'][gt_key]).astype(np.float32)
            if sharp_img.max() > 1.0: sharp_img /= 255.0

        # Stack 为 (3, C, H, W)
        blur_imgs = np.stack(blur_imgs, axis=0) 
        voxels = np.stack(voxels, axis=0)

        return torch.from_numpy(blur_imgs), torch.from_numpy(voxels), torch.from_numpy(sharp_img)


def main():
    # 1. 加载配置
    from config import Config
    opt = Config('REBlur_funtine_SCER.yml')
    
    # 【修复 AttributeError: future_frames】
    # 如果 config.py 或 yml 中没有这些参数，手动注入以适配 unet.py
    if not hasattr(opt, 'future_frames'):
        opt._C.future_frames = 1
    if not hasattr(opt, 'past_frames'):
        opt._C.past_frames = 1

    # 2. 路径设置
    train_dir = '/home/zy/data/zy/zhaoyue/Datasets/REBlur_EFNet/train'
    val_dir = '/home/zy/data/zy/zhaoyue/Datasets/REBlur_EFNet/test'
    model_save_dir = os.path.join(opt.TRAINING.SAVE_DIR, 'models', 'REBlur_SCER')
    utils.mkdir(model_save_dir)

    # 3. 初始化模型 (unet.Restoration 现在可以读取到 future_frames)
    model = unet.Restoration(3, 6, 3, opt)
    model.cuda()
    
    # 4. 数据加载器
    # 使用上面修改后的 Dataset 类
    train_dataset = REBlurVoxelH5Dataset(train_dir, opt, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    val_dataset = REBlurVoxelH5Dataset(val_dir, opt, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    # 5. 优化器与损失函数
    import torch.optim as optim
    optimizer = optim.Adam(model.parameters(), lr=opt.OPTIM.LR_INITIAL, betas=(0.9, 0.999))
    criterion = losses.CharbonnierLoss()
    
    # 6. 学习率调度器
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS - warmup_epochs, eta_min=opt.OPTIM.LR_MIN)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    
    # 7. 训练循环
    best_psnr = 0
    for epoch in range(1, opt.OPTIM.NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0
        
        for iteration, (img, voxel, gt) in enumerate(tqdm(train_loader), 1):
            # img: (B, 3, 3, H, W), voxel: (B, 3, 6, H, W)
            img, voxel, gt = img.cuda(), voxel.cuda(), gt.cuda()
            
            optimizer.zero_grad()
            
            # 模型内部会处理 frames=3 的逻辑
            restored = model(img, voxel) 
            
            # restored 是中间帧的重建结果，与 gt 计算损失
            loss = criterion(restored, gt)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        scheduler.step()
        print(f"Epoch {epoch} Loss: {epoch_loss/len(train_loader):.4f}")

        # 8. 验证
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model.eval()
            psnr_val = []
            with torch.no_grad():
                for img, voxel, gt in val_loader:
                    # 验证集同样输入 5 维张量
                    restored = model(img.cuda(), voxel.cuda())
                    psnr_val.append(utils.torchPSNR(restored, gt.cuda()))
            
            avg_psnr = torch.stack(psnr_val).mean().item()
            print(f"Validation PSNR: {avg_psnr:.2f}")
            
            # 保存最佳模型
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                save_path = os.path.join(model_save_dir, "model_best.pth")
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'psnr': best_psnr}, save_path)
                print(f"==> Best model saved to {save_path}")

if __name__ == '__main__':
    main()