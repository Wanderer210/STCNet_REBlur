import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from U_model import unet
import utils
import losses
from dataset_REBlur_SCER import REBlurVoxelH5Dataset  # 导入上面写的类
from warmup_scheduler import GradualWarmupScheduler

def main():
    # 1. 加载配置
    opt = Config('REBlur_funtine_SCER.yml')
    
    # 2. 路径设置 (请根据您的实际路径修改)
    train_dir = '/home/zy/data/zy/zhaoyue/Datasets/REBlur_EFNet/train'
    val_dir = '/home/zy/data/zy/zhaoyue/Datasets/REBlur_EFNet/test'
    model_save_dir = os.path.join(opt.TRAINING.SAVE_DIR, 'models', 'REBlur_SCER')
    utils.mkdir(model_save_dir)

    # 3. 初始化模型
    # 输入通道: 图像3, 体素6 (根据您的h5ls结果), 输出3
    model = unet.Restoration(3, 6, 3, opt)
    model.cuda()
    
    # 4. 数据加载器
    train_dataset = REBlurVoxelH5Dataset(train_dir, opt, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    val_dataset = REBlurVoxelH5Dataset(val_dir, opt, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    # 5. 优化器与损失函数
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
            img, voxel, gt = img.cuda(), voxel.cuda(), gt.cuda()
            
            optimizer.zero_grad()
            restored = model(img, voxel)
            loss = criterion(restored, gt)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        scheduler.step()
        print(f"Epoch {epoch} Loss: {epoch_loss/len(train_loader):.4f}")

        # 8. 验证与保存 (每隔几个Epoch验证一次)
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model.eval()
            psnr_val = []
            with torch.no_grad():
                for img, voxel, gt in val_loader:
                    restored = model(img.cuda(), voxel.cuda())
                    psnr_val.append(utils.torchPSNR(restored, gt.cuda()))
            
            avg_psnr = torch.stack(psnr_val).mean().item()
            print(f"Validation PSNR: {avg_psnr:.2f}")
            
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, 
                           os.path.join(model_save_dir, "model_best.pth"))

if __name__ == '__main__':
    main()