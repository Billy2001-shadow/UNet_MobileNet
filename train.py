import argparse
import logging
import os
import sys

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch import optim
from torch.backends import cudnn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

from utils.eval import eval_net
from utils.dataset import BasicDataset
from mobilenet.UNet_MobileNet import UNet


# 输入图片和标签的路径
# dir_img = '/home/users/nuo.wu/UNet-MobileNet-Pytorch/image_'
# dir_mask = '/home/users/nuo.wu/UNet-MobileNet-Pytorch/mask_'
# dir_checkpoint = '/home/chenwu/UNet-MobileNet/checkpoints'
dir_img = '/home/chenwu/UNet_MobileNet/lawn_obstacle/image_'
dir_mask = '/home/chenwu/UNet_MobileNet/lawn_obstacle/mask_'
dir_checkpoint = '/home/chenwu/UNet_MobileNet/checkpoints'

def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):

    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size,shuffle=True,num_workers=10,pin_memory=True,drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size,shuffle=True,num_workers=10,pin_memory=True,drop_last=True)
                            

    writer = SummaryWriter(log_dir = 'logs',
        comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')
    # 定义优化器
    optimizer = optim.RMSprop(net.parameters(), lr=lr,
                              weight_decay=1e-8, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 'min' if net.num_classes > 1 else 'max', patience=2)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True, threshold=0.01, threshold_mode='rel', cooldown=0, min_lr=0.00001, eps=1e-08)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma=0.5)
    # 定义损失函数，类别大于1使用交叉熵，否则使用BCE
    if net.num_classes > 1:
        # criterion = nn.CrossEntropyLoss(ignore_index=0)
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    # 开始训练
    val_loss_minimum = 1
    for epoch in range(epochs):
        net.train()
        num = 0
        val_loss = 0
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                
                mask_type = torch.float32 if net.num_classes == 1 else torch.long
                
                true_masks = true_masks.to(device=device, dtype=mask_type)
                
                masks_pred = net(imgs)
                loss = criterion(masks_pred, torch.squeeze(true_masks))
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                
                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram(
                            'weights/' + tag, value.data.cpu().numpy(), global_step) # 转换为 NumPy 并转换为 float32()并降低tensorboard和numpy的版本：TODO
                        # writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, val_loader, device)
                    val_loss += val_score
                    num += 1
                    # scheduler.step(val_score)
                    scheduler.step()
                    writer.add_scalar(
                        'learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.num_classes > 1:
                        logging.info(
                            'Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info(
                            'Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.num_classes == 1:
                        writer.add_images(
                            'masks/true', true_masks, global_step)
                        writer.add_images(
                            'masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
        val_loss_mean = val_loss / num
        if val_loss_mean < val_loss_minimum:
            val_loss_minimum = val_loss_mean
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       os.path.join(dir_checkpoint, f'MobileNet_UNet_epoch_best.pt'))
            logging.info(f'Checkpoint best saved !')
        # 保存模型
        if save_cp and (not epoch%10):
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       os.path.join(dir_checkpoint, f'MobileNet_UNet_epoch{epoch + 1}.pt'))
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=12,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default='',
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=20.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    args = get_args()

    # 判断是否使用GPU训练
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # 导入网络模型，这里需要结合自己的数据集调整n_channels和num_classes：
    # n_channels=3 for RGB images
    # num_classes的设置原则如下：1. 对于1个类别和背景，num_classes=1; 2. 对于2个类别，num_classes=1; 类别数N大于2，num_classes=N
    net = UNet(n_channels=3, num_classes=3)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.num_classes} output channels (classes)\n')

    # 是否导入预训练权重进行迁移学习
    if args.load:
        model_dict = net.state_dict()
        model_path = args.load
        pretrained_dict = torch.load(model_path, map_location=device)
        # 筛除不加载的层结构
        pretrained_dict = {k: v for k,
                           v in pretrained_dict.items() if k in model_dict}
        # 更新当前网络的结构字典
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        logging.info(f'Model loaded from {args.load}')

    # if torch.cuda.device_count() > 1:
    #     net = nn.DataParallel(net)
    
    net.to(device=device)

    # faster convolutions, but more memory
    cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pt')
        logging.info('Saved interrupt')
