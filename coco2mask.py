import os
import json
import shutil
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# 设置路径
base_dir = '/home/chenwu/UNet_MobileNet/lawn_obstacle/train'  # 原始图像和标注文件所在的文件夹
coco_json_path = os.path.join(base_dir, '_annotations_updated.coco.json')  # COCO格式的JSON文件
images_dir = '/home/chenwu/UNet_MobileNet/lawn_obstacle/image_' # 保存原图的目标文件夹
masks_dir = '/home/chenwu/UNet_MobileNet/lawn_obstacle/mask_' # 保存掩码的目标文件夹

# 创建输出文件夹
os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

# 读取COCO格式的JSON文件
with open(coco_json_path, 'r') as f:
    coco_data = json.load(f)

# 获取类别信息并分配颜色
category_to_color = {category['id']: idx + 1 for idx, category in enumerate(coco_data['categories'])}

# 移动图像并生成掩码
for image_info in tqdm(coco_data['images']):
    # 获取图像信息
    image_id = image_info['id']
    image_filename = image_info['file_name']
    width, height = image_info['width'], image_info['height']
    
    # 原图像路径和目标图像路径
    original_image_path = os.path.join(base_dir, image_filename)
    target_image_path = os.path.join(images_dir, image_filename)
    
    # 移动原图像到 images_ 文件夹
    if os.path.exists(original_image_path):
        shutil.move(original_image_path, target_image_path)
    
    # 创建空白掩码图像
    mask = Image.new('L', (width, height), 0)  # 单通道掩码图像，初始化为背景色 0

    # 获取与当前图像对应的注释
    annotations = [anno for anno in coco_data['annotations'] if anno['image_id'] == image_id]

    # 绘制掩码
    draw = ImageDraw.Draw(mask)
    for annotation in annotations:
        category_id = annotation['category_id']
        color = category_to_color[category_id]  # 获取类别对应的颜色
        if 'segmentation' in annotation and isinstance(annotation['segmentation'], list):
            for segmentation in annotation['segmentation']:
                polygon = [tuple(segmentation[i:i + 2]) for i in range(0, len(segmentation), 2)]
                draw.polygon(polygon, fill=color)  # 使用颜色填充多边形

    # 保存掩码到 masks_ 文件夹
    mask.save(os.path.join(masks_dir, f"{os.path.splitext(image_filename)[0]}_mask.png"))

print("原图已移动到 'images_' 文件夹，掩码已保存到 'masks_' 文件夹。")
