import os
import json

# 设置路径
images_dir = '/home/chenwu/UNet_MobileNet/lawn_obstacle/train/'  # 原始图像所在的文件夹
coco_json_path = os.path.join(images_dir, '_annotations.coco.json')  # COCO格式的JSON文件
output_json_path = os.path.join(images_dir, '_annotations_updated.coco.json')  # 保存修改后的JSON文件

# 读取COCO格式的JSON文件
with open(coco_json_path, 'r') as f:
    coco_data = json.load(f)

# 获取所有图像文件，排除标注文件
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg')) and f != '_annotations.coco.json'])
image_id_map = {}  # 存储原始文件名到新文件名的映射

# 重命名图像文件
for idx, original_file_name in enumerate(image_files, start=1):
    # 构建新的文件名
    new_file_name = f"{idx}.png"
    original_path = os.path.join(images_dir, original_file_name)
    new_path = os.path.join(images_dir, new_file_name)
    
    # 重命名图像文件
    os.rename(original_path, new_path)
    
    # 更新映射字典
    image_id_map[original_file_name] = new_file_name

# 更新COCO JSON文件中的文件名
for image_info in coco_data['images']:
    original_file_name = image_info['file_name']
    if original_file_name in image_id_map:
        image_info['file_name'] = image_id_map[original_file_name]

# 将更新后的JSON数据保存到新文件中
with open(output_json_path, 'w') as f:
    json.dump(coco_data, f, indent=4)

print(f"图像文件已重命名，并生成新的标注文件：{output_json_path}")
