# import argparse
# import logging
# import os

# import numpy as np
# import torch
# import torch.nn.functional as F
# from PIL import Image
# from torchvision import transforms

# from mobilenet.UNet_MobileNet import UNet
# from matplotlib import pyplot as plt

# def plot_image(origin_image, out_img,out_fn):
#     def get_pallete():
#         pallete = [
#             0, 0, 0,
#             255, 0, 0,
#             0, 255, 0,
#         ]
#         return pallete
#     out_img =  np.array(out_img)
#     out_img = Image.fromarray(out_img.astype(np.uint8))
#     out_img.putpalette(get_pallete())
#     plt.imshow(origin_image)
#     #alpha参数可设置掩膜的透明度
#     plt.imshow(out_img, alpha=0.2)
#     print(f"save predicting image with name {out_fn} ")
#     plt.savefig(out_fn)

# def get_args():
#     parser = argparse.ArgumentParser(description='Predict masks from input images',
#                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('--input_img', '-i', metavar='INPUT', nargs='+',
#                         help='filenames of input images', required=True)
#     parser.add_argument('--input_mask', '-m', metavar='INPUT', nargs='+',
#                         help='filenames of input mask', required=True)
#     parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
#                         help='Filenames of ouput images')

#     return parser.parse_args()

# def get_output_filenames(args):
#     in_files = args.input_img
#     out_files = []

#     if not args.output:
#         for f in in_files:
#             pathsplit = os.path.splitext(f)
#             out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
#     elif len(in_files) != len(args.output):
#         logging.error("Input files and output files are not of the same length")
#         raise SystemExit()
#     else:
#         out_files = args.output

#     return out_files

# if __name__ == "__main__":
#     args = get_args()
#     in_files_img = args.input_img
#     in_files_mask = args.input_mask
#     out_files = get_output_filenames(args)

#     img = Image.open(str(in_files_img[0]))
#     mask = Image.open(str(in_files_mask[0]))

#     # 获取图片的像素数据
#     pixels = list(mask.getdata())
#     # 获取不同的像素值
#     unique_pixels = set(pixels)
#     # 打印不同的像素值
#     print(unique_pixels)


#     plot_image(img, mask, str(out_files[0]))



import argparse
import logging
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def create_mask_image(mask, palette):
    """将掩膜图按类别上色"""
    mask_array = np.array(mask)
    mask_img = Image.fromarray(mask_array.astype(np.uint8))
    mask_img.putpalette(palette)
    return mask_img.convert("RGB")  # 转换为RGB模式以便拼接

def plot_side_by_side(original_image, mask_image, output_filename):
    """将原图和掩膜图并排拼接并保存"""
    # 调整掩膜图大小以匹配原图大小
    mask_image = mask_image.resize(original_image.size)
    combined_image = Image.new('RGB', (original_image.width * 2, original_image.height))
    combined_image.paste(original_image, (0, 0))
    combined_image.paste(mask_image, (original_image.width, 0))
    
    # 保存拼接图像
    combined_image.save(output_filename)
    print(f"Saved combined image as {output_filename}")

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_img', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)
    parser.add_argument('--input_mask', '-m', metavar='INPUT', nargs='+',
                        help='filenames of input mask', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+',
                        help='Filenames of output images')
    return parser.parse_args()

def get_output_filenames(args):
    in_files = args.input_img
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files

if __name__ == "__main__":
    args = get_args()
    in_files_img = args.input_img
    in_files_mask = args.input_mask
    out_files = get_output_filenames(args)

    # 定义颜色调色板
    palette = [
        0, 0, 0,       # 类别 0 (背景) - 黑色
        255, 0, 0,     # 类别 1 - 红色
        0, 255, 0      # 类别 2 - 绿色
    ]

    # 处理每对输入图像和掩膜
    for img_file, mask_file, output_file in zip(in_files_img, in_files_mask, out_files):
        original_image = Image.open(img_file)
        mask = Image.open(mask_file)
        
        # 检查掩膜中的种类
        mask_array = np.array(mask)
        unique_classes = np.unique(mask_array)
        print(f"Mask file: {mask_file}")
        print(f"Unique classes in mask: {unique_classes.tolist()}")

        # 生成掩膜图
        mask_image = create_mask_image(mask, palette)
        
        # 将原图和掩膜图并排拼接并保存
        plot_side_by_side(original_image, mask_image, output_file)