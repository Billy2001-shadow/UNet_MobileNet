import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from mobilenet.UNet_MobileNet import UNet
from matplotlib import pyplot as plt

def plot_image(origin_image, out_img,out_fn):
    def get_pallete():
        pallete = [
            0, 0, 0,
            255, 0, 0,
            0, 255, 0,
        ]
        return pallete
    out_img =  np.array(out_img)
    out_img = Image.fromarray(out_img.astype(np.uint8))
    out_img.putpalette(get_pallete())
    plt.imshow(origin_image)
    #alpha参数可设置掩膜的透明度
    plt.imshow(out_img, alpha=0.6)
    print(f"save predicting image with name {out_fn} ")
    plt.savefig(out_fn)

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_img', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)
    parser.add_argument('--input_mask', '-m', metavar='INPUT', nargs='+',
                        help='filenames of input mask', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')

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

    img = Image.open(str(in_files_img[0]))
    mask = Image.open(str(in_files_mask[0]))

    # 获取图片的像素数据
    pixels = list(mask.getdata())
    # 获取不同的像素值
    unique_pixels = set(pixels)
    # 打印不同的像素值
    print(unique_pixels)


    plot_image(img, mask, str(out_files[0]))



