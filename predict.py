import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from mobilenet.UNet_MobileNet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
from matplotlib import pyplot as plt

def plot_image(origin_image, out_img,out_fn):
    def get_pallete():
        pallete = [
            0, 0, 0,
            255, 0, 0,
            0, 255, 0,
        ]
        return pallete

    out_img.putpalette(get_pallete())
    plt.imshow(origin_image)
    plt.imshow(out_img, alpha=0.3)
    print(f"save predicting image with name {out_fn} ")
    plt.savefig(out_fn)


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))
    img = img.unsqueeze(0)
    
    img = img.to(device=device, dtype=torch.float32)
    np.save("img.npy",img.cpu().numpy())
    with torch.no_grad():
        output = net(img)
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.num_classes > 1:
            output = output.squeeze(0)
            probs = output.argmax(dim=0)
            probs = probs.to(device=device, dtype=torch.uint8)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        full_mask = probs.squeeze().cpu().numpy()
        
    return full_mask


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='/home/chenwu/UNet_MobileNet/checkpoints/MobileNet_UNet_epoch_best.pt',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
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


def mask_to_image(mask):
    return Image.fromarray(mask.astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, num_classes=3)
    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)
        img = img.convert("RGB")
        img = img.resize((512, 512),Image.NEAREST)
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        out_fn = out_files[i]
        result = mask_to_image(mask)
        plot_image(img, result,out_fn)


