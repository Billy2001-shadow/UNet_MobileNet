from os.path import join, splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from os.path import join as opj


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        # a = np.array(pil_img)
        # if len(a.shape) == 2:
        #     print("1:",np.sum(a == 1))
        #     print("2:",np.sum(a == 2))
        pil_img = pil_img.resize((newW, newH),Image.NEAREST)

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:

            img_nd = np.expand_dims(img_nd, axis=2)

        else:

            img_nd = img_nd / 255.0
            # img_nd = img_nd

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        # img_trans = img_nd
        return img_trans


    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(opj(self.masks_dir,idx+"_mask.*"))  # self.mask_suffix
        img_file = glob(opj(self.imgs_dir , idx + '.*'))

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale) 

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')