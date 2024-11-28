# author: Frank

import os
import sys
import cv2
import torch
import json

from PIL import Image
import numpy as np
import torch.utils.data as data
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
import scipy.io as sio

def get_city_transformations():
    """ Return cityscapes transformations for training and evaluationg """
    from data import transforms
    import torchvision

    # Training transformations
    if True:
        train_transforms = torchvision.transforms.Compose([  # from ATRC
            transforms.DirectResize(size=(512, 1024)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.PhotoMetricDistortion(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])

        # Testing
        valid_transforms = torchvision.transforms.Compose([
            transforms.DirectResize(size=(512, 1024)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])
        return train_transforms, valid_transforms

    else:
        return None, None

class CITYSCAPES(data.Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.transform = transform

        # 索引到具体的文件夹
        self.img_root = os.path.join(self.root, 'leftImg8bit', 'cityscapes_' + split)
        self.annotations_root = os.path.join(self.root, 'gtFine', '19classes_' + split)
        self.disparity_root = os.path.join(self.root, 'disparity', 'disparity_' + split)
        self.camera_root = os.path.join(self.root, 'camera', 'camera_' + split)
        # 提取文件夹中文件名称组成一个数组（包含后缀名）
        self.img_names = os.listdir(self.img_root)
        self.annotations_names = os.listdir(self.annotations_root)
        self.disparity_names = os.listdir(self.disparity_root)
        self.camera_names = os.listdir(self.camera_root)
        assert (len(self.img_names) == len(self.annotations_names) == len(self.disparity_names) == len(self.camera_names))

        self.images = [os.path.join(self.img_root, name) for name in self.img_names]
        self.annotations = [os.path.join(self.annotations_root, anno) for anno in self.annotations_names]
        self.disparitys = [os.path.join(self.disparity_root, dis) for dis in self.disparity_names]
        self.cameras = [os.path.join(self.camera_root, cam) for cam in self.camera_names]
        print("contain %d %s images" % (len(self.img_names), split))

    def __getitem__(self, index):
        # 因为是用os.listdir来索引的先看看名字对不对应
        # assert (self.img_names[index].split('_')[0:3] == self.annotations_names[index].split('_')[0:3])
        # assert (self.img_names[index].split('_')[0:3] == self.disparity_names[index].split('_')[0:3])
        # assert (self.img_names[index].split('_')[0:3] == self.camera_names[index].split('_')[0:3])

        sample = {}

        _img = self._load_img(index)
        sample['image'] = _img

        _semseg = self._load_semseg(index)      # (H, W, 1)
        sample['semseg'] = _semseg

        _depth = self._load_depth(index)        # (H, W, 1)
        # 本身为0的区域不参与计算 赋值为-1
        _depth[_depth == 0] = -1
        # 天空部分赋值为0 但是要参与计算
        sky_mask = _semseg == 10
        _depth[sky_mask] = 0
        sample['depth'] = _depth

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.img_names)


    def _load_img(self, index):
        assert os.path.isfile(self.images[index])
        _img = Image.open(self.images[index]).convert('RGB')
        _img = np.array(_img, dtype=np.float32, copy=False)  # 转为[H, W, C]
        return _img

    def _load_semseg(self, index):
        assert os.path.isfile(self.annotations[index])
        _semseg = Image.open(self.annotations[index])
        _semseg = np.expand_dims(np.array(_semseg, dtype=np.float32, copy=False), axis=2)
        return _semseg

    def _load_depth(self, index):
        assert os.path.isfile(self.disparitys[index])
        assert os.path.isfile(self.cameras[index])
        depth = cv2.imread(self.disparitys[index], cv2.IMREAD_UNCHANGED).astype(np.float32)     # disparity  (H, W)

        depth[depth > 0] = (depth[depth > 0] - 1) / 256  # disparity values

        if False:
            camera = json.load(open(self.cameras[index]))
            # real depth
            depth[depth > 0] = camera["extrinsic"]["baseline"] * camera["intrinsic"]["fx"] / depth[depth > 0]

        _depth = np.expand_dims(depth, axis=2)      # trans to (H, W, 1)
        return _depth



if __name__ == "__main__":
    root = 'E:/Frank/dataset/Cityscapes'
    train_transform, test_transform = get_city_transformations()
    # train_dataset = CITYSCAPES(root, 'train', transform=train_transform)
    test_dataset = CITYSCAPES(root, 'val', transform=test_transform)

    example = test_dataset[0]

    for name in example:
        print(name, example[name].shape)

    # figure = plt.figure(figsize=(10, 10))
    # cols, rows = 3, 3
    # for i in range(0, rows):
    #     sample_idx = torch.randint(len(test_dataset), size=(1,)).item()
    #     sample = test_dataset[sample_idx]
    #     j = i * 3 + 1
    #     for idx, name in enumerate(sample, j):
    #         print(name, sample[name].shape)
    #         figure.add_subplot(rows, cols, idx)
    #         plt.imshow(sample[name].permute(1, 2, 0))
    #
    # plt.show()



