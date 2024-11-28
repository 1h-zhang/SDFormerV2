import torch
import torch.nn as nn

import os
import sys
import cv2

from PIL import Image
import numpy as np
import torch.utils.data as data


NYU_CATEGORY_NAMES = ['wall', 'floor', 'cabinet', 'bed', 'chair',
                      'sofa', 'table', 'door', 'window', 'bookshelf',
                      'picture', 'counter', 'blinds', 'desk', 'shelves',
                      'curtain', 'dresser', 'pillow', 'mirror', 'floor mat',
                      'clothes', 'ceiling', 'books', 'refridgerator', 'television',
                      'paper', 'towel', 'shower curtain', 'box', 'whiteboard',
                      'person', 'night stand', 'toilet', 'sink', 'lamp',
                      'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop']


CLASS_WEIGHT = [0.7613, 0.8796, 0.9183, 0.9541, 0.9511, 0.9689, 0.9724, 0.9776, 0.9764,
                0.9774, 0.9785, 0.9821, 0.9824, 0.9868, 0.9849, 0.9894, 0.9882, 0.9889,
                0.9887, 0.9917, 0.9892, 0.9905, 0.9932, 0.9937, 0.9930, 0.9950, 0.9958,
                0.9955, 0.9946, 0.9967, 0.9963, 0.9966, 0.9961, 0.9967, 0.9965, 0.9971,
                0.9972, 0.9736, 0.9753, 0.9387]


# 深度估计损失函数---berhu---
class BerHu_loss(nn.Module):
    def __init__(self, c=0.2, ignore_index=255):
        super(BerHu_loss, self).__init__()
        self.c = c
        self.ignore_index = ignore_index

    def forward(self, out, label, reduction='mean'):
        """out and label shape both are [B, 1, H, W], float type"""
        mask = (label != self.ignore_index).all(dim=1, keepdim=True)  # all操作对深度方向为1的没有影响(对深度方向进行all操作)
        n_valid = torch.sum(mask).item()
        masked_out = torch.masked_select(out, mask)         # masked_select输出一个一维向量  预测值
        masked_label = torch.masked_select(label, mask)     # 真实值

        diff = torch.abs(masked_out - masked_label)
        delta = self.c * torch.max(diff).item()             # delta is scaler
        berhu_loss = torch.where(diff <= delta, diff, (diff**2 + delta**2) / (2 * delta))

        if reduction == 'mean':
            # return torch.mean(berhu_loss)
            return berhu_loss.sum() / max(n_valid, 1)
        elif reduction == 'sum':
            return torch.sum(berhu_loss)
        elif reduction == 'none':
            return berhu_loss



# 学习率schedule创建
def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)



class NYUD_MT(data.Dataset):
    """
    from MTI-Net, changed for using ATRC data
    NYUD dataset for multi-task learning.
    Includes semantic segmentation and depth prediction.

    this is for calculate the class weight

    """

    def __init__(self, root=None, split='train', overfit=False, do_semseg=True):
        self.root = root

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        # Original Images
        self.im_ids = []
        self.images = []
        _image_dir = os.path.join(root, 'images')

        # Semantic segmentation
        self.do_semseg = do_semseg
        self.semsegs = []
        _semseg_gt_dir = os.path.join(root, 'segmentation')

        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(root, 'gt_sets')

        print('Initializing dataloader for NYUD {} set'.format(''.join(self.split)))
        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), 'r') as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                # Images
                _image = os.path.join(_image_dir, line + '.png')
                assert os.path.isfile(_image)
                self.images.append(_image)
                self.im_ids.append(line.rstrip('\n'))

                # Semantic Segmentation
                _semseg = os.path.join(self.root, _semseg_gt_dir, line + '.png')
                assert os.path.isfile(_semseg)
                self.semsegs.append(_semseg)

        if self.do_semseg:
            assert (len(self.images) == len(self.semsegs))

        # Uncomment to overfit to one image
        if overfit:
            n_of = 64
            self.images = self.images[:n_of]
            self.im_ids = self.im_ids[:n_of]

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))

    def __getitem__(self, index):
        sample = {}

        _img = self._load_img(index)
        sample['image'] = _img

        if self.do_semseg:
            _semseg = self._load_semseg(index)
            if _semseg.shape[:2] != _img.shape[:2]:
                print('RESHAPE SEMSEG')
                _semseg = cv2.resize(_semseg, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['semseg'] = _semseg

        # 转化为tensor格式
        for key, val in sample.items():
            sample[key] = torch.from_numpy(val.transpose((2, 0, 1))).long()

        return sample

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _img = np.array(_img, dtype=np.float32, copy=False)  # 转为[H, W, C]
        return _img

    def _load_semseg(self, index):
        # Note: We ignore the background class (40-way classification), as in related work:
        _semseg = Image.open(self.semsegs[index])
        _semseg = np.expand_dims(np.array(_semseg, dtype=np.float32, copy=False), axis=2)
        # _semseg[_semseg == -1] = 255
        return _semseg

    def __str__(self):
        return 'NYUD Multitask (split=' + str(self.split) + ')'



# 损失函数实现
def calculate_class_weights(dataset):
    num_classes = len(NYU_CATEGORY_NAMES) + 1      # 41
    total_count = 0

    class_weight = torch.zeros(num_classes)
    class_counts = torch.zeros(num_classes)
    for i in range(len(dataset)):
        image, label = dataset[i]['image'], dataset[i]['semseg']
        label_count = torch.bincount(label.view(-1), minlength=num_classes)
        class_counts += label_count

        total_count += torch.nonzero(label.view(-1)).size(0)

        if i % 50 == 0:
            print('current image index is:', i)

    nonzero = torch.sum(class_counts) - class_counts[0]
    total_count = torch.tensor(total_count, dtype=torch.float)

    # diff = nonzero - total_count
    # print(diff)

    class_weight = torch.div((total_count - class_counts.float()), total_count)[1:]

    return class_weight


if __name__ == '__main__':
    # class MyModel(nn.Module):
    #     def __init__(self):
    #         super(MyModel, self).__init__()
    #         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
    #         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
    #         self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
    #         self.relu = nn.ReLU()
    #         self.berhu_loss = BerHu_loss()
    #
    #     def forward(self, x, target):
    #         x = self.relu(self.conv1(x))
    #         x = self.relu(self.conv2(x))
    #         x = self.conv3(x)
    #         loss = self.berhu_loss(x, target)
    #         return x, loss
    #
    #
    # model = MyModel()
    #
    # x_input = torch.rand(2, 3, 64, 64)
    # label = torch.rand(2, 1, 64, 64)
    #
    # out, loss = model(x_input, label)
    # print(out.shape, loss)

    weight = [0.59, 0.78, 0.99, 0.97]
    weight = torch.tensor(weight, dtype=torch.float)

    loss = nn.CrossEntropyLoss(weight=weight)
    inp = torch.randn((1, 4, 10, 10), requires_grad=True)
    target = torch.randint(0, 4, size=(1, 10, 10))

    out = loss(inp, target)
    print(out, target)

