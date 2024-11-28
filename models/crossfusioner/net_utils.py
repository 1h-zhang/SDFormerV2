import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvBlock(nn.Module):
    """conv3 -- bn -- relu"""
    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(ConvBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                              padding=dilation, groups=groups, bias=False, dilation=dilation)

        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out


class ConvBNReLU(nn.Module):
    """任意核大小的Conv-bn-relu"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, bias='auto',
                 inplace=True, affine=True):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.use_norm = norm_layer is not None
        self.use_activation = activation_layer is not None
        if bias == 'auto':
            bias = not self.use_norm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                              dilation=dilation, groups=groups, bias=bias)
        if self.use_norm:
            self.bn = norm_layer(out_channels, affine=affine)
        if self.use_activation:
            self.activation = activation_layer(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm:
            x = self.bn(x)
        if self.use_activation:
            x = self.activation(x)
        return x



class h_sigmoid(nn.Module):
    """
    used for context injection fusion
    """
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6



class context_inject_fusion_block(nn.Module):
    """feature fusion function 2 -- topformer"""
    def __init__(self, task_chan, fea_chan, reduction=2, norm_layer=nn.BatchNorm2d):
        super(context_inject_fusion_block, self).__init__()
        self.reduction = reduction
        self.embed_chan = int(task_chan // self.reduction)

        self.task_embed = nn.Sequential(
            nn.Conv2d(in_channels=task_chan, out_channels=self.embed_chan, kernel_size=1),
            norm_layer(self.embed_chan)
        )

        self.fea_embed = nn.Sequential(
            nn.Conv2d(in_channels=fea_chan, out_channels=self.embed_chan, kernel_size=1),
            norm_layer(self.embed_chan)
        )

        self.task_act = nn.Sequential(
            nn.Conv2d(in_channels=task_chan, out_channels=self.embed_chan, kernel_size=1),
            norm_layer(self.embed_chan)
        )
        self.act = h_sigmoid()

    def forward(self, x_c, x_f, H, W):
        """
        :param x_c: task context feature [B, N, C]
        :param x_f: backbone feature  [B, C, H, W]
        """
        B, N, _C = x_c.shape
        x_c = x_c.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        # backbone feature conv-bn
        back_feature = self.fea_embed(x_f)
        # task feature conv-bn-sigmoid-up
        task_act = self.task_act(x_c)
        sig_act = F.interpolate(self.act(task_act), size=x_f.shape[-2:], mode='bilinear', align_corners=False)
        # task feature conv-bn-up
        task_feature = self.task_embed(x_c)
        task_feature = F.interpolate(task_feature, size=x_f.shape[-2:], mode='bilinear', align_corners=False)

        fusion_out = back_feature * sig_act + task_feature
        fusion_out = fusion_out.flatten(2).transpose(1, 2)      # 再次变换为B, N, C

        return fusion_out





def window_partition(x, h, w, pad_r, pad_b, group, num_head):
    """
    根据 num_head 将feature map 从空间方向上划分为一个个没有重叠的window
    """
    B, C, N = x.shape
    x = x.view(B, C, h, w)

    x = F.pad(x, (0, pad_r, 0, pad_b, 0, 0))
    # 高和宽方向上的分组数目相等
    # num_group = int(math.sqrt(num_head))
    # 如果输入图片的H,W不是num_group的整数倍，需要进行padding
    # pad_input = (h % num_group != 0) or (w % num_group != 0)
    # if pad_input:
    #     # to pad last 3 dimensions
    #     # (l, r, t, b, front, back)
    #     x = F.pad(x, (0, num_group - w % num_group, 0, num_group - h % num_group, 0, 0))

    _, _, H, W = x.shape
    # view: [B, C, H, W] -> [B, C, Ng, Ng_size, Ng, Ng_size]
    x = x.view(B, C, group, H // group, group, W // group)
    # permute: [B, C, Ng, Ng_size, Ng, Ng_size] -> [B, C, Ng, Ng, Ng_size, Ng_size]
    # view: [B, C, Ng, Ng, Ng_size, Ng_size] -> [B, C, num_head, N // num_head]
    windows = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(-1, C, num_head, (H * W) // num_head)
    # permute: [B, C, num_head, N // num_head] -> [B, num_head, C, N // num_head]
    windows = windows.permute(0, 2, 1, 3).contiguous()

    return windows


def window_reverse(x, h, w, pad_r, pad_b, group):
    """
    将空间层面的dim根据空间位置还原回去
    """
    B, _, C, _ = x.shape
    new_w = w + pad_r
    new_h = h + pad_b
    # transpose: [B, head, C, dim] -> [B, C, head, dim]
    # view: [B, C, head, dim] -> [B, C, num_h, num_w, h' // num_h, w' // num_w]
    x = x.transpose(1, 2).view(B, C, group, group, new_h // group, new_w // group).contiguous()
    # permute: [B, C, num_h, num_w, h' // num_h, w' // num_w] -> [B, C, num_h, h' // num_h, num_w, w' // num_w]
    # view: [B, C, num_h, h' // num_h, num_w, w' // num_w] -> [B, C, h', w']
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, new_h, new_w)
    # if pad remove the pad part
    if pad_r > 0 or pad_b > 0:
        x = x[:, :, :h, :w].contiguous()
    # [B, C, N]
    x = x[:, :, :h, :w].flatten(2)

    return x