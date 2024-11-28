import os
import torch
import torch.nn as nn
import math

from einops import rearrange as o_rearrange
def rearrange(*args, **kwargs):
    return o_rearrange(*args, **kwargs).contiguous()


# 用于mix_ffn
class DWConv(nn.Module):
    """
    Depthwise convolution bloc: input: x with size(B N C); output size (B N C)
    """
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()     # B N C -> B C N -> B C H W
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)    # B C H W -> B N C

        return x


class Mix_Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        """
        MLP Block: 
        """
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class SRAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Linear embedding
        self.q1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.q2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            # self.sr1 = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr1 = nn.AvgPool2d(kernel_size=sr_ratio, padding=0, stride=sr_ratio, ceil_mode=True)
            self.norm1 = nn.LayerNorm(dim)

            # self.sr2 = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr2 = nn.AvgPool2d(kernel_size=sr_ratio, padding=0, stride=sr_ratio, ceil_mode=True)
            self.norm2 = nn.LayerNorm(dim)

    def forward(self, x1, x2, h, w):
        B, N, C = x1.shape

        q1 = self.q1(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = self.q1(x2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        if self.sr_ratio > 1:
            x_ = x1.permute(0, 2, 1).reshape(B, C, h, w)
            x_ = self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm1(x_)
            kv1 = self.kv1(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

            x_ = x2.permute(0, 2, 1).reshape(B, C, h, w)
            x_ = self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm2(x_)
            kv2 = self.kv2(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            kv2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k1, v1 = kv1.unbind(0)
        k2, v2 = kv2.unbind(0)

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale  # matrix[s1,d2]亲和力矩阵
        attn1 = attn1.softmax(dim=-1)
        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale  # matrix[d2,s1]亲和力矩阵
        attn2 = attn2.softmax(dim=-1)

        attn1 = self.attn_drop(attn1)  # 这里的dropout一般为0
        attn2 = self.attn_drop(attn2)  # 这里的dropout一般为0

        # s1-->d2 的亲和力矩阵要从 d2中提取利于s1的信息--->x1
        x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C).contiguous()
        x1 = self.proj1(x1)
        x1 = self.proj_drop(x1)
        # d2-->s1 的亲和力矩阵要从 s1中提取利于d2的信息--->x2
        x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C).contiguous()
        x2 = self.proj2(x2)
        x2 = self.proj_drop(x2)

        return x1, x2



class ReduceSpatial_Fusion_block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super(ReduceSpatial_Fusion_block, self).__init__()
        self.norm11 = norm_layer(dim)
        self.norm12 = norm_layer(dim)

        self.spatial_cross_attn = SRAttention(dim, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.norm21 = norm_layer(dim)
        self.norm22 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        # 暂且先设置为两个分支独立于norm和mlp
        self.mlp1 = Mix_Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mix_Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, seg, depth, h, w):
        """
        seg and depth are branch feature map respectively, shape -> [B, N, C] h * w = N
        """
        offset_seg, offset_depth = self.spatial_cross_attn(self.norm11(seg), self.norm12(depth), h, w)

        seg = seg + offset_seg
        depth = depth + offset_depth

        seg = seg + self.mlp1(self.norm21(seg), h, w)  # [B, N, C]
        depth = depth + self.mlp2(self.norm22(depth), h, w)

        return seg, depth


# base transformer layers

class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)   # linear只会改变最后一个维度的长度(matrix)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x



if __name__ == '__main__':
    from thop import profile, clever_format

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # spatial = torch.Tensor(1, 16384, 256).to(device)
    semseg = torch.Tensor(1, 8192, 256).to(device)
    depth = torch.Tensor(1, 8192, 256).to(device)

    embed_dim = 256

    # 定义Multi-transformer的模型
    MTB = ReduceSpatial_Fusion_block(dim=256, num_heads=4,
                                     mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                     drop=0., attn_drop=0., sr_ratio=4).to(device)

    base_layer = Block(dim=256, num_heads=4, qkv_bias=True, qk_scale=None).to(device)

    input1 = torch.Tensor(1, 16384, 256).to(device)

    flops, params = profile(model=MTB, inputs=(semseg, depth, 64, 128))
    flops, params = clever_format([flops, params], '%0.3f')
    print('Flops:', flops, ',Params', params)
