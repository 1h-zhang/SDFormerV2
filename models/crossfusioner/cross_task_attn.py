import torch
import torch.nn as nn

from einops import rearrange as o_rearrange
def rearrange(*args, **kwargs):
    return o_rearrange(*args, **kwargs).contiguous()




class ConvFFN(nn.Module):
    """
    目前 版本为1的ConvFFN 3x3 深度可分离矩阵 输出feature的shape为 [B, C, H, w]
    """
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ConvFFN, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=3, stride=1, padding=1,
                      bias=True, groups=out_channels // reduction),
            norm_layer(out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True)
        )

        self.norm = norm_layer(out_channels)

    def forward(self, x, h, w):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, h, w).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        # out shape: [B, C, h, w]
        out = self.norm(residual + x)

        return out


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





class SpatialCrossAttention(nn.Module):
    """
    空间交叉注意力计算
    """
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.
                 ):
        super(SpatialCrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.proj_qkv1 = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj_qkv2 = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1, x2):
        # [batch_size, num_patches, total_embed_dim]
        B, N, C = x1.shape
        # reshape: -> [batch_size, num_patches, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches, embed_dim_per_head]
        qkv1 = self.proj_qkv1(x1).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        qkv2 = self.proj_qkv2(x2).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        q1, k1, v1 = qkv1.unbind(0)
        q2, k2, v2 = qkv2.unbind(0)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches]
        # @: multiply -> [batch_size, num_heads, num_patches, num_patches]
        attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale        # matrix[s1,d2]亲和力矩阵
        attn1 = attn1.softmax(dim=-1)
        attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale        # matrix[d2,s1]亲和力矩阵
        attn2 = attn2.softmax(dim=-1)

        attn1 = self.attn_drop(attn1)  # 这里的dropout一般为0
        attn2 = self.attn_drop(attn2)  # 这里的dropout一般为0

        # s1-->d2 的亲和力矩阵要从 d2中提取利于s1的信息--->x1
        x1 = (attn1 @ v2).transpose(1, 2).reshape(B, N, C).contiguous()
        x1 = self.proj1(x1)
        x1 = self.proj_drop(x1)
        # d2-->s1 的亲和力矩阵要从 s1中提取利于d2的信息--->x2
        x2 = (attn2 @ v1).transpose(1, 2).reshape(B, N, C).contiguous()
        x2 = self.proj2(x2)
        x2 = self.proj_drop(x2)

        return x1, x2


class Spatial_Fusion_block(nn.Module):
    """
    spatial fusion block: spatial-cross-attention --> Mix FFN
    """
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Spatial_Fusion_block, self).__init__()
        self.norm11 = norm_layer(dim)
        self.norm12 = norm_layer(dim)

        self.spatial_cross_attn = SpatialCrossAttention(dim, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                        attn_drop=attn_drop, proj_drop=drop)
        self.norm21 = norm_layer(dim)
        self.norm22 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        # 暂且先设置为两个分支独立于norm和mlp
        self.mlp1 = Mix_Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mix_Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, seg, depth, h, w):
        """
        seg and depth are branch feature map respectively, shape -> [B, N, C] h * w = N
        后续会更改 offset 和 origin 的融合方式
        """
        offset_seg, offset_depth = self.spatial_cross_attn(self.norm11(seg), self.norm12(depth))

        seg = seg + offset_seg
        depth = depth + offset_depth

        seg = seg + self.mlp1(self.norm21(seg), h, w)  # [B, N, C]
        depth = depth + self.mlp2(self.norm22(depth), h, w)

        return seg, depth



class task_pattern_propagation(nn.Module):
    """
    构建各自任务注意力模式，形成共享任务模式，然后再进行任务模式分配
    """
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super(task_pattern_propagation, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.proj_qkv1 = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj_qkv2 = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        # channel reduction
        self.attn_conv = nn.Conv2d(in_channels=2*num_heads, out_channels=num_heads, kernel_size=1)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1, x2):
        # [batch_size, num_patches, total_embed_dim]
        B, N, C = x1.shape
        # reshape: -> [batch_size, num_patches, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches, embed_dim_per_head]
        qkv1 = self.proj_qkv1(x1).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                 4).contiguous()
        qkv2 = self.proj_qkv2(x2).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                 4).contiguous()
        q1, k1, v1 = qkv1.unbind(0)
        q2, k2, v2 = qkv2.unbind(0)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches]
        # @: multiply -> [batch_size, num_heads, num_patches, num_patches]
        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale  # x1的任务模式注意力
        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale  # x2的任务模式注意力
        # concat: -> [batch, 2*num_heads, num_patches, num_patches]
        cat_attn = torch.cat((attn1, attn2), dim=1)
        # channel reduction: -> [batch, num_heads, num_patches, num_patches]
        cat_attn = self.attn_conv(cat_attn)                 # 共享任务模式

        cat_attn = cat_attn.softmax(dim=-1)
        cat_attn = self.attn_drop(cat_attn)
        # share task pattern
        x1 = (cat_attn @ v1).transpose(1, 2).reshape(B, N, C).contiguous()
        x1 = self.proj1(x1)
        x1 = self.proj_drop(x1)

        x2 = (cat_attn @ v2).transpose(1, 2).reshape(B, N, C).contiguous()
        x2 = self.proj2(x2)
        x2 = self.proj_drop(x2)

        return x1, x2




if __name__ == '__main__':
    x1 = torch.randn(2, 49, 128)
    x2 = torch.randn(2, 49, 128)

    # tpp_module = task_pattern_propagation(128, num_heads=8, qkv_bias=True, qk_scale=None)
    TPP_module = task_pattern_propagation(128, num_heads=8, mlp_ratio=4, qkv_bias=True, qk_scale=None)
    x11, x22 = task_pattern_propagation_module(x1, x2)
    print(x11.shape)
    print(x22.shape)
