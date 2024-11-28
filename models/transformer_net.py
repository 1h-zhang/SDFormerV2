import torch
import torch.nn as nn
import torch.nn.functional as F
from config import create_config

from models.decoders.transformer_decoder import TransformerDecoder


class TransformerNet(nn.Module):
    def __init__(self, p, heads):
        super(TransformerNet, self).__init__()

        self.channels = [64, 128, 320, 512]
        self.pretrained = p.pretrained
        self.tasks = p.TASKS.NAMES

        assert (p.backbone in ['swin_s', 'swin_b'])
        if p.backbone == "swin_s":
            print("Using backbone: Swin-Transformer-small")
            from models.encoders.swin_transformer import swin_small_patch4_window7_224 as backbone
            self.channels = [96, 192, 384, 768]
            p.backbone_channels = self.channels
            self.backbone = backbone()
        elif p.backbone == 'swin_b':
            print("Using backbone: Swin-Transformer-base")
            from models.encoders.swin_transformer import swin_base_patch4_window12_384_in22k as backbone
            self.channels = [128, 256, 512, 1024]
            p.backbone_channels = self.channels
            self.backbone = backbone()
        else:
            print("load error!")
            self.backbone = None

        self.heads = heads
        self.trans_decoder = TransformerDecoder(p)


        self.init_weights(self.pretrained)  # 预训练权重加载


    def init_weights(self, pretrained=None):
        if pretrained:
            print('loading pretrained mdoel: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)


    def forward(self, x):
        img_size = x.size()[-2:]
        out = {}

        # backbone [B, C, H, W]
        selected_fea = self.backbone(x)

        # transformer decoder  inter_pred用于强监督，但是并未开启
        task_features, inter_preds= self.trans_decoder(selected_fea)


        # Generate predictions
        out = task_features
        for t in self.tasks:
            out[t] = F.interpolate(self.heads[t](task_features[t]), img_size, mode='bilinear', align_corners=False)
        # out['inter_preds'] = {t: F.interpolate(v, img_size, mode='bilinear') for t, v in inter_preds.items()}

        return out


class MLPHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MLPHead, self).__init__()

        self.linear_pred = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.linear_pred(x)
        return x

def get_head(p, backbone_channels, task):
    """return the decoder head"""
    return MLPHead(backbone_channels, p.TASKS.NUM_OUTPUT[task])


def get_model(p):
    """return the model"""
    feat_channels = p.decoder_embed_dim
    heads = torch.nn.ModuleDict({task: get_head(p, feat_channels, task) for task in p.TASKS.NAMES})
    model = TransformerNet(p, heads)

    return model

if __name__ == "__main__":
    p = create_config('../data/nyud_swins_config.yml')
    model = get_model(p)

    # image = torch.rand(2, 3, 448, 576)
    image = torch.rand(2, 3, 112, 112)
    out = model(image)
    for name in p.TASKS.NAMES:
        print(name, out[name].shape)

