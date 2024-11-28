import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoders.new_fusion_decoder import Fusion_Decoder



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
        self.trans_decoder = Fusion_Decoder(p, spatial_num_heads=4, mlp_ratio=4, qkv_bias=True,
                                            qk_scale=None, drop=0., attn_drop=0.)


        self.init_weights(self.pretrained)  # 预训练权重加载


    def init_weights(self, pretrained=None):
        if pretrained:
            print('loading pretrained mdoel: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)


    def forward(self, x):
        img_size = x.size()[-2:]
        out = {}

        # backbone
        selected_fea = self.backbone(x)

        # transformer decoder  inter_pred用于强监督，但是并未开启
        task_features, inter_preds= self.trans_decoder(selected_fea)


        # Generate predictions
        out = task_features
        for t in self.tasks:
            out[t] = F.interpolate(self.heads[t](task_features[t]), img_size, mode='bilinear', align_corners=False)
        # out['inter_preds'] = {t: F.interpolate(v, img_size, mode='bilinear', align_corners=False)
        #                       for t, v in inter_preds.items()}

        return out







if __name__ == '__main__':
    from utils.common_config import get_model
    from config import create_config

    p = create_config('../data/city_config.yml')
    model = get_model(p)

    image = torch.rand(2, 3, 512, 1024)
    out = model(image)
    for name, val in out.items():
        print(name, val.shape)

    # print(out)
