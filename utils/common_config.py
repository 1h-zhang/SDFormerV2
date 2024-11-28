import torch
from torch.utils.data import DataLoader
from utils.custom_collate import collate_mil


def get_head(p, backbone_channels, task):
    """return the decoder head"""
    from models.decoders.new_fusion_decoder import MLPHead
    return MLPHead(backbone_channels, p.TASKS.NUM_OUTPUT[task])


def get_model(p):
    """return the model"""
    from models.seg_depth_net1 import TransformerNet
    feat_channels = p.decoder_embed_dim // 4      # 特征输入head时的通道数
    heads = torch.nn.ModuleDict({task: get_head(p, feat_channels, task) for task in p.TASKS.NAMES})
    model = TransformerNet(p, heads)

    return model



def get_head_(p, backbone_channels, task):
    """return the decoder head -- SDFormer_plus_plus """
    from models.transformer_net import MLPHead
    return MLPHead(backbone_channels, p.TASKS.NUM_OUTPUT[task])


def get_model_(p):
    """return the model -- SDFormer_plus_plus """
    from models.transformer_net import TransformerNet
    feat_channels = p.decoder_embed_dim
    heads = torch.nn.ModuleDict({task: get_head(p, feat_channels, task) for task in p.TASKS.NAMES})
    model = TransformerNet(p, heads)

    return model



def get_transformations(p):
    """ Return transformations for training and evaluationg """
    from data import transforms
    import torchvision

    # Training transformations
    if p['train_db_name'] == 'NYUD' or p['train_db_name'] == 'PASCALContext':
        train_transforms = torchvision.transforms.Compose([  # from ATRC
            transforms.RandomScaling(scale_factors=[0.5, 2.0], discrete=False),
            transforms.RandomCrop(size=p.TRAIN.SCALE, cat_max_ratio=0.75),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.PhotoMetricDistortion(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=p.TRAIN.SCALE),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])
        # Testing
        valid_transforms = torchvision.transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=p.TEST.SCALE),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])
        return train_transforms, valid_transforms
    elif p['train_db_name'] == 'CITYSCAPE':
        train_transforms = torchvision.transforms.Compose([  # from ATRC
            transforms.DirectResize(size=p.TRAIN.SCALE),
            transforms.RandomScaling(scale_factors=[0.75, 1.25], discrete=False),  # [0.5, 2.0]
            transforms.RandomCrop(size=p.TRAIN.SCALE, cat_max_ratio=0.75),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.PhotoMetricDistortion(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=p.TRAIN.SCALE),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])
        # Testing
        valid_transforms = torchvision.transforms.Compose([
            transforms.DirectResize(size=p.TEST.SCALE),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])
        return train_transforms, valid_transforms
    else:
        return None, None


def get_city_transformations(p):
    """ Return cityscapes transformations for training and evaluationg """
    from data import transforms
    import torchvision
    # Training transformations
    if p['train_db_name'] == 'CITYSCAPE':
        train_transforms = torchvision.transforms.Compose([  # from ATRC
            transforms.DirectResize(size=p.TRAIN.SCALE),
            transforms.RandomScaling(scale_factors=[0.75, 1.25], discrete=False),  # [0.5, 2.0]
            transforms.RandomCrop(size=p.TRAIN.SCALE, cat_max_ratio=0.75),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.PhotoMetricDistortion(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=p.TRAIN.SCALE),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])
        # Testing
        valid_transforms = torchvision.transforms.Compose([
            transforms.DirectResize(size=p.TEST.SCALE),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])
        return train_transforms, valid_transforms

    else:
        return None, None


def get_train_dataset(p, transforms=None):
    """ Return the train dataset """

    db_name = p['train_db_name']
    print('Preparing train dataset for db: {}'.format(db_name))

    if db_name == 'NYUD':
        from data.nyud import NYUD_MT
        database = NYUD_MT(p.db_path, download=False, split='train', transform=transforms,
                           do_semseg=True, do_depth=True, overfit=False)
    elif db_name == 'CITYSCAPE':
        from data.cityscapes import CITYSCAPES
        database = CITYSCAPES(p.db_path, split='train', transform=transforms)
    else:
        database = None


    return database


def get_train_dataloader(p, dataset):
    """ return the train dataloader"""
    collate = collate_mil
    trainloader = DataLoader(dataset, batch_size=p.trBatch, num_workers=p.num_workers, drop_last=True,
                             pin_memory=True, shuffle=True, collate_fn=collate)
    return trainloader


def get_test_dataset(p, transforms=None):
    """ Return the test dataset """

    db_name = p['train_db_name']
    print('Preparing test dataset for db: {}'.format(db_name))

    if db_name == 'NYUD':
        from data.nyud import NYUD_MT
        database = NYUD_MT(p.db_path, download=False, split='val', transform=transforms,
                           do_semseg=True, do_depth=True, overfit=False)
    elif db_name == 'CITYSCAPE':
        from data.cityscapes import CITYSCAPES
        database = CITYSCAPES(p.db_path, split='val', transform=transforms)
    else:
        database = None

    return database


def get_test_dataloader(p, dataset):
    """ return the test dataloader"""
    collate = collate_mil
    testloader = DataLoader(dataset, batch_size=p.valBatch, num_workers=p.num_workers, drop_last=False,
                            pin_memory=True, shuffle=False, collate_fn=collate)
    return testloader


"""
    loss functions
"""
def get_loss(p, task=None):
    """ Return loss function for a specific task """

    if task == 'edge':
        from losses.loss_functions import BalancedBinaryCrossEntropyLoss
        criterion = BalancedBinaryCrossEntropyLoss(pos_weight=p['edge_w'], ignore_index=p.ignore_index)

    elif task == 'semseg' or task == 'human_parts':
        from losses.loss_functions import CrossEntropyLoss
        criterion = CrossEntropyLoss(ignore_index=p.ignore_index)

    elif task == 'normals':
        from losses.loss_functions import L1Loss
        criterion = L1Loss(normalize=True, ignore_index=p.ignore_index)

    elif task == 'sal':
        from losses.loss_functions import CrossEntropyLoss
        criterion = CrossEntropyLoss(balanced=True, ignore_index=p.ignore_index)

    elif task == 'depth':
        from losses.loss_functions import L1Loss, L1Loss_v1
        criterion = L1Loss()
        # criterion = L1Loss_v1(ignore_invalid_area=True, ignore_index=-1)      # cityscape

    else:
        criterion = None

    return criterion


def get_criterion(p):
    from losses.loss_schemes import MultiTaskLoss
    loss_ft = torch.nn.ModuleDict({task: get_loss(p, task) for task in p.TASKS.NAMES})
    loss_weights = p['loss_kwargs']['loss_weights']
    return MultiTaskLoss(p, p.TASKS.NAMES, loss_ft, loss_weights)



