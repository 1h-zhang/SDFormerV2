# Original authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import torch
import torch.nn.functional as F
import pdb

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            pass


def get_output(output, task):
    """Borrow from MTI-Net"""
    
    if task == 'normals':
        output = output.permute(0, 2, 3, 1)
        output = (F.normalize(output, p = 2, dim = 3) + 1.0) * 255 / 2.0
    
    elif task in {'semseg'}:
        output = output.permute(0, 2, 3, 1)
        _, output = torch.max(output, dim=3)

    elif task in {'human_parts'}:
        output = output.permute(0, 2, 3, 1)
        _, output = torch.max(output, dim=3)
    
    elif task in {'edge'}:
        output = output.permute(0, 2, 3, 1)
        output = torch.squeeze(255 * 1 / (1 + torch.exp(-output)))

    elif task in {'sal'}:
        output = output.permute(0, 2, 3, 1)
        output = F.softmax(output, dim=3)[:, :, :, 1] *255 # torch.squeeze(255 * 1 / (1 + torch.exp(-output)))
    
    elif task in {'depth'}:
        output.clamp_(min=0.)
        output = output.permute(0, 2, 3, 1)
    
    else:
        raise ValueError('Select one of the valid tasks')

    return output

def to_cuda(batch):
    if type(batch) == dict:
        out = {}
        for k, v in batch.items():
            if k == 'meta':
                out[k] = v
            else:
                out[k] = to_cuda(v)
        return out
    elif type(batch) == torch.Tensor:
        return batch.cuda(non_blocking=True)
    elif type(batch) == list:
        return [to_cuda(v) for v in batch]
    else:
        return batch

# From PyTorch internals
import collections.abc as container_abcs
from itertools import repeat
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

to_2tuple = _ntuple(2)



def update_tb(tb_writer, tag, loss_dict, iter_no):
    """添加训练过程中各任务损失数值到tensorboard"""

    for k, v in loss_dict.items():
        tb_writer.add_scalar(f'{tag}/{k}', v.item(), iter_no)


def tb_update_perf(p, tb_writer_test, curr_result, cur_iter):
    """添加验证过程中各任务性能指标到tensorboard"""
    if 'semseg' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/semseg_miou', curr_result['semseg']['mIoU'], cur_iter)
    if 'human_parts' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/human_parts_mIoU', curr_result['human_parts']['mIoU'], cur_iter)
    if 'sal' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/sal_maxF', curr_result['sal']['maxF'], cur_iter)
    if 'edge' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/edge_val_loss', curr_result['edge']['loss'], cur_iter)
    if 'normals' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/normals_mean', curr_result['normals']['mean'], cur_iter)
    if 'depth' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/depth_rmse', curr_result['depth']['rmse'], cur_iter)