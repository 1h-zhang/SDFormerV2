import torch
import torch.nn as nn
import os
import json
from evaluation.evaluate_utils import PerformanceMeter
import utils.distributed_utils as utils
from utils.utils import to_cuda, get_output, update_tb, tb_update_perf



def seg_depth_train_one_epoch(p, model, optimizer, criterion, train_loader, device, epoch, scheduler,
                              tb_writer, iter_count, print_freq=100):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for cpu_batch in metric_logger.log_every(train_loader, print_freq, header):
        batch = to_cuda(cpu_batch)
        images = batch['image']
        output = model(images)
        iter_count += 1

        # measure loss
        loss_dict = criterion(output, batch, tasks=p.TASKS.NAMES)

        if tb_writer is not None:
            update_tb(tb_writer, 'Train_Loss', loss_dict, iter_count)

        # Backward
        optimizer.zero_grad()
        loss_dict['total'].backward()
        optimizer.step()

        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss_dict['total'].item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr, iter_count


def seg_depth_evaluate_phase(p, test_dataloader, model, tb_writer_test, iter_count):
    tasks = p.TASKS.NAMES
    performance_meter = PerformanceMeter(p, tasks)

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test'

    with torch.no_grad():
        for batch in metric_logger.log_every(test_dataloader, 50, header):
            images = batch['image'].cuda(non_blocking=True)
            targets = {task: batch[task].cuda(non_blocking=True) for task in tasks}

            output = model(images)

            # Measure loss and performance
            performance_meter.update({t: get_output(output[t], t) for t in tasks},
                                     {t: targets[t] for t in tasks})

    eval_results = performance_meter.get_score(verbose=True)
    tb_update_perf(p, tb_writer_test, eval_results, iter_count)         # 添加到tensorboard查看参数

    print('Evaluate results at iteration {}: \n'.format(iter_count))
    print(eval_results)
    with open(os.path.join(p['save_dir'], p.version_name + '_' + str(iter_count) + '.txt'), 'w') as f:
        json.dump(eval_results, f, indent=4)    # 保存参数信息


def seg_depth_evaluate(p, test_dataloader, model):
    tasks = p.TASKS.NAMES
    performance_meter = PerformanceMeter(p, tasks)

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test'

    with torch.no_grad():
        for batch in metric_logger.log_every(test_dataloader, 50, header):
            images = batch['image'].cuda(non_blocking=True)
            targets = {task: batch[task].cuda(non_blocking=True) for task in tasks}

            output = model(images)

            # Measure loss and performance
            performance_meter.update({t: get_output(output[t], t) for t in tasks},
                                     {t: targets[t] for t in tasks})

    eval_results = performance_meter.get_score(verbose=True)