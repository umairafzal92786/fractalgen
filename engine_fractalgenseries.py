import math
import os
import shutil
import sys
import time
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch_fidelity

import util.lr_sched as lr_sched
import util.misc as misc


def train_one_epoch(model, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # print("samples shape: ", samples.shape)
        # print("labels shape: ", labels.shape)

        # forward
        with torch.cuda.amp.autocast():
            loss = model(samples, labels)

        loss_value = loss.item()

        # print("loss: ", loss_value)
        # exit()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(), update_grad=True)
        optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def validate(model, data_loader: Iterable, device: torch.device, epoch: int, log_writer=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    with torch.no_grad():
        for data_iter_step, (samples, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            samples = samples.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # forward
            with torch.cuda.amp.autocast():
                loss = model(samples, labels)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping validation".format(loss_value))
                sys.exit(1)

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)
            lr = 0.0  # No learning rate during validation
            metric_logger.update(lr=lr)

            loss_value_reduce = misc.all_reduce_mean(loss_value)
            if log_writer is not None:
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar('validation_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # Return the average validation loss
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def compute_nll(model: torch.nn.Module, data_loader: Iterable, device: torch.device, N: int):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = ''
    print_freq = 20

    total_samples = 0
    total_bpd = 0.0

    for _, (samples, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        loss = 0.0
        # Average multiple forward passes for a stable NLL estimate.
        for _ in range(N):
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    one_loss = model(samples, labels)
                    loss += one_loss
        loss /= N
        loss_value = loss.item()

        # convert loss to bits/dim
        bpd_value = loss_value / math.log(2)
        total_samples += samples.size(0)
        total_bpd += bpd_value * samples.size(0)

        torch.cuda.synchronize()
        metric_logger.update(bpd=bpd_value)

    print("BPD: {:.5f}".format(total_bpd / total_samples))


def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None):
    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    num_steps = args.num_series // (batch_size * world_size) + 1

    # Construct the folder name for saving generated series.
    save_folder = os.path.join(
        args.output_dir,
        "ariter{}-temp{}-{}cfg{}-filter{}-series{}".format(
            args.num_iter_list, args.temperature, args.cfg_schedule,
            args.cfg, args.filter_threshold, args.num_series
        )
    )
    if args.evaluate_gen:
        save_folder += "_evaluate"
    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Ensure that the number of series per class is equal.
    class_num = args.class_num
    assert args.num_series % class_num == 0, "Number of series per class must be the same"
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_series // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    used_time = 0.0
    gen_img_cnt = 0

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        start_idx = world_size * batch_size * i + local_rank * batch_size
        end_idx = start_idx + batch_size
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_gen = torch.Tensor(labels_gen).long().cuda()

        torch.cuda.synchronize()
        start_time = time.time()

        # generation
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                class_embedding = model_without_ddp.class_emb(labels_gen)
                if not args.cfg == 1.0:
                    # Concatenate fake latent for classifier-free guidance.
                    class_embedding = torch.cat(
                        [class_embedding, model_without_ddp.fake_latent.repeat(batch_size, 1)],
                        dim=0
                    )
                sampled_series = model_without_ddp.sample(
                    cond_list=[class_embedding for _ in range(args.num_conds)],
                    num_iter_list=[int(num_iter) for num_iter in args.num_iter_list.split(",")],
                    cfg=args.cfg, cfg_schedule=args.cfg_schedule,
                    temperature=args.temperature,
                    filter_threshold=args.filter_threshold,
                    fractal_level=0
                )

        # Measure generation speed (skip first batch).
        torch.cuda.synchronize()
        batch_time = time.time() - start_time
        if i >= 1:
            used_time += batch_time
            gen_img_cnt += batch_size
            print("Generating {} series takes {:.5f} seconds, {:.5f} sec per series".format(gen_img_cnt, used_time, used_time / gen_img_cnt))

        torch.distributed.barrier()

        sampled_series = sampled_series.detach().cpu()

        # distributed save series
        for b_id in range(sampled_series.size(0)):
            img_id = i * sampled_series.size(0) * world_size + local_rank * sampled_series.size(0) + b_id
            if img_id >= args.num_series:
                break
            gen_img = np.round(np.clip(sampled_series[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

    torch.distributed.barrier()
    time.sleep(10)

    # compute FID and IS
    if log_writer is not None:
        if args.img_size == 64:
            fid_statistics_file = 'fid_stats/adm_in64_stats.npz'
        elif args.img_size == 256:
            fid_statistics_file = 'fid_stats/adm_in256_stats.npz'
        else:
            raise NotImplementedError
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        postfix = "_cfg{}".format(args.cfg)
        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
        if not args.evaluate_gen:
            # remove temporal saving folder for online eval
            shutil.rmtree(save_folder)

    torch.distributed.barrier()
    time.sleep(10)
