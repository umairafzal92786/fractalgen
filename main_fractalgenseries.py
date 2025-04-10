import argparse
import datetime
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.utils
import torch.utils.data
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

import util.misc as misc
from engine_fractalgenseries import compute_nll, evaluate, train_one_epoch
from models import fractalgentimeseries
from util.misc import NativeScalerWithGradNormCount as NativeScaler


class StockCSVFolderDataset(Dataset):
    def __init__(self, root_dir, series_len, normalize=True):
        self.series_len = series_len
        self.normalize = normalize
        self.samples = []

        for fname in os.listdir(root_dir):
            if fname.endswith(".csv"):
                csv_path = os.path.join(root_dir, fname)
                label = os.path.splitext(fname)[0]
                self.samples.append((csv_path, label))

        labels = sorted(set(label for _, label in self.samples))
        self.label_to_idx = {label: idx for idx, label in enumerate(labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        csv_path, label = self.samples[idx]
        df = pd.read_csv(csv_path)
        df = df.drop(columns=['Ticker', 'Date'], errors='ignore')
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)

        series = df.values

        if series.shape[0] < self.series_len:
            pad_len = self.series_len - series.shape[0]
            pad = np.zeros((pad_len, series.shape[1]))
            series = np.vstack([series, pad])
        else:
            series = series[:self.series_len, :]

        if self.normalize:
            # Normalize each feature (column-wise standardization)
            mean = series.mean(axis=0, keepdims=True)
            std = series.std(axis=0, keepdims=True) + 1e-6  # avoid divide by zero
            series = (series - mean) / std

        series_tensor = torch.tensor(series, dtype=torch.float32)
        label_idx = self.label_to_idx[label]
        return series_tensor, label_idx



def get_args_parser():
    parser = argparse.ArgumentParser('Fractal Generative Models', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * # GPUs)')
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='Folder that contains checkpoint to resume from')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Model parameters
    parser.add_argument('--model', default='fractaltimeseriesmar_in64', type=str, metavar='MODEL',
                        help='Name of the model to train')
    parser.add_argument('--series_len', default=1024, type=int, help='series length')

    # Generation parameters
    parser.add_argument('--num_iter_list', default='64,16', type=str,
                        help='Number of autoregressive iterations for each fractal level')
    parser.add_argument('--num_series', default=10, type=int,
                        help='Number of series to generate')
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Classifier-free guidance factor')
    parser.add_argument('--cfg_schedule', default='linear', type=str)
    parser.add_argument('--temperature', default=1.0, type=float,
                        help='Sampling temperature')
    parser.add_argument('--filter_threshold', default=1e-4, type=float,
                        help='Filter threshold for low probability tokens in cfg')
    parser.add_argument('--label_drop_prob', default=0.1, type=float)
    parser.add_argument('--eval_freq', type=int, default=40,
                        help='Frequency (in epochs) for evaluation')
    parser.add_argument('--save_last_freq', type=int, default=5,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--evaluate_nll', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=1,
                        help='Generation batch size')
    parser.add_argument('--nll_bsz', type=int, default=1,
                        help='NLL evaluation batch size')
    parser.add_argument('--nll_forward_number', type=int, default=1,
                        help='Number of forward passes used to evaluate the NLL for each data sample. '
                             'This does not affect the NLL of  AR model, but for the MAR model, multiple passes (each '
                             'randomly sampling a masking ratio) result in a more accurate NLL estimation.'
    )
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay (default: 0.05)')
    parser.add_argument('--grad_checkpointing', action='store_true')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='cosine',
                        help='Learning rate schedule')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='Epochs to warm up LR')

    # Fractal generator parameters
    parser.add_argument('--num_conds', type=int, default=1,
                        help='Number of conditions to use')
    parser.add_argument('--grad_clip', type=float, default=3.0,
                        help='Gradient clipping value')
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.1,
                        help='Projection dropout rate')

    # Dataset parameters
    parser.add_argument('--data_path', default='./raw_data', type=str,
                        help='Path to the dataset')
    parser.add_argument('--class_num', default=483, type=int)
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # Set up TensorBoard logging (only on main process)
    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    # Data augmentation transforms (following DiT and ADM)
    # transform_train = transforms.Compose([
    #     transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    # transform_val = transforms.Compose([
    #     transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])



    dataset_train = StockCSVFolderDataset(os.path.join(args.data_path, 'train'), series_len=args.series_len, normalize=True)
    dataset_val = StockCSVFolderDataset(os.path.join(args.data_path, 'val'), series_len=args.series_len, normalize=True)
    
    # sample = dataset_train[0]
    # print("Sample shape:", sample[0].shape)
    # print("Sample label:", dataset_train.idx_to_label[sample[1]])
    # print("Dataset length:", len(dataset_train))


    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train =", sampler_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    #check shape of data_loader_train
    # sample = next(iter(data_loader_train))
    # print(len(sample[0]))
    # print("Sample shape:", sample[0].shape)
    
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, shuffle=True,
        batch_size=args.nll_bsz,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # Create fractal generative model
    model = fractalgentimeseries.__dict__[args.model](
        label_drop_prob=args.label_drop_prob,
        class_num=args.class_num,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        num_conds=args.num_conds,
        grad_checkpointing=args.grad_checkpointing
    )

    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.2f}M".format(n_params / 1e6))

    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0])
    model.to(device)
    # model_without_ddp = model.module
    model_without_ddp = model

    # Set up optimizer with weight decay adjustment for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    # Resume from checkpoint if provided
    checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resumed checkpoint from", args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("Loaded optimizer & scaler state!")
        del checkpoint
    else:
        print("Training from scratch")

    # Evaluation modes
    if args.evaluate_gen:
        torch.cuda.empty_cache()
        evaluate(model_without_ddp, args, 0, batch_size=args.gen_bsz, log_writer=log_writer)
        return

    if args.evaluate_nll:
        torch.cuda.empty_cache()
        compute_nll(model, data_loader_val, device, N=args.nll_forward_number)
        return

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, loss_scaler, log_writer=log_writer, args=args
        )

        # Save checkpoint periodically
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
                epoch_name="last"
            )

        # Perform online evaluation at specified intervals
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            evaluate(model_without_ddp, args, epoch, batch_size=args.gen_bsz, log_writer=log_writer)
            torch.cuda.empty_cache()

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
