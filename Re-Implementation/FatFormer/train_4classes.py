import torch
import random
import numpy as np
import argparse
import os
import torch.distributed as dist
from utils.misc import get_rank, init_distributed_mode
from utils.dataset import Dataset_Creator_4classes
import utils.misc as utils
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from models import build_model
from sklearn.metrics import average_precision_score, accuracy_score
from tensorboardX import SummaryWriter

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # dataset parameters
    parser.add_argument('--dataset_path', type=str, default='./dataset/ProGAN')
    parser.add_argument('--img_resolution', type=int, default=256)
    parser.add_argument('--crop_resolution', type=int, default=224)
    # parser.add_argument('--test_selected_subsets', nargs='+', required=True)
    parser.add_argument('--batchsize', type=int, default=50)

    # model
    parser.add_argument('--name', type=str, default='FatFormer')
    parser.add_argument('--backbone', type=str, default='CLIP:ViT-L/14')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_vit_adapter', type=int, default=8)

    # text encoder
    parser.add_argument('--num_context_embedding', type=int, default=8)
    parser.add_argument('--init_context_embedding', type=str, default="")

    # frequency
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--clip_vision_width', type=int, default=1024)
    parser.add_argument('--frequency_encoder_layer', type=int, default=2)
    parser.add_argument('--decoder_layer', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=12)

    # training
    parser.add_argument('--epochs', default=25, type=int)

    # output
    parser.add_argument('--pretrained_model', type=str, default="")
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--print_freq', default=50, type=int)
    parser.add_argument('--save_latest_freq', default=10000, type=int)
    parser.add_argument('--save_dir', type=str, default="/data/home/wanghy/FatFormer/4classes_results")

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')

    return parser



def save_models(epoch, save_dir, model, optimizer, total_steps):
    save_filename = 'model_epoch_%s.pth' % epoch
    save_path = os.path.join(save_dir, save_filename)

    # serialize model and optimizer to dict
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'total_steps': total_steps,
    }

    torch.save(state_dict, save_path)


@torch.no_grad()
def validate(model, data_loader, device, args=None, test=False):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = args.print_freq

    y_true, y_pred = [], []
    for samples in metric_logger.log_every(data_loader, print_freq, header):
        images, labels = [sample.to(device) for sample in samples]
        outputs = model(images)

        y_pred.extend(outputs.softmax(dim=1)[:, 1].flatten().tolist())
        y_true.extend(labels.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)

    return r_acc, f_acc, acc, ap

def main(args):
    init_distributed_mode(args)

    # set device
    device = torch.device(args.device)

    # fix the seed
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

    # training dataset
    dataset_creator = Dataset_Creator_4classes(dataset_path=args.dataset_path, batch_size=args.batchsize,
                                      num_workers=args.num_workers, img_resolution=args.img_resolution,
                                      crop_resolution=args.crop_resolution)

    train_dataset = dataset_creator.build_dataset("train", selected_subsets='all')
    val_dataset = dataset_creator.build_dataset("val", selected_subsets='all')

    if args.distributed:
        sampler_train = DistributedSampler(train_dataset, shuffle=False)
        sampler_val = DistributedSampler(val_dataset, shuffle=False)

    else:
        sampler_train = SequentialSampler(train_dataset)
        sampler_val = SequentialSampler(val_dataset)

    train_loader = DataLoader(train_dataset, args.batchsize, sampler=sampler_train,
                                                       drop_last=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, args.batchsize, sampler=sampler_val,
                                  drop_last=False, num_workers=args.num_workers)

    train_writer = SummaryWriter(os.path.join(args.save_dir, "train"))
    val_writer = SummaryWriter(os.path.join(args.save_dir, "val"))

    # build model
    model = build_model(args)
    for name, param in model.named_parameters():
        if 'clip_model.visual' in name:
            if 'forgery_aware_adapter' in name:
                param.requires_grad = True
            elif 'ln_post' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        elif 'clip_model' in name:
            if 'clip_model.transformer' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        else:
            param.requires_grad = True

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
    else:
        model = model.to(device)

    # Optimizer setting strictly following the original paper
    # learning_rate = 4e-4, betas = (0.9, 0.999)
    # decay_factor = 0.9, decay_epoch = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    total_steps = 0
    best_ap = 0.0

    # training
    # training epochs is set to 25 strictly following the original paper
    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            total_steps += 1

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if total_steps % args.print_freq == 0:
                print("Train loss: {} at step: {}".format(loss.item(), total_steps))
                train_writer.add_scalar('loss', loss.item(), total_steps)


            if total_steps % args.save_latest_freq == 0:
                print('saving the latest model %s (epoch %d, model.total_steps %d)' %
                      (args.name, epoch, total_steps))
                save_models('latest', args.save_dir, model, optimizer, total_steps)

        # Validation
        model.eval()
        with torch.no_grad():
            r_acc, f_acc, acc, ap = validate(model, val_loader, device, args)

        val_writer.add_scalar('r_acc', r_acc, total_steps)
        val_writer.add_scalar('f_acc', f_acc, total_steps)
        val_writer.add_scalar('acc', acc, total_steps)
        val_writer.add_scalar('ap', ap, total_steps)

        print("(Val @ epoch {}) r_acc: {} ; f_acc: {} ; acc: {} ; ap: {} ;".format(epoch, ap, r_acc, f_acc,
                                                                                           acc, ap))
        # save best model according to AP
        if ap > best_ap:
            best_ap = ap
            print(f'New best model found at epoch {epoch + 1} with AP: {ap:.4f}')
            save_models('best', args.save_dir, model, optimizer, total_steps)

        scheduler.step()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
