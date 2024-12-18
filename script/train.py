import os
import time
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp

from data import create_dataloader
from script.validate import validate
from script.earlystop import EarlyStopping
from options.train_options import TrainOptions

"""Currently assumes jpg_prob, blur_prob 0 or 1"""


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.data_label = 'val'
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt


def train_worker(rank, world_size, opt):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(opt.ddp_port)

    # initialize the process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )

    opt.device = rank
    val_opt = get_val_opt()
    val_opt.device = rank

    val_opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.val_split)
    opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)

    train_dataloader = create_dataloader(opt)

    if rank == 0:
        print("Length of data loader: %d" % (len(train_dataloader)))
        train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
        val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
        early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    dist.barrier()
    stop_now = False

    # build trainer
    if opt.detection_model == 'LGrad':
        from models.trainers import LGradTrainer
        model = LGradTrainer(opt)
    elif opt.detection_model == 'LNP':
        from models.trainers import LNPTrainer
        model = LNPTrainer(opt)
    elif opt.detection_model == 'GramNet':
        from models.trainers import GramNetTrainer
        model = GramNetTrainer(opt)
    elif opt.detection_model == 'FreDect':
        from models.trainers import FreDectTrainer
        model = FreDectTrainer(opt)
    elif opt.detection_model == 'CNNSpot':
        from models.trainers import CNNSpotTrainer
        model = CNNSpotTrainer(opt)
    elif opt.detection_model == 'UnivFD':
        from models.trainers import UnivFDTrainer
        model = UnivFDTrainer(opt)
    elif opt.detection_model == 'NPR':
        from models.trainers import NPRTrainer
        model = NPRTrainer(opt)
    elif opt.detection_model == 'SAFE':
        from models.trainers import SAFETrainer
        model = SAFETrainer(opt)
    elif opt.detection_model == 'FatFormer':
        from models.trainers import FatFormerTrainer
        model = FatFormerTrainer(opt)
    elif opt.detection_model == 'SimE':
        from models.trainers import SimETrainer
        model = SimETrainer(opt)
    else:
        raise ValueError(f'{opt.model} not supported for now')

    start_time = time.time()

    for epoch in range(opt.niter):
        epoch_iter = 0
        model.train()

        for i, data in enumerate(train_dataloader):
            model.total_steps += 1
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if rank == 0:
                if model.total_steps % opt.loss_freq == 0:
                    print('Step {}: Loss ({:.6f})'.format(model.total_steps, model.loss))
                    train_writer.add_scalar('loss', model.loss, model.total_steps)
                    print("Iter time: ", ((time.time() - start_time) / model.total_steps))

                if model.total_steps % opt.save_latest_freq == 0:
                    print('saving the latest model %s (epoch %d, model.total_steps %d)' %
                          (opt.name, epoch, model.total_steps))
                    model.save_networks('latest')

            dist.barrier()

        # Validation
        if rank == 0:
            model.eval()
            acc, ap, r_acc, f_acc, auc = validate(model.model, val_opt, val_opt.device)

            val_writer.add_scalar('accuracy', acc, model.total_steps)
            val_writer.add_scalar('ap', ap, model.total_steps)
            val_writer.add_scalar('auc', auc, model.total_steps)
            print("(Val @ epoch {}) acc: {}; ap: {}; auc: {}".format(epoch, acc, ap, auc))

            early_stopping(ap, model)
            if early_stopping.early_stop:
                cont_train = model.adjust_learning_rate()
                if cont_train:
                    print("Learning rate dropped by 10, continue training...")
                    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
                else:
                    print("Early stopping.")
                    stop_now = True

        dist.barrier()
        if stop_now:
            break

    dist.destroy_process_group()
if __name__ == '__main__':
    opt = TrainOptions().parse()
    mp.spawn(train_worker, args=(opt.ddp_world_size, opt), nprocs=opt.ddp_world_size, join=True)