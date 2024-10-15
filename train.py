import sys
from  utils.NiftiPromptDataset import  NiftiPromptDataset
import  utils.NiftiDataset as NiftiDataset
from torch.utils.data import DataLoader, DistributedSampler
from options.train_options import TrainOptions 
import time
from collections import OrderedDict
from models import create_model
from utils.visualizer import Visualizer
from utils import util
from test import inference
import torch
import random
import numpy as np
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel  
from torch.distributed import init_process_group

import warnings

# Ignore all user warnings
warnings.filterwarnings("ignore", category=UserWarning)

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
def DDP_step_up():
    init_process_group(backend='nccl')

DDP_flag= False
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_gpu_memory():
    torch.cuda.synchronize() # Make sure all operations are done
    return torch.cuda.memory_allocated() / 1e9

def DDP_reduce_loss_dict(loss_dict, world_size):
    """
    Reduce the loss dictionary across all processes so that all of them have the averaged results.
    """
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        
        # Stack all losses into a single tensor and reduce
        all_losses = torch.stack(all_losses, 0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # Only the master process will get the correct results
            all_losses /= world_size
            reduced_losses = {name: loss for name, loss in zip(loss_names, all_losses)}
        else:
            # Other processes get dummy data
            reduced_losses = {name: torch.tensor(0) for name in loss_names}
    return reduced_losses

if __name__ == '__main__':
    # seed = 50
    # seed_everything(seed)
    # -----  Loading the init options -----
    opt = TrainOptions().parse() # get training options
    print("--------------------------------------------->:",os.environ.get('CUDA_VISIBLE_DEVICES'))

    if DDP_flag is True:
        DDP_step_up()
        local_rank = int(os.environ["LOCAL_RANK"])
        # world_size = dist.get_world_size()
        # world_size = dist.get_world_size()
        world_size = dist.get_world_size()
        print(torch.cuda.is_available())
        opt.device = torch.device(local_rank)
        print('----------------->world_size: ', world_size)
        print('----------------->local_rank: ', local_rank) 
    else:
        opt.device = torch.device(int(opt.gpu_ids[0]))
    print('opt.device : ', opt.device)

    # Check for CUDA's availability
    cuda_available = torch.cuda.is_available()

    # Print whether CUDA is available
    print(f"CUDA is available: {cuda_available}")

    # If CUDA is available, print the number and name of available GPUs
    if cuda_available:
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPUs detected.")
 
    # -----------------------------------------------------
    model = create_model(opt)  # creation of the model
    model.setup(opt)
    if opt.epoch_count > 1:
        model.load_networks(opt.which_epoch)
    visualizer = Visualizer(opt)
    total_steps = 0

    if DDP_flag is True:
        model = DistributedDataParallel(model, device_ids=[local_rank])
  

    # -----  Transformation and Augmentation process for the data  -----
    min_pixel = int(opt.min_pixel * ((opt.patch_size[0] * opt.patch_size[1] * opt.patch_size[2]) / 50)) 
    trainTransforms = [
        NiftiDataset.Padding(
            (opt.patch_size[0], opt.patch_size[1], opt.patch_size[2])),
        NiftiDataset.RandomCrop(
            (opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]), opt.drop_ratio, min_pixel),
    ]

 

    train_set = NiftiPromptDataset(opt.data_path, path_A=opt.generate_path, path_B=opt.input_path,
                             which_direction='AtoB', transforms=trainTransforms, shuffle_labels=False, train=True,
                             add_graph=False, label2mask=True if opt.model in ["medddpm", "medddpmtext", "medddpmtextcross", "medddpmvisualprompt"] else False)
    print('lenght train list:', len(train_set)) 
    
    # Check if the dataset has a 'collate_fn' method
    if hasattr(train_set, 'collate_fn'):
        # If it does, use it in the DataLoader
        train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=True, collate_fn=train_set.collate_fn)
    else:
        # If it does not, use the DataLoader without specifying a custom collate_fn
        if DDP_flag is False:
            train_loader = DataLoader(train_set, batch_size=opt.batch_size,shuffle=True, num_workers=opt.workers, pin_memory=True)
        else:
            sampler = DistributedSampler(train_set, shuffle=True)
            train_loader = DataLoader( train_set, batch_size=opt.batch_size, sampler=sampler, num_workers=opt.workers,pin_memory=True)

 


    initial_memory = get_gpu_memory()
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(train_loader):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters(step=total_steps)
            memory_used_by_batch = get_gpu_memory() - initial_memory

            losses = model.get_current_losses()
            t = (time.time() - iter_start_time) / opt.batch_size

            # The `visualizer` is a helper class that provides functions for visualizing and printing
            # the current losses and results during training. It is used to display and save images
            # and loss values for monitoring the training progress.
            if DDP_flag is True and local_rank == 0:
                for key in losses.keys():
                    losses  = DDP_reduce_loss_dict(losses, world_size=world_size) 
            visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)

            print('I finished (epoch %d, total_steps %d/)' %
                  (epoch, total_steps))
            if total_steps % opt.print_freq == 0 or (DDP_flag is True and local_rank == 0):
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(
                    epoch, epoch_iter, losses, t, t_data)
                

                # ------- visualisation of 3D slicer -----------
                z_slice = 31
                generated = model.get_current_visuals(epoch=epoch)
                generated = generated['fake_B']
                print("generated shape:", generated.shape)
                if opt.segmentation == 1:
                    if data['label'].shape[1] > 1:
                        input_label = util.tensor2label(
                            data['label'][0, :1, :, :, z_slice], 5)
                    else:
                        input_label = util.tensor2label(
                            data['label'][0, :, :, :, z_slice], 5)
                else:
                    input_label = util.tensor2im(data['label'][0, :, :, :, z_slice])
                visuals = OrderedDict([('input_label', input_label),
                                       # ('input_label', util.tensor2label(data[1][0, :, :, :, z_slice], 5)),
                                       ('synthesized_image', util.tensor2im(  # `generated` is a
                                           # variable that stores the
                                           # output of the model's
                                           # forward pass. It
                                           # represents the
                                           # synthesized image
                                           # generated by the model.
                                           generated.data[0, :, :, :, z_slice])),
                                       ('real_image', util.tensor2im(data['image'][0, :, :, :, z_slice]))])

                visualizer.display_current_results(visuals, epoch, total_steps)
                # ------- visualisation of 3D slicier -----------


            if (total_steps % opt.save_latest_freq == 0) or (DDP_flag is True and local_rank == 0):
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')
            print(f'------------------>  Memory_used_by_batch: {memory_used_by_batch}GB')

            iter_data_time = time.time()

        if (epoch % opt.save_epoch_freq == 0)  or  (DDP_flag is True and local_rank == 0):
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

