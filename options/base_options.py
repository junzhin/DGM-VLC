import argparse
import os
from utils.utils import *
import models


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--name', type=str, default='debugging',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--data_path', type=str,
                            default='../../data/3D-CycleGan-Pytorch_data', help='Train images path')
        parser.add_argument('--generate_path', type=str,
                            default='img', help='path for target generation')
        parser.add_argument('--input_path', type=str,
                            default='gtlung', help='path for input')
        parser.add_argument('--segmentation', type=int, default=1,
                            help='Whether the input mask is segmentation mask. 0 = not segmentation, 1= segmentation')
        parser.add_argument(
            '--val_path', type=str, default='/media/xiaodan/3d_synthesis/Validate/', help='Validation images path')
        parser.add_argument('--batch_size', type=int,
                            default=2, help='input batch size')
        parser.add_argument('--patch_size', default='64,64,32',
                            help='Size of the patches extracted from the image')
        parser.add_argument('--input_nc', type=int, default=4,
                            help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=1,
                            help='# of output image channels')
        parser.add_argument('--resample', default=True,
                            help='Decide or not to rescale the images to a new resolution')
        parser.add_argument('--new_resolution', default='1,1,2',
                            help='New resolution (if you want to resample the data again during training')
        parser.add_argument('--min_pixel', default=0.1,
                            help='Percentage of minimum non-zero pixels in the cropped label')
        parser.add_argument('--drop_ratio', default=0, type=float,
                            help='Probability to drop a cropped area if the label is empty. All empty patches will be dropped for 0 and accept all cropped patches if set to 1')

        parser.add_argument('--ngf', type=int, default=64,
                            help='# of gen filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=64,
                            help='# of discrim filters in first conv layer')
        parser.add_argument('--netD', type=str, default='n_layers',
                            help='selects model to use for netD, not used in diffusion-type model')
        parser.add_argument('--n_layers_D', type=int,
                            default=3, help='only used if netD==n_layers')
        parser.add_argument('--netG', type=str, default='resnet_9blocks_with_text_encoder3',
                            help='selects model to use for netG. Look on Networks3D to see the all list, not used in diffusion-type model')

        parser.add_argument('--gpu_ids', default="1") # 已经被修改只能接受一个gpu 的id
        parser.add_argument('--model', type=str, default='medddpmtext',
                            help='chooses which model to use. cycle_gan')
        
        parser.add_argument('--which_direction', type=str,
                            default='AtoB', help='AtoB or BtoA (keep it AtoB)')
        parser.add_argument('--checkpoints_dir', type=str,
                            default='./checkpoints', help='models are saved here')
        parser.add_argument('--workers', default=1, type=int,
                            help='number of data loading workers')
        parser.add_argument('--norm', type=str, default='instance',
                            help='instance normalization or batch normalization')

        parser.add_argument('--no_dropout', action='store_true',
                            help='no dropout for the generator')
        parser.add_argument('--init_type', type=str, default='kaiming',
                            help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--verbose', action='store_true',
                            help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str,
                            help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}') 

        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with the new defaults

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):

        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))
                      ) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # # set gpu ids
        str_ids = list(opt.gpu_ids)
        # # str_ids.remove(',')
        str_ids = [i for i in str_ids if i.isdigit()]
        # print('gpu_ids: ', str_ids)
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        resolutions = opt.patch_size.split(',')
        opt.patch_size = [int(a) for a in resolutions]

        resolutions = opt.new_resolution.split(',')
        opt.new_resolution = tuple([float(a) for a in resolutions])

        self.opt = opt
        return self.opt