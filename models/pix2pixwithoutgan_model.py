import torch
import itertools
import random
from .base_model import BaseModel
from . import networks3D 
from models.model_helper import define_D, define_G

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(
                        0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images


class Pix2pixWithoutGanModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_identity', type=float, default=2,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of '
                                'scaling the weight of the identity mapping loss. For example, if the weight of the'
                                ' identity loss should be 10 times smaller than the weight of the reconstruction loss, '
                                'please set lambda_identity = 0.1')
            '''
            adjust the weight of correlation coefficient loss
            '''
            parser.add_argument('--lambda_co_A', type=float, default=2,
                                help='weight for correlation coefficient loss (A -> B)')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.opt = opt

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G', 'idt_A', 'cor_coe_GA']
        self.visual_names = ['real_A', 'fake_B']

        if self.isTrain:
            self.model_names = ['G_A']
        else:  # during test time, only load Gs
            self.model_names = ['G_A']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,   # nc number channels
                                          not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            
            # print('use_sigmoid: ', use_sigmoid)
            # self.netD_A = networks3D.define_D(opt.output_nc, opt.ndf, opt.netD,opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            # `lsgan` is a flag that determines whether to
            # use least squares GAN (LSGAN) loss or not.
            # LSGAN loss is an alternative to the
            # traditional binary cross entropy GAN loss. It
            # aims to improve the stability and convergence
            # of GAN training by using a least squares loss
            # function. In LSGAN, the discriminator tries
            # to minimize the squared difference between
            # the predicted output and the target output,
            # while the generator tries to maximize it.
            # This leads to more stable training and better
            # image quality in some cases.
            self.criterionGAN = networks3D.GANLoss(
                use_lsgan=not opt.no_lsgan).to(self.opt.device)

            # self.criterionIdt = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.MSELoss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_D = torch.optim.Adam(self.netD_A.parameters(),
                                                # lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            # self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        # AtoB = self.opt.which_direction == 'AtoB'
        # self.real_A = input[0 if AtoB else 1].to(self.opt.device)
        # self.real_B = input[1 if AtoB else 0].to(self.opt.device)
        self.real_A = input["label"].to(self.opt.device)
        self.real_B = input["image"].to(self.opt.device)
        self.image_id = input["image_id"]

        # self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    # def backward_D_A(self):
    #     fake_B = self.fake_B_pool.query(self.fake_B)
    #     self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        '''
        lambda_coA & lambda_coB
        '''

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.

            self.loss_idt_A = self.criterionIdt(
                self.fake_B, self.real_B) * lambda_idt

        else:
            self.loss_idt_A = 0

        # GAN loss D_A(G_A(A))
        # self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)

        '''
        self.cor_coeLoss
        '''
        self.loss_cor_coe_GA = networks3D.Cor_CoeLoss(self.fake_B,
                                                      self.real_B) * 0.5  # fake ct & real mr; Evaluate the Generator of ct(G_A)

        # combined loss
        # self.loss_G = self.loss_G_A + self.loss_idt_A + self.loss_cor_coe_GA
        self.loss_G = self.loss_idt_A + self.loss_cor_coe_GA


        self.loss_G.backward()

    def optimize_parameters(self, **others):
        # forward
        self.forward()
        # 去除这个部分的鉴别器的loss 函数
        # # D_A and D_B
        # self.set_requires_grad([self.netD_A], True)
        # self.set_requires_grad([self.netG_A], False)
        # self.optimizer_D.zero_grad()
        # self.backward_D_A()
        # self.optimizer_D.step()
        # G_A and G_B
        # self.set_requires_grad([self.netD_A], False)

        self.set_requires_grad([self.netG_A], True)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    
    def get_current_visuals(self, **others):
        self.test()

        # self.fake_B=all_images.reshape([self.opt.patch_size[0], self.opt.patch_size[1], self.opt.patch_size[2]])

        return super().get_current_visuals()


    def test(self):
        with torch.no_grad():
            self.forward()
        
        return self.image_id