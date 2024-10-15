import torch
import random
from .base_model import BaseModel
from . import networks3D
import copy
from medclip import MedCLIPTextModel, MedCLIPProcessor 
from models.GaussianDiffusion import GaussianDiffusion
from models.gaussiandiffusion_modules.unet import create_model
import os
from apex import amp

    
   
 
from torch.utils.tensorboard import SummaryWriter
class medddpmmodel(BaseModel):
    def name(self):
        return "medddpm_model"
    
    @staticmethod
    def modify_commandline_options(parser, is_train=True): 
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument(
                "--ema_decay",
                type=float,
                default=0.995,
                help="Decay rate for the Exponential Moving Average of model parameters."
            ) 
            parser.add_argument(
                "--step_start_ema",
                type=int,
                default=2000,
                help="Step to start updating the Exponential Moving Average of model parameters."
            )
            parser.add_argument(
                "--update_ema_every",
                type=int,
                default=10,
                help="Frequency of updates to the Exponential Moving Average of model parameters."
            )
            parser.add_argument(
                "--save_and_sample_every",
                type=int,
                default=1000,
                help="Frequency of saving checkpoints and sampling."
            )
            parser.add_argument(
                "--with_condition",
                action='store_true',
                help="Train with condition. Set to True if training with conditions."
            )
            parser.add_argument(
                "--with_pairwised",
                action='store_true',
                default=False,
                help="Train with pairwised data. Set to True if training with pairwised data."
            )
            parser.add_argument("--timesteps", type = int, default = 250)
            parser.add_argument("--num_channels", type = int, default = 64)
            parser.add_argument(
                "--num_res_blocks",
                type = int,
                default=1, 
            ) 
            parser.add_argument(
            "--fp16",
            action='store_true',
            help="Use 16-bit floating-point precision in training. Set to True to enable."
            )
            parser.add_argument("--betaschedule", type = str, default = "cosine") 

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.opt = opt

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = [
            "G",  
            "G_A", 
        ]
        self.visual_names = ["real_A", "fake_B"]
        self.model_names = ["G_A"]
        self.step = 0 
    
        
        self.backbone = create_model(self.opt.patch_size[0], self.opt.num_channels, self.opt.num_res_blocks, in_channels=self.opt.input_nc, out_channels=self.opt.output_nc).to(self.opt.device)

        self.netG_A =  GaussianDiffusion(self.backbone,image_size = self.opt.patch_size[0],depth_size = self.opt.patch_size[2],timesteps =self.opt.timesteps,  loss_type = 'l1',  with_condition=self.opt.with_condition,channels=self.opt.output_nc,betaschedule=opt.betaschedule).to(self.opt.device)

        self.ema = EMA(self.opt.ema_decay)
        self.ema_model = copy.deepcopy(self.netG_A).to(self.opt.device)
        self.update_ema_every = self.opt.update_ema_every

        self.step_start_ema = self.opt.step_start_ema
        # self.save_and_sample_every = self.opt.save_and_sample_every

        # 半精度训练, 节约显存
        self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(), lr=self.opt.lr) 
        self.train_batch_size = self.opt.batch_size
        self.with_condition = self.opt.with_condition

        self.fp16 = opt.fp16
        # self.fp16 = False

        if self.fp16:
            (self.netG_A, self.ema_model), self.opt = amp.initialize([self.netG_A, self.ema_model], self.optimizer_G, opt_level='O3')
    
 
        
        self.results_folder = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(self.results_folder, exist_ok=True)
        # self.log_dir = self.create_log_dir()
        self.writer = SummaryWriter(log_dir=self.results_folder)
        self.reset_parameters()

         
        # clip model settings
        # bert_type = "medicalai/ClinicalBERT"
        bert_type = "emilyalsentzer/Bio_ClinicalBERT"
        self.clip_text_encoder = MedCLIPTextModel( bert_type = bert_type).to(self.opt.device)
            
           
        # Freeze all parameters in clip_text_encoder
        self.set_requires_grad([self.clip_text_encoder], False)
        
    
        self.optimizers = []
        self.optimizers.append(self.optimizer_G)


        self.token_processor = MedCLIPProcessor()
 


    def set_input(self, input):
        self.real_A = input["labelmask"].to(self.opt.device)
        print('self.real_A: ', self.real_A.shape)
        self.real_B = input["image"].to(self.opt.device)
        print('self.real_B: ', self.real_B.shape)  

        self.read_A_prompt = self.token_processor(
            text=input["prompt"], 
            images=None, 
            return_tensors="pt", 
            padding=True
        ).to(self.opt.device)
        self.image_id = input["image_id"] 

        # self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):

        self.read_A_prompt_embeddings = self.clip_text_encoder(self.read_A_prompt["input_ids"],self.read_A_prompt["attention_mask"])  

        # print('self.read_A_prompt_embeddings: ', self.read_A_prompt_embeddings.shape)
        print('self.real_A: ', self.real_A.shape)

        if len(self.real_A.shape) < 5:
            self.real_A = self.real_A.unsqueeze(0)

        loss= self.netG_A(self.real_B, condition_tensors=self.real_A)
        self.loss_G_A = loss.sum()/self.opt.batch_size


    def backward_G(self): 

        self.loss_G = self.loss_G_A

        if self.fp16:
            with amp.scale_loss(self.loss_G, self.optimizer_G) as scaled_loss:
                scaled_loss.backward()
        else:
            self.loss_G.backward()
      

    def optimize_parameters(self, **kwargs):
        # forward
         
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.step = kwargs["step"]

        if kwargs["step"] % self.opt.update_ema_every == 0:
            self.step_ema(kwargs["step"])
    
    def test(self):
        with torch.no_grad():
            batches = self.num_to_groups(1, self.opt.batch_size)

            
            if len(self.real_A.shape) < 5:
                self.real_A = self.real_A.unsqueeze(0)
            
            self.sample_input = self.real_A.transpose(4, 2)[:1,:,:,:,:]

            all_images_list = list(map(lambda n: self.ema_model.sample(batch_size=n, condition_tensors=self.sample_input), batches))
            print('all_images_list: ', all_images_list)
            print('all_images_list: ', len(all_images_list))

            all_images = torch.cat(all_images_list, dim=0)
            

            self.fake_B = all_images.transpose(4, 2)
            print('fake_B: ', self.fake_B.shape) 

        return self.image_id
    

    def save_networks(self, milestone):
        model_to_save_net_G = self.netG_A.module if hasattr(self.netG_A, 'module') else self.netG_A
        model_to_save_ema = self.ema_model.module if hasattr(self.ema_model, "module") else self.ema_model
        
        # Assuming `self.optimizer_G` is already defined as the optimizer for `self.netG_A`
        optimizer_G_state = self.optimizer_G.state_dict()

        data = {
            'step': self.step,
            'model': model_to_save_net_G.state_dict(),
            'ema': model_to_save_ema.state_dict(),
            'optimizer_G': optimizer_G_state  # Save the state of the optimizer
        }
        
        torch.save(data, os.path.join(self.results_folder, f'model-{milestone}.pth'))

        
    def load_networks(self, milestone):
        data = torch.load(os.path.join(self.results_folder, f'model-{milestone}.pth'), map_location=self.opt.device)
        
        self.step = data['step']
        self.netG_A.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

        # Load the optimizer state if it exists in the checkpoint
        if 'optimizer_G' in data:
            self.optimizer_G.load_state_dict(data['optimizer_G'])
        else:
            print("Warning: No optimizer state found in the checkpoint! Optimizer will be reinitialized.")

    def num_to_groups(self, num, divisor):
        groups = num // divisor
        remainder = num % divisor
        arr = [divisor] * groups
        if remainder > 0:
            arr.append(remainder)
        return arr
    
    def get_current_visuals(self, **others):
        self.test()
        
        # self.fake_B=all_images.reshape([self.opt.patch_size[0], self.opt.patch_size[1], self.opt.patch_size[2]])

        return super().get_current_visuals()
  

    # ------ other model implementations for ema -----

    def step_ema(self,step):
        if step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model,  self.netG_A)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.netG_A.state_dict())
     

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
