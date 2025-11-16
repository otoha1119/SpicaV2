"""
Modified MedicalCycleGAN model for SpicaV2

This file is derived from the original `models/medical_cycle_gan_model.py` in the
SpicaV2 repository.  The original implementation extends CycleGAN with
downsample, upsample and SSIM losses inspired by SR‑CycleGAN.  These losses
encourage the generated high‑resolution images to remain close to the
low‑resolution input domain by comparing either a downsampled version of the
generated image to the input image or upsampled/generated images using
structural similarity.  When the goal is to convert energy‑integrating
detector CT (EID‑CT) images into photon‑counting detector CT (PCD‑CT) images
and fully replace the original domain, such losses are undesirable because
they explicitly pull the generator output back towards the EID domain.

The modifications implemented here remove the downsample/upsample and SSIM
losses from the training objective.  Concretely:

* Command‑line options controlling the downsample, upsample and SSIM losses
  (`--lambda_downsample_loss`, `--downsample_loss`, `--lambda_upsample_loss`,
  `--upsample_loss`, `--lambda_clinical_ssim`, `--clinical_ssim_loss`,
  `--lambda_micro_ssim`, `--micro_ssim_loss`) now default to 0.0.  This
  means the losses are effectively disabled.
* `self.loss_names` does not include entries for ``downsample``, ``upsample``,
  ``clinical_ssim`` or ``micro_ssim`` since these losses are no longer
  computed.
* In `backward_G` the sections computing downsample/upsample and SSIM losses
  have been removed.  Instead the corresponding loss attributes are set to
  zero, and the final generator loss only sums the adversarial, cycle and
  identity losses.

These changes simplify the training objective to a plain CycleGAN with
identity losses where applicable.  The goal is to allow the generator to
focus solely on mapping EID‑CT images into the PCD‑CT domain without being
penalised for deviating from the low‑resolution input.
"""

import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torch.nn import AvgPool2d
from torch.nn import UpsamplingNearest2d, UpsamplingBilinear2d
from pytorch_msssim import SSIM
import numpy as np
import os
import cv2
import nibabel as nib
import pdb
import torch.nn.functional as F


class MedicalCycleGANModel(BaseModel):
    """
    MedicalCycleGANModel implements an unpaired image‑to‑image translation
    network based on CycleGAN for medical imaging.  The original SR‑CycleGAN
    implementation included downsample/upsample and SSIM losses to encourage
    structural consistency with the low‑resolution inputs.  In this modified
    version those losses are disabled by default to enable full domain
    replacement from EID‑CT to PCD‑CT.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset‑specific options, and rewrite default values for existing options.

        The SR‑CycleGAN paper introduces a number of additional losses beyond
        the standard CycleGAN objective.  Those losses are controlled by
        command‑line flags.  To disable them in this variant, their default
        weights are set to 0.0.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=1.0,
                                help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=1.0,
                                help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5,
                                help=('use identity mapping. Setting lambda_identity other than 0 has an effect '
                                      'of scaling the weight of the identity mapping loss.')) #default0.2
            # Downsample and upsample losses disabled by default
            parser.add_argument('--lambda_downsample_loss', type=float, default=0.0,
                                help='weight for downsample loss (set to 0.0 to disable)')
            parser.add_argument('--downsample_loss', type=float, default=0.0,
                                help='downsample loss flag (set to 0.0 to disable)')
            parser.add_argument('--lambda_upsample_loss', type=float, default=0.0,
                                help='weight for upsample loss (set to 0.0 to disable)')
            parser.add_argument('--upsample_loss', type=float, default=0.0,
                                help='upsample loss flag (set to 0.0 to disable)')
            # SSIM losses disabled by default
            parser.add_argument('--lambda_clinical_ssim', type=float, default=0.0,
                                help='weight for SSIM between downsampled fake microCT and clinicalCT (set to 0.0 to disable)')
            parser.add_argument('--clinical_ssim_loss', type=float, default=0.0,
                                help='clinical SSIM loss flag (set to 0.0 to disable)')
            parser.add_argument('--lambda_micro_ssim', type=float, default=0.0,
                                help='weight for SSIM between upsampled fake clinicalCT and microCT (set to 0.0 to disable)')
            parser.add_argument('--micro_ssim_loss', type=float, default=0.0,
                                help='micro SSIM loss flag (set to 0.0 to disable)')
            # Additional optional losses retained from original code (random mesh, structure, etc.)
            parser.add_argument('--random_mesh_ssim', type=float, default=-1.0,
                                help='inspired by the paper https://openreview.net/pdf?id=BktMD6isM')
            parser.add_argument('--lambda_random_mesh_ssim', type=float, default=0.5,
                                help='weight for random mesh SSIM loss')
            parser.add_argument('--random_mesh_size', type=int, default=30, help='size of random mesh')  # originally 20
            parser.add_argument('--random_mesh_num', type=int, default=20, help='number of random meshes')  # originally 10
            parser.add_argument('--random_mesh_average', type=int, default=-1, help='use or not use mesh average')
            parser.add_argument('--lambda_random_mesh_average', type=float, default=0.2, help='weight of random mesh average')
            parser.add_argument('--sobel_loss', type=int, default=0,
                                help='use Sobel filter or not (not modified here)')
            parser.add_argument('--structure_loss', type=float, default=-1.,
                                help='use structure loss or not (not modified here)')
            parser.add_argument('--lambda_structure_loss', type=float, default=0.8,
                                help='weight of structure loss')
            parser.add_argument('--lambda_G_A', type=float, default=1.0, help='weight of G_A')
            parser.add_argument('--lambda_G_B', type=float, default=1.0, help='weight of G_B')
            parser.add_argument('--clinical_inter', type=float, default=-1.0,
                                help='use clinical inter loss or not (unchanged)')
            parser.add_argument('--lambda_clinical_inter', type=float, default=5.0,
                                help='weight of clinical inter loss (unchanged)')
            parser.add_argument('--fillhole', type=float, default=-1.0,
                                help='use fill hole loss or not (unchanged)')
            parser.add_argument('--lambda_fillhole', type=float, default=0.1,
                                help='weight of fill hole loss (unchanged)')
        return parser

    def __init__(self, opt):
        """Initialise the MedicalCycleGAN class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # Create SSIM module for potential use; although SSIM losses are disabled by default,
        # the module is still initialised to avoid runtime errors if enabled later.
        # Assume tanh outputs in range [-1,1], therefore data_range=2.0 and channel=1 for CT.
        self.ssim = SSIM(data_range=2.0, channel=1)
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # Downsample, upsample and SSIM losses are omitted here because they are disabled by default.
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A',
                           'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualise G_B(A) and G_A(B)
            visual_names_A.append('idt_A')
            visual_names_B.append('idt_B')
        self.visual_names = visual_names_A + visual_names_B  # combine visualisations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']
        # define networks (both Generators and discriminators)
        self.netG_A = networks.define_G(self.opt.medical_input_nc, self.opt.medical_output_nc, self.opt.ngf,
                                        self.opt.clinical2micronetG, self.opt.norm, not self.opt.no_dropout,
                                        init_type=self.opt.init_type, init_gain=self.opt.init_gain, gpu_ids=self.gpu_ids,
                                        sampling_times=self.opt.sampling_times)
        self.netG_B = networks.define_G(self.opt.medical_input_nc, self.opt.medical_output_nc, self.opt.ngf,
                                        self.opt.micro2clinicalnetG, self.opt.norm, not self.opt.no_dropout,
                                        init_type=self.opt.init_type, init_gain=self.opt.init_gain, gpu_ids=self.gpu_ids,
                                        sampling_times=self.opt.sampling_times)
        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.medical_output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.medical_input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert (opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialise optimisers; schedulers will be automatically created by function <BaseModel.setup>
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre‑processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'  # true or false
        self.real_A = input['clinical' if AtoB else 'micro'].to(self.device)
        self.real_B = input['micro' if AtoB else 'clinical'].to(self.device)
        if getattr(self.opt, 'verbose', False) and not hasattr(self, '_dbg_inp'):
            try:
                print(f"[DBG] real_A {tuple(self.real_A.shape)}  real_B {tuple(self.real_B.shape)}")
            except Exception:
                pass
            self._dbg_inp = True
        self.image_paths = input['clinical_paths' if AtoB else 'micro_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.  We also call loss_D.backward() to calculate the gradients.
        """
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self, epoch):
        """Calculate the loss for generators G_A and G_B."""
        # scale factor for down/up sampling operations (not used when those losses are disabled)
        scale = int(2 ** int(self.opt.sampling_times))
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(F.interpolate(self.real_B, size=self.real_A.shape[-2:], mode='bilinear', align_corners=False))
            
            # print(f"[DBG] idt_A {tuple(self.idt_A.shape)}")
            # print(f"[DBG] real_B {tuple(self.real_B.shape)}")
            # m = AvgPool2d(scale, stride=scale)
            # self.idt_A = m(self.idt_A)
            # Align idt_A spatial size with real_B
            # if self.idt_A.shape[-2:] != self.real_B.shape[-2:]:
            #     self.idt_A = F.interpolate(self.idt_A, size=self.real_B.shape[-2:], mode='bilinear', align_corners=False)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(F.interpolate(self.real_A, size=self.real_B.shape[-2:], mode='bilinear', align_corners=False))
            # print(f"[DBG] idt_B {tuple(self.idt_B.shape)}")
            # print(f"[DBG] real_A {tuple(self.real_A.shape)}")
            # n = UpsamplingBilinear2d(scale_factor=scale)
            # self.idt_B = n(self.idt_B)
            # # Align idt_B spatial size with real_A
            # if self.idt_B.shape[-2:] != self.real_A.shape[-2:]:
            #     self.idt_B = F.interpolate(self.idt_B, size=self.real_A.shape[-2:], mode='bilinear', align_corners=False)
            # Multiply by 0.1 to balance identity terms as in original implementation
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True) * self.opt.lambda_G_A
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True) * self.opt.lambda_G_B
        # Forward cycle loss || G_B(G_A(A)) - A||
        # Align rec_A with real_A before computing cycle loss
        if self.rec_A.shape[-2:] != self.real_A.shape[-2:]:
            self.rec_A = F.interpolate(self.rec_A, size=self.real_A.shape[-2:], mode='bilinear', align_corners=False)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        if self.rec_B.shape[-2:] != self.real_B.shape[-2:]:
            self.rec_B = F.interpolate(self.rec_B, size=self.real_B.shape[-2:], mode='bilinear', align_corners=False)
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # Downsample/upsample and SSIM losses are disabled in this variant
        self.loss_downsample = 0
        self.loss_upsample = 0
        self.loss_clinical_ssim = 0
        self.loss_micro_ssim = 0
        # Combined generator loss (exclude disabled losses)
        self.loss_G = (self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B +
                       self.loss_idt_A + self.loss_idt_B)
        self.loss_G.backward()

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration."""
        # forward
        self.forward()      # compute fake images and reconstruction images
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimising Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G(epoch)             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate gradients for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights