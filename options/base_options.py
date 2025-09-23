# -*- coding: utf8 -*-
import argparse
import os
from util import util
import torch
import models
import data


class BaseOptions():
    """Options used during both training and test time."""

    def __init__(self):
        self.initialized = False
        # isTrain は TestOptions/TrainOptions 側でセットされる想定
        # ここでは存在する前提で使います

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot', type=str, default='/', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. "0", "0,1,2". use "-1" for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

        # model parameters
        parser.add_argument('--model', type=str, default='medical_cycle_gan', help='chooses which model to use. [cycle_gan | pix2pix | test | medical_cycle_gan | colorization]')
        parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]')
        parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--norm', type=str, default='instance', help='instance | batch | none')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal/xavier/orthogonal')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')

        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='medical_2D', help='[unaligned | aligned | single | medical_2D | dicom_ctpcct_2x | dicom_ctpcct_2x_test]')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order, otherwise randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--max_dataset_size', type=int, default=int(40000), help='Maximum number of samples allowed per dataset')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='[resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true', help='do not flip images for data augmentation')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size')

        # additional parameters
        parser.add_argument('--epoch', type=str, default='200', help='which epoch to load? set to "latest" to use latest cached model')
        parser.add_argument('--load_iter', type=int, default=0, help='if >0, load models by iter_[load_iter]; else by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix, e.g., {model}_{netG}_size{load_size}')

        # medical parameters
        parser.add_argument('--clinical2micronetG', type=str, default='clinical_to_micro_resnet_9blocks', help='network transform clinicalCT to microCT')
        parser.add_argument('--micro2clinicalnetG', type=str, default='micro_to_clinical_resnet_9blocks', help='network transform microCT to clinicalCT')
        parser.add_argument('--medical_input_nc', type=int, default=1, help='# of input image channels (medical)')
        parser.add_argument('--medical_output_nc', type=int, default=1, help='# of output image channels (medical)')
        parser.add_argument('--limit_per_patient', type=int, default=0, help='limit #scans per patient (0=unlimited)')
        # ↑ ここまで initialize

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser, then add model- and dataset-specific options."""
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # model-specific options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()

        # dataset-specific options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        for k, v in sorted(vars(opt).items()):
            default = self.parser.get_default(k)
            if v != default:
                message += f'{k}: {v} [default: {default}]\n'
            else:
                message += f'{k}: {v}\n'

        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up device safely."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # ---- safe GPU/CPU selection ----
        # normalize gpu_ids (string like "0,1" or "-1")
        str_ids = str(opt.gpu_ids).split(',')
        gpu_ids = []
        for sid in str_ids:
            sid = sid.strip()
            if sid == '':
                continue
            try:
                idx = int(sid)
                if idx >= 0:
                    gpu_ids.append(idx)
            except ValueError:
                pass
        opt.gpu_ids = gpu_ids

        if torch.cuda.is_available() and len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])
            opt.device = "cuda"
        else:
            opt.gpu_ids = []
            opt.device = "cpu"

        self.opt = opt
        return self.opt
