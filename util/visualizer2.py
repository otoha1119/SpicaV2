# visualizer2.py
import os
import time
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchvision.utils as vutils

class Visualizer2:
    """Visualizer for TensorBoard (losses and images)"""

    def __init__(self, opt):
        self.opt = opt
        self.name = opt.name
        self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, "logs_tb")
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        self.global_step = 0

    def reset(self):
        """Reset any per-iteration states if needed"""
        pass

    def display_current_results(self, visuals, epoch, save_result):
        """Display images in TensorBoard"""
        for label, image_tensor in visuals.items():
            if image_tensor is None:
                continue
            # Normalize for visualization
            grid = vutils.make_grid(image_tensor.detach().cpu(), normalize=True, scale_each=True)
            self.writer.add_image(label, grid, self.global_step)

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """Plot training losses to TensorBoard"""
        for name, value in losses.items():
            self.writer.add_scalar(f"loss/{name}", value, self.global_step)

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """Still print to console for quick check"""
        message = f"(epoch: {epoch}, iters: {iters}, time: {t_comp:.3f}, data: {t_data:.3f}) "
        for k, v in losses.items():
            message += f"{k}: {v:.3f} "
        print(message)

    def close(self):
        self.writer.close()
