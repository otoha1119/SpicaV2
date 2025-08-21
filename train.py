# -*- coding: utf8 -*-
"""General-purpose training script for SR-CycleGAN with tqdm progress bar"""

import os
import time
import io
from contextlib import redirect_stdout
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
#from util.visualizer import Visualizer
from util.visualizer2 import Visualizer2
from tqdm import tqdm

Visualizer = Visualizer2

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)

    model = create_model(opt)
    model.setup(opt)

    visualizer = Visualizer(opt)
    total_iters = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        steps_per_epoch = (dataset_size + opt.batch_size - 1) // opt.batch_size
        with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch}", unit="batch", leave=False) as pbar:
            for i, data in enumerate(dataset, start=1):
                iter_start_time = time.time()
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                visualizer.reset()
                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
                model.set_input(data)
                model.optimize_parameters(epoch)

                if total_iters % opt.display_freq == 0:
                    save_result = total_iters % opt.update_html_freq == 0
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                if total_iters % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size

                    _sink = io.StringIO()
                    with redirect_stdout(_sink):
                        visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                    nice_losses = {k: f"{v:.3f}" for k, v in losses.items() if isinstance(v, (int, float))}
                    keys = list(nice_losses.keys())[:6]
                    pbar.set_postfix({k: nice_losses[k] for k in keys})

                if total_iters % opt.save_latest_freq == 0:
                    tqdm.write(f"saving the latest model (epoch {epoch}, total_iters {total_iters})")
                    save_suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
                    model.save_networks(save_suffix)

                iter_data_time = time.time()
                pbar.update(1)

        if epoch % opt.save_epoch_freq == 0:
            tqdm.write(f"saving the model at the end of epoch {epoch}, iters {total_iters}")
            model.save_networks('latest')
            model.save_networks(epoch)

        tqdm.write(f"End of epoch {epoch} / {opt.niter + opt.niter_decay} \t Time Taken: {time.time() - epoch_start_time} sec")
        model.update_learning_rate()
