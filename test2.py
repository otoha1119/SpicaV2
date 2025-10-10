# -*- coding: utf-8 -*-
"""
Convenience runner for batch inference.
- Sets the typical args you were using in test.sh
- Processes every DICOM under /workspace/DataSet/ImageCAS/001.ImgCast
- Saves outputs under /workspace/results/<new_folder>/..._SR2x.dcm
"""

import os
import datetime
import subprocess
import sys

IN_DIR = "/workspace/DataSet/ImageCAS/001.ImgCast"
# Create a fresh results subfolder like: results/001.ImgCast_SR2x_YYYYmmdd-HHMMSS
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
OUT_DIR = f"/workspace/results/001.ImgCast_SR2x_{stamp}"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cmd = [
        sys.executable, "inference_multi.py",
        "--dataset_mode", "dicom_ctpcct_2x_test",
        "--model", "medical_cycle_gan",
        "--clinical2micronetG", "clinical_to_micro_resnet_9blocks",
        "--micro2clinicalnetG", "micro_to_clinical_resnet_9blocks",
        "--netG", "resnet_9blocks",
        "--ngf", "64",
        "--input_nc", "1", "--output_nc", "1",
        "--name", "SR_CycleGAN",
        "--checkpoints_dir", "/workspace/checkpoints_mac",
        "--epoch", "167",
        "--use_G", "A",
        "--halves_pixel_spacing",
        "--pad_mod", "4",
        "--gpu_ids", "0",
        "--sampling_times", "1",
        # batch-specific
        "--input_dir", IN_DIR,
        "--output_dir", OUT_DIR,
    ]

    print("[INFO] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[DONE] Results saved under:", OUT_DIR)

if __name__ == "__main__":
    main()
