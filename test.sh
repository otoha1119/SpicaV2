"""
chmod +x test.sh 
./test.sh
"""

pip install pytorch-ssim

python medicaltest_dicom.py \
  --dataset_mode dicom_ctpcct_2x_test \
  --model medical_cycle_gan \
  --clinical2micronetG clinical_to_micro_resnet_9blocks \
  --micro2clinicalnetG micro_to_clinical_resnet_9blocks \
  --netG resnet_9blocks \
  --ngf 64 \
  --input_nc 1 --output_nc 1 \
  --name SR_CycleGAN \
  --checkpoints_dir /workspace/checkpoints_mac \
  --epoch 167 \
  --use_G A \
  --input_dicom /workspace/IM_091.dcm \
  --output_dicom /workspace/results/SR_2x.dcm \
  --halves_pixel_spacing \
  --pad_mod 4 \
  --gpu_ids -1 \
  --sampling_times 1
