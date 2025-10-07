#実行コード
# tensorboard --logdir checkpoints/SR_CycleGAN/runs --port 6006 --host 0.0.0.0
# chmod +x run_auto.sh 
# ./run_auto.sh

pip install pytorch-msssim
set -e

echo "=== 依存関係をインストールします ==="
# apt-get update
# apt-get install -y libgl1-mesa-glx

# pip install --upgrade pip
# pip install -r requirements.txt
# python -m pip install nibabel

echo "=== 依存関係インストール完了 ==="

echo "=== 学習を開始します ==="
python train.py --dataroot /workspace/DataSet/ImageCAS \
                --name SR_CycleGAN \
                --model medical_cycle_gan \
                --direction AtoB \
                --dataset_mode dicom_ctpcct_2x \
                --batch_size 1 \
                --epoch 200 \
                --niter 100 \
                --niter_decay 100 \
                --gpu_ids -1 \
                --hr_root /workspace/DataSet/photonCT/PhotonCT1024v2 \
                --lr_root /workspace/DataSet/ImageCAS \
                --num_threads 4 \
                --fast_scan \
                --limit_per_patient 0 \
                #--verbose \
                --sampling_times 1
