import os
import torch
import numpy as np
import pydicom

from options.test_options import TestOptions
from data import create_dataset
from models import create_model

from util.dicom_io import read_normalized_pixels, denormalize_to_int16


def save_dicom_like(reference_path: str, output_path: str, norm_img: np.ndarray):
    """学習と同じ方式で逆正規化し、メタだけコピーして保存"""
    ref = pydicom.dcmread(reference_path)
    out = ref.copy()

    # --- 学習と完全一致する逆正規化 ---
    img_int16 = denormalize_to_int16(norm_img.astype(np.float32))

    H, W = img_int16.shape
    out.Rows = int(H)
    out.Columns = int(W)
    out.PhotometricInterpretation = "MONOCHROME2"
    out.SamplesPerPixel = 1

    # DICOM 必須タグ（最小）
    out.BitsAllocated = 16
    out.BitsStored = 16
    out.HighBit = 15
    out.PixelRepresentation = 1

    out.PixelData = img_int16.tobytes()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.save_as(output_path, write_like_original=False)

    print(f"[OK] Saved: {output_path}")


def main():
    opt = TestOptions().parse()

    # 推論時は1枚だけ
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True

    # 強制的に test dataset を使う
    opt.dataset_mode = "dicom_ctpcct_2x_test"

    # モデル読み込み
    dataset = create_dataset(opt)
    model = create_model(opt)

    # setup（学習と同じ初期化）
    model.setup(opt)
    model.eval()

    # データ1枚取得
    data = next(iter(dataset))

    # モデル入力
    with torch.no_grad():
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()

        out = visuals["fake_B"]  # A→B のみ使用
        out = out[0, 0].cpu().numpy()  # (1,1,H,W) → (H,W)

        out = np.clip(out, 0.0, 1.0)

    # 保存
    ref_path = data["A_paths"]
    if isinstance(ref_path, (list, tuple)):
        ref_path = ref_path[0]   
    
    save_dicom_like(
        reference_path=ref_path,
        output_path=opt.output_dicom,
        norm_img=out
    )


if __name__ == "__main__":
    main()
