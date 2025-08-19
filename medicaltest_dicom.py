from __future__ import annotations
import os, torch, numpy as np
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import dicom_io as dio

def pick_fake_B(vis_dict):
    for k in ['fake_B','fake_micro','fake_B_full','fake_B_sr','SR']:
        if k in vis_dict: return vis_dict[k]
    best = None; best_hw = -1
    for v in vis_dict.values():
        if torch.is_tensor(v) and v.dim()>=2:
            h,w = v.shape[-2], v.shape[-1]
            if h*w > best_hw: best_hw=h*w; best=v
    if best is None: raise RuntimeError("Cannot find generated output in visuals.")
    return best

def main():
    opt = TestOptions().parse()
    if opt.dataset_mode != 'dicom_ctpcct_2x_test':
        print("[Info] override dataset_mode -> dicom_ctpcct_2x_test")
        opt.dataset_mode = 'dicom_ctpcct_2x_test'
    if not hasattr(opt,'output_dicom'):
        setattr(opt,'output_dicom', os.environ.get('OUTPUT_DICOM','./results/SR_2x.dcm'))
    if not hasattr(opt,'halves_pixel_spacing'):
        setattr(opt,'halves_pixel_spacing', True)

    dataset = create_dataset(opt)
    model = create_model(opt); model.setup(opt)
    if opt.eval: model.eval()

    data = next(iter(dataset))
    model.set_input(data)
    with torch.no_grad():
        model.test()
    visuals = model.get_current_visuals()
    fake = pick_fake_B(visuals)
    if fake.dim()==4: fake = fake[0,0]
    elif fake.dim()==3: fake = fake[0]
    out_norm = torch.clamp(fake, 0, 1).cpu().numpy().astype(np.float32)

    in_path = data.get('A_paths', data.get('clinical_path'))
    if isinstance(in_path,(list,tuple)): in_path = in_path[0]
    out_path = getattr(opt,'output_dicom','./results/SR_2x.dcm')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dio.save_dicom_like(in_path, out_path, out_norm, halves_pixel_spacing=bool(getattr(opt,'halves_pixel_spacing',True)))
    print(f"[OK] Saved: {out_path}")

if __name__ == "__main__":
    main()
