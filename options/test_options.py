from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def parse(self):
        return BaseOptions.parse(self)

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # 共有オプション

        # ---- テスト用の追加フラグ ----
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')

        # ---- ここが重要 ----
        # BaseOptions ですでに --phase 定義済みなので add しない
        # デフォルトだけ test に上書き
        parser.set_defaults(phase='test')

        # モデルを "test" にしない（医用モデルを使う）
        # parser.set_defaults(model='test')  # ← 消す！

        # 必要ならデータセットの既定をテスト用に（任意）
        # parser.set_defaults(dataset_mode='dicom_ctpcct_2x_test')

        # クロップ回避したい場合の慣例（load_size=crop_size）
        parser.set_defaults(load_size=parser.get_default('crop_size'))

        # ---- ここから下はあなたのカスタム項目（必要なら残す/整理）----
        parser.add_argument('--clinical_folder', type=str, default='', help='floder of clinicalCT')
        parser.add_argument('--micro_folder', type=str, default='', help='folder of microCT')
        parser.add_argument('--all_clinical_paths', type=str, default=[
            '/homes/tzheng/CTdata/CTMicroNUrespsurg/cct/nulung031.512.nii.gz',
            '/homes/tzheng/CTdata/CTMicroNUrespsurg/cct/nulung050.nii.gz',
            '/homes/tzheng/CTdata/CTMicroNUrespsurg/cct/nulung030.nii.gz'
        ], help='all medical paths')
        parser.add_argument('--all_micro_paths', type=str, default=[
            '/homes/tzheng/CTdata/CTMicroNUrespsurg/nii/nulung050/nulung050_053_000.nii.gz',
            '/homes/tzheng/CTdata/CTMicroNUrespsurg/converted/DICOM_nulung030_cb_004_zf_ringRem_med3.nii.gz'
        ], help='all clinical paths')
        parser.add_argument('--batch_num', type=int, default=2000, help='the batch num')
        parser.add_argument('--clinical_patch_size', type=int, default=32, help='patch size of clinicalCT')
        parser.add_argument('--micro_patch_size', type=int, default=256, help='patch size of microCT')
        parser.add_argument('--sampling_times', type=int, default=3, help='2^n times of upsampling from clinicalCT to microCT')
        parser.add_argument('--maskdatafolder', type=str, default='/homes/tzheng/CTdata/CTMicroNUrespsurg/Mask')

        self.isTrain = False
        return parser
