# train_options.py  (抜粋／変更点のみ)

from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        
        parser.add_argument('--sampling_times', type=int, default=1,
                    help='2^n times of upsampling; default=1 → 2x')

        parser.add_argument('--pool_size', type=int, default=50,
                    help='the size of image buffer that stores previously generated images')

        parser = BaseOptions.initialize(self, parser)
        
        # 可視化(Visdom/HTML)まわり
        parser.add_argument('--display_id', type=int, default=0)
        parser.add_argument('--display_port', type=int, default=8097)
        parser.add_argument('--display_server', type=str, default='http://localhost')
        parser.add_argument('--display_env', type=str, default='main')
        parser.add_argument('--display_ncols', type=int, default=4)
        parser.add_argument('--update_html_freq', type=int, default=1000)
        parser.add_argument('--no_html', action='store_true')

        # ---- 可視化/保存/学習系 基本項目 ----
        parser.add_argument('--display_freq', type=int, default=400)
        parser.add_argument('--print_freq', type=int, default=100)
        parser.add_argument('--save_latest_freq', type=int, default=5000)
        parser.add_argument('--save_epoch_freq', type=int, default=1)
        parser.add_argument('--save_by_iter', action='store_true')
        parser.add_argument('--continue_train', action='store_true')
        parser.add_argument('--epoch_count', type=int, default=1)
        parser.add_argument('--niter', type=int, default=100)
        parser.add_argument('--niter_decay', type=int, default=100)
        parser.add_argument('--beta1', type=float, default=0.5)
        parser.add_argument('--lr', type=float, default=1e-5)
        parser.add_argument('--gan_mode', type=str, default='lsgan')
        parser.add_argument('--lr_policy', type=str, default='linear')
        parser.add_argument('--lr_decay_iters', type=int, default=50)

        # ====== ここから dataset 固有設定を「中央集約」 ======
        # ルート（元々 dataset.modify_commandline_options にあった）
        parser.add_argument('--lr_root', type=str, default='/workspace/DataSet/ImageCAS',
                            help='Root directory for LR DICOM tree')
        parser.add_argument('--hr_root', type=str, default='/workspace/DataSet/photonCT/PhotonCT1024v2',
                            help='Root directory for HR DICOM tree')

        # クロップ設定：単一ソース化（推奨：lr_patch + scale から hr_patch を導出）
        parser.add_argument('--lr_patch', type=int, default=98, help='LR patch (pixels)')
        parser.add_argument('--scale', type=int, default=2, choices=[2,4,8], help='SR scale factor')
        parser.add_argument('--hr_patch', type=int, default=0,
                            help='(optional) If 0, set to lr_patch*scale; if >0, must equal lr_patch*scale')

        # サンプリング・エポック制御
        parser.add_argument('--hr_oversample_ratio', type=float, default=1.0,
                            help='>1.0 increases probability of sampling HR slices (unpaired)')
        parser.add_argument('--epoch_size', type=int, default=0,
                            help='If >0, overrides dataset length per epoch')
        parser.add_argument('--fast_scan', action='store_true',
                            help='Skip header reads; sort by filename only')

        # ボディマスク（OFF が既定）
        parser.add_argument('--use_body_mask', action='store_true')
        parser.add_argument('--body_thresh_norm', type=float, default=0.1)
        parser.add_argument('--min_body_coverage', type=float, default=0.3)

        # ---- 以下、元の大量の専用パス等は削除/非推奨化 ----
        # * all_*_paths / encoder_path / code_channel など研究固有のものは一旦撤去。
        # * 必要になったら個別の実験で CLI から渡すか、別の config ファイルで管理。
        self.isTrain = True
        return parser


# 例: BaseOptions.parse() の末尾 self.opt を返す直前など
# (train/test 両方で整合したいならここに置くのが楽)


