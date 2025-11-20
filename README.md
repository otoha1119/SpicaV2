# SpicaV2

## 作業タグ
add：新規機能追加<br>
modify：バグではない修正<br>
fix：既存バグ修正<br>
hotfix：クリティカルなバグ修正<br>
refactor：整理<br>
delete：削除<br>
move:ファイル・フォルダの移動<br>
revert：変更取り消し<br>
other：その他<br>


## 概要
本プロジェクトはSRーCycleGANを改良した超解像モデルである<br>
dockerを用いて環境を構築している，50シリーズGPUと30シリーズGPUの両方ようのdockerフォルダがある<br>

## 実行ファイル
・学習<br>
train.shにて主要パラメータを指定し，shファイルを実行<br>

・推論<br>
評価のためのDICOM一括推論と1枚だけの推論ができる二種類のスクリプトを用意している<br>
それぞれinference_multi.shとinference.single.shで実行する<br>

・評価<br>
MTFを作成する場合はmtf.shを実行<br>
MTFのROI作成数の信ぴょう性を確かめるスクリプトがmtf_convergence.pyを直接実行である<br>

・その他<br>
gitの過去のコミットに強制的に戻るコマンドを自動化したものがreset-branch.shである<br>

・Tensorboard実行コード<br>
tensorboard --logdir checkpoints/SR_CycleGAN/runs --port 6006 --host 0.0.0.0<br>
http://localhost:6006<br>

