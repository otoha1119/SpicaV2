clc; clear;

%% ============================================================
% 0. ユーザー設定（ここだけ書き換えればOK）
% =============================================================
dicomPath = "/Users/otoha/Documents/Resurch/Results/07IM_091_SR2x.dcm";

% JPEG 出力フォルダ（存在しなければ自動生成）
outputImageDir = "/Users/otoha/Documents/Resurch/Results";
outputHistDir  = outputImageDir;

%保存の有無
save_flug = 0;

%% ============================================================
% 出力フォルダ作成
% =============================================================
if ~exist(outputImageDir, 'dir')
    mkdir(outputImageDir);
end
if ~exist(outputHistDir, 'dir')
    mkdir(outputHistDir);
end

%% ============================================================
% 1. 入力ファイルチェック
% =============================================================
if ~exist(dicomPath, 'file')
    error('指定した DICOM ファイルが存在しません: %s', dicomPath);
end

%% ============================================================
% 2. DICOM 読み込み + HU変換
% =============================================================
info = dicominfo(dicomPath);
img  = double(dicomread(info));

% HU 変換（RescaleSlope・Intercept があれば適用）
if isfield(info, 'RescaleSlope') && isfield(info, 'RescaleIntercept')
    img = img * info.RescaleSlope + info.RescaleIntercept;
end

%% ============================================================
% 3. HU を [-2048, +2048] にクリップ
% =============================================================
minHU = -2048;
maxHU = 2048;
img_clipped = max(min(img, maxHU), minHU);

%% ============================================================
% 4. JPEG 用に 0〜255 正規化
% =============================================================
img_norm = uint8(255 * (img_clipped - minHU) / (maxHU - minHU));

%% ============================================================
% 5. JPEG 画像表示 ＋ 保存
% =============================================================
[~, baseName, ~] = fileparts(dicomPath);
jpegImageName = fullfile(outputImageDir, baseName + "_converted.jpg");

% figure をハンドル付きで作成して、ウィンドウ名を変更
fig = figure;
set(fig, 'Name', baseName, 'NumberTitle', 'off');

imshow(img_norm, []);
title("Converted JPEG Image (Clipped -2048〜2048)");

if(save_flug ==1)
    imwrite(img_norm, jpegImageName);
    fprintf("画像 JPEG を保存しました: %s\n", jpegImageName);
end



%% ============================================================
% 6. ヒストグラム表示 ＋ JPEG 保存
% =============================================================
img_for_hist = img(img >= minHU & img <= maxHU);

fig = figure;
histogram(img_for_hist, 512, ...
    'FaceColor', [0.4 0.4 0.4], ...
    'EdgeColor', 'none');

title(sprintf('HU Distribution (%s)', baseName), 'Interpreter','none');
xlabel('HU value');
ylabel('Frequency');
xlim([minHU, maxHU]);

% 保存
jpegHistName = fullfile(outputHistDir, baseName + "_hist.jpg");
saveas(fig, jpegHistName);

fprintf("ヒストグラム JPEG を保存しました: %s\n", jpegHistName);

fprintf("\n=== 全処理完了しました！ ===\n");


    