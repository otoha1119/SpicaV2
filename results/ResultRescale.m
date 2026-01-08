clc; clear;

%% ============================================================
% 0. ユーザー設定（ここだけ書き換えればOK）
% =============================================================
dicomPath = "/Users/otoha/Library/CloudStorage/OneDrive-学校法人立命館/Results/06IM_091_SR2x.dcm";

% PNG 出力フォルダ（存在しなければ自動生成）
outputImageDir = "/Users/otoha/Library/CloudStorage/OneDrive-学校法人立命館/Results";
outputHistDir  = outputImageDir;

%保存の有無 1で保存0で保存なし
save_flug = 1;

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
% 4. PNG 用に 0〜255 正規化
% =============================================================
img_norm = uint8(255 * (img_clipped - minHU) / (maxHU - minHU));

%% ============================================================
% 5. PNG 画像表示 ＋ 保存
% =============================================================
[~, baseName, ~] = fileparts(dicomPath);
pngImageName = fullfile(outputImageDir, "converted_"+ baseName + ".png");

% figure をハンドル付きで作成して、ウィンドウ名を変更
fig = figure;
set(fig, 'Name', baseName, 'NumberTitle', 'off');

imshow(img_norm, []);
title("Converted PNG Image (Clipped -2048〜2048)");

if(save_flug ==1)
    imwrite(img_norm, pngImageName);
    fprintf("画像 PNG を保存しました: %s\n", pngImageName);
end



%% ============================================================
% 6. ヒストグラム表示 ＋ JPEG 保存（縦軸：パーセンテージ）
% =============================================================
img_for_hist = img(img >= minHU & img <= maxHU);

fig = figure( ...
    'Units','pixels', ...
    'Position',[100 100 812 595], ...
    'Color','w');

histogram(img_for_hist, 512, ...
    'FaceColor',[0.6 0.6 0.6], ...
    'EdgeColor','none', ...
    'Normalization','probability');

xlim([minHU maxHU]);
ylim([0 0.02]);

xlabel('HU value');
ylabel('Percentage (%)');
title(sprintf('HU Distribution (%s)', baseName), 'Interpreter','none');

ax = gca;
ax.Units = 'normalized';
ax.Position = [0.12 0.10 0.83 0.80];

ax.YTickLabel = compose('%.1f', ax.YTick * 100);

if save_flug == 1
    pngHistName = fullfile(outputHistDir, "hist_" + baseName + ".png");
    exportgraphics(fig, pngHistName, ...
        'Resolution', 72);   % ← 72 dpi 固定
end



fprintf("\n=== 全処理完了しました ===\n");



    