#!/bin/bash
# reset-branch.sh
# Usage: ./reset-branch.sh <commit> <branch>
"""
chmod +x reset-branch.sh 
./reset-branch.sh <commit> <branch>
"""

# --- 設定項目（毎回変えるなら引数でもOK）---
TARGET_COMMIT=${1}     # 戻したいコミット
TARGET_BRANCH=${2} # 操作対象ブランチ
REMOTE_NAME=origin              # リモート名（通常 origin）

# --- 実行処理 ---
set -e

echo "[INFO] Checkout to ${TARGET_BRANCH}"
git checkout ${TARGET_BRANCH}

echo "[INFO] Reset branch to ${TARGET_COMMIT}"
git reset --hard ${TARGET_COMMIT}

echo "[INFO] Force push to ${REMOTE_NAME}/${TARGET_BRANCH}"
git push ${REMOTE_NAME} ${TARGET_COMMIT}:${TARGET_BRANCH} --force-with-lease

echo "[DONE] ${TARGET_BRANCH} is now at ${TARGET_COMMIT} (both local & remote)"
