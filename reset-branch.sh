#!/usr/bin/env bash
# reset-branch.sh
# 安全に「ブランチを特定コミットへ戻す（リモートも上書き）」ためのスクリプト
# Usage:
#   ./reset-branch.sh -c <commit> -b <branch> [-r <remote>] [--no-backup] [--no-config] [--dry-run]
# Examples:
#   ./reset-branch.sh -c da5ff249 -b Improvements1
#   ./reset-branch.sh -c fda828a -b main -r origin


"""
chmod +x reset-branch.sh
./reset-branch.sh -c da5ff249 -b Improvements1

他端末
git fetch --prune
git checkout Improvements1 || git switch -C Improvements1
git reset --hard origin/Improvements1

"""

set -euo pipefail

# ===== デフォルト値（必要なら書き換え可） =====
COMMIT_DEFAULT="fda828a"
BRANCH_DEFAULT="Improvements1"
REMOTE_DEFAULT="origin"
DO_BACKUP=1         # 0で無効化
DO_CONFIG=1         # 0で無効化（pull.ff only / upstream設定）
DRY_RUN=0

# ===== 引数処理 =====
COMMIT="$COMMIT_DEFAULT"
BRANCH="$BRANCH_DEFAULT"
REMOTE="$REMOTE_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--commit)   COMMIT="$2"; shift 2;;
    -b|--branch)   BRANCH="$2"; shift 2;;
    -r|--remote)   REMOTE="$2"; shift 2;;
    --no-backup)   DO_BACKUP=0; shift;;
    --no-config)   DO_CONFIG=0; shift;;
    --dry-run)     DRY_RUN=1; shift;;
    -h|--help)
      cat <<EOF
Usage: $0 -c <commit> -b <branch> [-r <remote>] [--no-backup] [--no-config] [--dry-run]

Options:
  -c, --commit    戻したいコミット（SHA, 省略時: ${COMMIT_DEFAULT}）
  -b, --branch    対象ブランチ（省略時: ${BRANCH_DEFAULT}）
  -r, --remote    リモート名（省略時: ${REMOTE_DEFAULT}）
  --no-backup     退避ブランチ/タグを作らない
  --no-config     pull.ff only / upstream 設定をしない
  --dry-run       変更内容を表示のみ（実行しない）
EOF
      exit 0;;
    *)
      echo "[ERROR] Unknown arg: $1" >&2; exit 1;;
  esac
done

log() { echo -e "[INFO] $*"; }
warn() { echo -e "[WARN] $*" >&2; }
err() { echo -e "[ERROR] $*" >&2; }

# ===== 事前チェック =====
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  err "ここはGitリポジトリではありません。"; exit 1
fi

if ! git rev-parse --verify --quiet "${COMMIT}^{commit}" >/dev/null; then
  err "指定コミットが見つかりません: ${COMMIT}"
  exit 1
fi

if ! git remote | grep -qx "${REMOTE}"; then
  err "指定リモートが存在しません: ${REMOTE}"
  exit 1
fi

TS=$(date +%Y%m%d-%H%M%S)
BACKUP_BRANCH="backup/${BRANCH}-${TS}"
BACKUP_TAG="pre-reset-${BRANCH}-${TS}"

run() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "DRY: $*"
  else
    eval "$@"
  fi
}

# ===== 設定（pull.ff only / upstream） =====
if [[ "${DO_CONFIG}" -eq 1 ]]; then
  log "pull.ff only（FF以外のpullを禁止）をこのリポジトリに設定します。"
  run "git config pull.ff only" || warn "pull.ff only の設定に失敗しました。"

  # 後でcheckoutするので、その後に upstream 設定を行う
fi

# ===== 退避（ブランチ & タグ） =====
if [[ "${DO_BACKUP}" -eq 1 ]]; then
  log "退避ブランチを作成します: ${BACKUP_BRANCH}"
  # 今のHEADを退避
  run "git show -s --oneline HEAD || true"
  run "git branch \"${BACKUP_BRANCH}\" || true"
  log "退避タグを作成します: ${BACKUP_TAG}"
  run "git tag -a \"${BACKUP_TAG}\" -m \"before reset ${BRANCH} at ${TS}\" || true"

  log "退避（ブランチ/タグ）をリモートへプッシュします。"
  run "git push \"${REMOTE}\" \"${BACKUP_BRANCH}\" || true"
  run "git push \"${REMOTE}\" --tags || true"
else
  warn "退避をスキップします（--no-backup）。"
fi

# ===== フェッチ & チェックアウト =====
log "リモート最新を取得（--prune）します。"
run "git fetch \"${REMOTE}\" --prune"

log "対象ブランチに切り替えます: ${BRANCH}"
# -B: ブランチを作成/上書き（存在すれば上書き）→ 常にそのブランチで作業
run "git checkout -B \"${BRANCH}\" || git switch -C \"${BRANCH}\""

# upstream 設定（pull時のエラー対策）
if [[ "${DO_CONFIG}" -eq 1 ]]; then
  log "upstream を設定します: ${BRANCH} -> ${REMOTE}/${BRANCH}"
  run "git branch --set-upstream-to=\"${REMOTE}/${BRANCH}\" \"${BRANCH}\" || true"
fi

# ===== 実本命：reset & push =====
log "ブランチをコミットへ合わせます: ${BRANCH} -> ${COMMIT}"
run "git reset --hard \"${COMMIT}\""

log "リモートを上書きします（--force-with-lease 推奨）。"
run "git push \"${REMOTE}\" \"${COMMIT}:${BRANCH}\" --force-with-lease"

# ===== 同期のヒント =====
echo
log "✅ 完了: ${BRANCH} は ${COMMIT} に揃いました（ローカル & リモート）。"
echo
echo "他端末の同期コマンド（コピペ用）:"
echo "  git fetch --prune"
echo "  git checkout ${BRANCH} || git switch -C ${BRANCH}"
echo "  git branch --set-upstream-to=${REMOTE}/${BRANCH} ${BRANCH} || true"
echo "  git reset --hard ${REMOTE}/${BRANCH}"
echo
