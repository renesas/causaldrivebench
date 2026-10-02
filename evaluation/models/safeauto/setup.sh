#!/bin/bash
# =====================================================================
# SafeAuto setup: download LanguageBind vision encoders required by the
# SafeAuto-DriveLM checkpoint.
#
# The LLaVA builder hardcodes cache_dir='./cache_dir', so the models
# must live at:
#   <this_script_dir>/SafeAuto/cache_dir/LanguageBind_Image
#   <this_script_dir>/SafeAuto/cache_dir/LanguageBind_Video_merge
#
# Both towers appear in the checkpoint config ("X": ["Video", "Image"])
# and are loaded on model init, so both are required.
#
# Run once before the first inference:
#   bash evaluation/models/safeauto/setup.sh
# =====================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAFEAUTO_REPO="$SCRIPT_DIR/SafeAuto"
CACHE_DIR="$SAFEAUTO_REPO/cache_dir"

if [ ! -d "$SAFEAUTO_REPO" ]; then
    echo "ERROR: SafeAuto repo not found at $SAFEAUTO_REPO"
    echo "Please clone https://github.com/AI-secure/SafeAuto there first."
    exit 1
fi

mkdir -p "$CACHE_DIR"

# git-lfs is required for LanguageBind large weight files
if ! command -v git-lfs &> /dev/null; then
    echo "Installing git-lfs..."
    sudo apt-get install -y git-lfs
fi
git lfs install

download_if_missing() {
    local url="$1"
    local dest="$2"
    if [ -d "$dest" ] && [ "$(ls -A "$dest")" ]; then
        echo "  Already present: $dest — skipping."
    else
        echo "  Cloning $url → $dest ..."
        GIT_LFS_SKIP_SMUDGE=1 git clone "$url" "$dest"
        cd "$dest"
        git lfs pull
        cd - > /dev/null
        echo "  Done: $dest"
    fi
}

echo "====================================================================="
echo "  Downloading LanguageBind encoders to $CACHE_DIR"
echo "====================================================================="

download_if_missing \
    "https://huggingface.co/LanguageBind/LanguageBind_Image" \
    "$CACHE_DIR/LanguageBind_Image"

download_if_missing \
    "https://huggingface.co/LanguageBind/LanguageBind_Video_merge" \
    "$CACHE_DIR/LanguageBind_Video_merge"

echo "====================================================================="
echo "  Setup complete."
echo "  LanguageBind_Image       : $CACHE_DIR/LanguageBind_Image"
echo "  LanguageBind_Video_merge : $CACHE_DIR/LanguageBind_Video_merge"
echo "====================================================================="
