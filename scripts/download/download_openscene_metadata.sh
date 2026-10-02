#!/bin/bash
# Download OpenScene v1.1 metadata (pickle logs with annotations, ego poses, etc.)
# This is required before running build_openscene_records.py.
# ~2 GB download.
#
# Usage: bash download_openscene_metadata.sh [output_dir]

set -e

OUT_DIR="${1:-./data/navsim_logs}"

echo "Downloading OpenScene metadata to $OUT_DIR ..."

wget -q --show-progress \
    https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_trainval.tgz

tar -xzf openscene_metadata_trainval.tgz
rm openscene_metadata_trainval.tgz

mkdir -p "$OUT_DIR"
mv openscene-v1.1/meta_datas/* "$OUT_DIR/"
rm -rf openscene-v1.1

echo "Done. Metadata saved to $OUT_DIR"
