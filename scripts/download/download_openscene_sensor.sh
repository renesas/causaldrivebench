#!/bin/bash
# ============================================================================
# Download OpenScene sensor blobs (camera images) for selected logs
#
# This script downloads camera images from the OpenScene v1.1 HuggingFace
# dataset. Each log's images are packed into numbered "shards" (~200 total).
# This script downloads only the shards you need and extracts the cameras
# you specify.
#
# USAGE:
#   1. Edit the CONFIG section below to set your paths and select logs
#   2. Run:  bash download_openscene_sensor.sh
#
# To find shard numbers for new logs, download the metadata first
# (download_openscene_metadata.sh) and inspect the pickle files.
# ============================================================================

set -e

# ========================== CONFIG — EDIT THIS ==============================

# Where to store extracted sensor blobs
DATA_DIR="./data/sensor_blobs/trainval"

# Temp directory for downloading & extracting (cleaned up automatically)
TEMP_DIR="./data/_tmp"

# Which cameras to download (CAM_F0 = front, CAM_L0 = front-left,
# CAM_R0 = front-right, CAM_B0 = back)
CAMERAS="CAM_F0 CAM_L0 CAM_R0 CAM_B0"

# Shard-to-log mapping: each OpenScene camera shard (.tgz) contains images
# for one or more logs. To download a log, you need its shard number.
#
# Format:  SHARD_LOGS[<shard_number>]="<log_name>"
#
# Add or remove entries to control which logs are downloaded.
# To find the shard number for a log, check the OpenScene manifest or
# search the metadata pickles.
declare -A SHARD_LOGS

# --- Default: 10 logs used in our experiments ---
SHARD_LOGS[5]="2021.06.09.11.54.15_veh-12_05108_05331"
SHARD_LOGS[6]="2021.06.09.12.39.51_veh-26_03409_03722"
SHARD_LOGS[48]="2021.06.09.11.54.15_veh-12_04821_05096"
SHARD_LOGS[54]="2021.05.12.23.36.44_veh-35_01133_01535"
SHARD_LOGS[59]="2021.06.09.12.39.51_veh-26_00055_00360"
SHARD_LOGS[81]="2021.05.12.23.36.44_veh-35_02035_02387"
SHARD_LOGS[87]="2021.06.09.12.39.51_veh-26_02989_03385"
SHARD_LOGS[99]="2021.05.12.23.36.44_veh-35_01735_01957"
SHARD_LOGS[103]="2021.05.12.22.00.38_veh-35_00005_00118"
SHARD_LOGS[106]="2021.05.12.19.36.12_veh-35_00215_00405"

# ======================== END CONFIG ========================================

BASE_URL="https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_trainval_camera"

mkdir -p "$DATA_DIR" "$TEMP_DIR"

TOTAL=${#SHARD_LOGS[@]}
CURRENT=0

echo "Downloading $TOTAL shards for $(echo $CAMERAS | wc -w) cameras each..."
echo "Output: $DATA_DIR"
echo ""

for shard_num in $(echo "${!SHARD_LOGS[@]}" | tr ' ' '\n' | sort -n); do
    log="${SHARD_LOGS[$shard_num]}"
    CURRENT=$((CURRENT + 1))

    # Check if all cameras already exist for this log
    all_exist=true
    for cam in $CAMERAS; do
        if [ ! -d "$DATA_DIR/${log}/${cam}" ]; then
            all_exist=false
            break
        fi
    done
    if $all_exist; then
        echo "[$CURRENT/$TOTAL] Shard $shard_num ($log) — already done, skipping."
        continue
    fi

    TGZ_FILE="$TEMP_DIR/camera_${shard_num}.tgz"

    echo "[$CURRENT/$TOTAL] Shard $shard_num — downloading..."
    wget -q --show-progress -O "$TGZ_FILE" \
        "${BASE_URL}/openscene_sensor_trainval_camera_${shard_num}.tgz" || {
        echo "  ERROR: Download failed for shard $shard_num, skipping."
        rm -f "$TGZ_FILE"
        continue
    }

    # Build extraction patterns for only the cameras we need
    TAR_PATTERNS=""
    for cam in $CAMERAS; do
        TAR_PATTERNS="$TAR_PATTERNS openscene-v1.1/sensor_blobs/trainval/${log}/${cam}/"
    done

    echo "  Extracting $(echo $CAMERAS)..."
    tar -xzf "$TGZ_FILE" -C "$TEMP_DIR" $TAR_PATTERNS 2>/dev/null || {
        # Fallback: extract one camera at a time
        for cam in $CAMERAS; do
            tar -xzf "$TGZ_FILE" -C "$TEMP_DIR" \
                "openscene-v1.1/sensor_blobs/trainval/${log}/${cam}/" 2>/dev/null || true
        done
    }

    # Move to final destination
    for cam in $CAMERAS; do
        src="$TEMP_DIR/openscene-v1.1/sensor_blobs/trainval/${log}/${cam}"
        if [ -d "$src" ]; then
            mkdir -p "$DATA_DIR/${log}"
            mv "$src" "$DATA_DIR/${log}/${cam}"
            n=$(ls "$DATA_DIR/${log}/${cam}" | wc -l)
            echo "  ${log}/${cam}: ${n} images"
        else
            echo "  WARNING: ${cam} not found in shard $shard_num for $log"
        fi
    done

    rm -rf "$TEMP_DIR/openscene-v1.1"
    rm -f "$TGZ_FILE"
done

echo ""
echo "=== Done ==="
echo "Verifying camera coverage:"
for shard_num in $(echo "${!SHARD_LOGS[@]}" | tr ' ' '\n' | sort -n); do
    log="${SHARD_LOGS[$shard_num]}"
    cams=$(ls -d "$DATA_DIR/${log}"/CAM_* 2>/dev/null | xargs -I{} basename {} | sort | tr '\n' ' ')
    echo "  $log: $cams"
done
