"""
build_image_cache.py — Build the persistent cross-model image cache.

Walks a CausalDriveBench dataset directory, encodes all JPEG/PNG images
to base64, and writes sharded pickle files that can be loaded in seconds
by any model's inference script.

Usage:
    python evaluation/build_image_cache.py \\
        --bench-dir /path/to/causal_drive_bench/causal_nuscenes \\
        --cache-dir /path/to/image_cache \\
        --image-size 448 \\
        --n-shards 8 \\
        --workers 64

    # Custom data-root (keys stored relative to this):
    python evaluation/build_image_cache.py \\
        --bench-dir /path/to/causal_drive_bench/causal_nuscenes \\
        --data-root /path/to/causal_drive_bench \\
        --cache-dir /path/to/image_cache \\
        --image-size 448

The cache can then be loaded inside any inference script:
    from evaluation.common.image_cache import PersistentImageCache
    cache = PersistentImageCache().load(
        cache_dir=Path("/path/to/image_cache"),
        dataset_name="causal_nuscenes",
        image_size=448,
        data_root=Path("/path/to/causal_drive_bench"),
    )
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow running as a standalone script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.common.image_cache import PersistentImageCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_image_cache")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def collect_image_paths(bench_dir: Path) -> list[Path]:
    """
    Recursively walk bench_dir and return all unique image paths.

    Skips hidden files and directories (prefixed with '.').
    """
    logger.info("Scanning for images under %s …", bench_dir)
    paths: list[Path] = []
    for p in bench_dir.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTENSIONS and not any(
            part.startswith(".") for part in p.parts
        ):
            paths.append(p)
    unique = list(set(paths))
    logger.info("Found %d unique images.", len(unique))
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build persistent image cache for CausalDriveBench.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bench-dir",
        required=True,
        type=Path,
        help="Path to a causal_drive_bench dataset directory (e.g. .../causal_nuscenes).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Base directory for computing relative paths stored in the cache. "
            "Defaults to bench_dir.parent (i.e. the causal_drive_bench directory)."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        type=Path,
        help="Directory where shard and index files will be written.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help=(
            "Name used in file naming, e.g. 'causal_nuscenes'. "
            "Defaults to bench_dir.name."
        ),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=448,
        help="Resize images to this square side length before encoding.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG compression quality (1–95).",
    )
    parser.add_argument(
        "--n-shards",
        type=int,
        default=8,
        help="Number of shard pickle files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Number of parallel encoding threads.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the cache already exists (deletes existing shards).",
    )
    args = parser.parse_args()

    bench_dir: Path = args.bench_dir.resolve()
    if not bench_dir.exists():
        logger.error("bench_dir does not exist: %s", bench_dir)
        sys.exit(1)

    data_root: Path = (args.data_root or bench_dir.parent).resolve()
    dataset_name: str = args.dataset_name or bench_dir.name
    cache_dir: Path = args.cache_dir.resolve()

    logger.info("bench_dir    : %s", bench_dir)
    logger.info("data_root    : %s", data_root)
    logger.info("cache_dir    : %s", cache_dir)
    logger.info("dataset_name : %s", dataset_name)
    logger.info("image_size   : %d", args.image_size)
    logger.info("jpeg_quality : %d", args.jpeg_quality)
    logger.info("n_shards     : %d", args.n_shards)
    logger.info("workers      : %d", args.workers)

    pcache = PersistentImageCache()

    # Handle --force: remove existing shards and index
    if args.force and pcache.cache_exists(cache_dir, dataset_name, args.image_size):
        logger.info("--force: removing existing cache files …")
        for p in pcache.cache_files(cache_dir, dataset_name, args.image_size):
            p.unlink(missing_ok=True)
        index_path = cache_dir / PersistentImageCache._index_filename(dataset_name, args.image_size)
        index_path.unlink(missing_ok=True)

    if pcache.cache_exists(cache_dir, dataset_name, args.image_size):
        logger.info(
            "Cache already exists for '%s' @ %dpx in %s. "
            "Use --force to rebuild.",
            dataset_name, args.image_size, cache_dir,
        )
        # Report file sizes
        shards = pcache.cache_files(cache_dir, dataset_name, args.image_size)
        total_mb = sum(p.stat().st_size for p in shards) / 1e6
        logger.info("Existing cache: %d shards, %.1f MB total.", len(shards), total_mb)
        return

    t0 = time.monotonic()
    image_paths = collect_image_paths(bench_dir)
    if not image_paths:
        logger.error("No images found under %s. Is bench_dir correct?", bench_dir)
        sys.exit(1)

    pcache.build(
        image_paths=image_paths,
        data_root=data_root,
        image_size=args.image_size,
        jpeg_quality=args.jpeg_quality,
        cache_dir=cache_dir,
        n_shards=args.n_shards,
        num_workers=args.workers,
        dataset_name=dataset_name,
    )

    elapsed = time.monotonic() - t0

    # Final stats
    shards = pcache.cache_files(cache_dir, dataset_name, args.image_size)
    total_mb = sum(p.stat().st_size for p in shards) / 1e6
    print(
        f"\nCache build complete:\n"
        f"  Images    : {len(image_paths)}\n"
        f"  Shards    : {len(shards)} files\n"
        f"  Total size: {total_mb:.1f} MB\n"
        f"  Elapsed   : {elapsed:.1f} s\n"
        f"  Cache dir : {cache_dir}\n"
    )


if __name__ == "__main__":
    main()
