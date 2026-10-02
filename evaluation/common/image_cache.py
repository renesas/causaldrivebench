"""
Persistent, cross-model image cache for CausalDriveBench.

Encodes images once (PIL open → LANCZOS resize → JPEG → base64) and saves
the results to sharded pickle files on disk.  Subsequent inference runs load
the shards in seconds instead of spending ~60 s re-encoding from scratch.

The cache is model-agnostic: any model that uses base64-encoded images can
call PersistentImageCache().load() to get a drop-in replacement for the
in-memory dict produced by preload_image_cache() in inference.py.

File layout (inside cache_dir):
  <dataset_name>_<image_size>px_shard00of08.pkl  # shard data
  <dataset_name>_<image_size>px_shard01of08.pkl
  ...
  <dataset_name>_<image_size>px.index.json       # metadata

Shard format:
  dict[str, str]  →  {rel_path: base64_str}
  where rel_path = str(abs_path.relative_to(data_root))

Load return format (drop-in compat with existing code):
  dict[(str(abs_path), image_size), base64_str]
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_tqdm(iterable, **kwargs):
    """Wrap iterable with tqdm if available, else return as-is."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, **kwargs)
    except ImportError:
        total = kwargs.get("total", "?")
        desc = kwargs.get("desc", "")
        if desc:
            print(f"{desc}: 0/{total}", flush=True)
        return iterable


def _encode_image(abs_path: Path, size: int, quality: int) -> str:
    """Resize to square `size`, JPEG-encode at `quality`, return base64 string."""
    from PIL import Image

    try:
        img = Image.open(abs_path).convert("RGB")
    except Exception:
        logger.warning("[image_cache] Failed to open %s — using grey placeholder", abs_path)
        img = Image.new("RGB", (size, size), color=(128, 128, 128))

    img = img.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# PersistentImageCache
# ---------------------------------------------------------------------------

class PersistentImageCache:
    """Persistent, cross-model image cache backed by sharded pickle files."""

    # ------------------------------------------------------------------
    # File naming helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _shard_filename(dataset_name: str, image_size: int, shard_idx: int, n_shards: int) -> str:
        return f"{dataset_name}_{image_size}px_shard{shard_idx:02d}of{n_shards:02d}.pkl"

    @staticmethod
    def _index_filename(dataset_name: str, image_size: int) -> str:
        return f"{dataset_name}_{image_size}px.index.json"

    @classmethod
    def cache_files(cls, cache_dir: Path, dataset_name: str, image_size: int) -> List[Path]:
        """Return the list of expected shard paths (reads n_shards from index)."""
        index_path = cache_dir / cls._index_filename(dataset_name, image_size)
        if not index_path.exists():
            return []
        try:
            with open(index_path, "r") as f:
                meta = json.load(f)
            n_shards = meta["n_shards"]
            return [
                cache_dir / cls._shard_filename(dataset_name, image_size, i, n_shards)
                for i in range(n_shards)
            ]
        except Exception as exc:
            logger.warning("[image_cache] Could not read index %s: %s", index_path, exc)
            return []

    @classmethod
    def cache_exists(cls, cache_dir: Path, dataset_name: str, image_size: int) -> bool:
        """Return True if the index and all shard files exist."""
        index_path = cache_dir / cls._index_filename(dataset_name, image_size)
        if not index_path.exists():
            return False
        shards = cls.cache_files(cache_dir, dataset_name, image_size)
        if not shards:
            return False
        return all(p.exists() for p in shards)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        image_paths: Iterable[Path],
        data_root: Path,
        image_size: int,
        jpeg_quality: int = 85,
        cache_dir: Optional[Path] = None,
        n_shards: int = 8,
        num_workers: int = 64,
        dataset_name: str = "",
    ) -> None:
        """
        Encode all images and write sharded pickle files to cache_dir.

        Idempotent: if all shards + index already exist, returns immediately.

        Args:
            image_paths:   Absolute paths to source images to encode.
            data_root:     Base directory; keys stored as paths relative to this.
            image_size:    Resize images to this square side length (pixels).
            jpeg_quality:  JPEG compression quality (1–95).
            cache_dir:     Directory to write shard and index files.
            n_shards:      Number of shard pickle files to write.
            num_workers:   Thread pool size for parallel encoding.
            dataset_name:  Used in shard/index file naming.
        """
        if cache_dir is None:
            raise ValueError("cache_dir is required — pass the directory where shards will be written")

        cache_dir = Path(cache_dir)
        data_root = Path(data_root)

        # Fast exit if already built
        if self.cache_exists(cache_dir, dataset_name, image_size):
            logger.info(
                "[image_cache] Cache already exists for %s @ %dpx in %s — skipping build.",
                dataset_name, image_size, cache_dir,
            )
            return

        cache_dir.mkdir(parents=True, exist_ok=True)

        # Deduplicate
        unique_paths: List[Path] = list({Path(p) for p in image_paths})
        n_images = len(unique_paths)
        logger.info(
            "[image_cache] Building cache: %d images, %d shards, %d workers, quality=%d",
            n_images, n_shards, num_workers, jpeg_quality,
        )

        t0 = time.monotonic()

        # Assign each path to a shard bucket deterministically via hash
        shards: List[Dict[str, str]] = [{} for _ in range(n_shards)]

        def _encode_one(abs_path: Path):
            try:
                b64 = _encode_image(abs_path, image_size, jpeg_quality)
            except Exception as exc:
                logger.warning("[image_cache] Encode failed for %s: %s", abs_path, exc)
                b64 = _grey_placeholder(image_size, jpeg_quality)
            try:
                rel_path = str(abs_path.relative_to(data_root))
            except ValueError:
                # Path not under data_root — store as absolute string
                rel_path = str(abs_path)
            shard_idx = int(hashlib.md5(rel_path.encode()).hexdigest(), 16) % n_shards
            return rel_path, b64, shard_idx

        try:
            from tqdm import tqdm as _tqdm
            _use_tqdm = True
        except ImportError:
            _use_tqdm = False

        with ThreadPoolExecutor(max_workers=num_workers) as exe:
            futures = {exe.submit(_encode_one, p): p for p in unique_paths}
            done_iter = as_completed(futures)
            if _use_tqdm:
                from tqdm import tqdm as _tqdm
                done_iter = _tqdm(done_iter, total=n_images, desc="Encoding images", unit="img")
            for fut in done_iter:
                try:
                    rel_path, b64, shard_idx = fut.result()
                    shards[shard_idx][rel_path] = b64
                except Exception as exc:
                    logger.warning("[image_cache] Worker error: %s", exc)

        encode_elapsed = time.monotonic() - t0
        logger.info("[image_cache] Encoding done in %.1f s. Writing shards…", encode_elapsed)

        # Write shards
        t_write = time.monotonic()
        for i, shard_data in enumerate(shards):
            shard_path = cache_dir / self._shard_filename(dataset_name, image_size, i, n_shards)
            with open(shard_path, "wb") as f:
                pickle.dump(shard_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug("[image_cache] Wrote shard %d: %d entries, %.1f MB", i, len(shard_data), shard_path.stat().st_size / 1e6)

        # Write index
        index_path = cache_dir / self._index_filename(dataset_name, image_size)
        meta = {
            "n_shards": n_shards,
            "n_images": n_images,
            "image_size": image_size,
            "dataset_name": dataset_name,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "jpeg_quality": jpeg_quality,
            "data_root_hint": str(data_root),
        }
        with open(index_path, "w") as f:
            json.dump(meta, f, indent=2)

        total_elapsed = time.monotonic() - t0
        total_size_mb = sum(
            (cache_dir / self._shard_filename(dataset_name, image_size, i, n_shards)).stat().st_size
            for i in range(n_shards)
        ) / 1e6
        logger.info(
            "[image_cache] Cache built: %d images, %.1f MB total, %.1f s elapsed.",
            n_images, total_size_mb, total_elapsed,
        )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(
        self,
        cache_dir: Path,
        dataset_name: str,
        image_size: int,
        data_root: Path,
        num_workers: int = 8,
    ) -> dict:
        """
        Load shard files into RAM and return a cache dict compatible with
        the existing _image_cache format:
            {(str(abs_path), image_size): base64_str}

        Args:
            cache_dir:    Directory containing shard and index files.
            dataset_name: Must match the name used during build().
            image_size:   Must match the size used during build().
            data_root:    Prepended to stored relative paths to recover abs paths.
            num_workers:  Thread pool size for parallel shard loading.

        Returns:
            dict keyed by (abs_path_str, image_size) for drop-in compat.

        Raises:
            FileNotFoundError: If the cache does not exist.
            ValueError: If the index is corrupt or shard files are missing.
        """
        cache_dir = Path(cache_dir)
        data_root = Path(data_root)

        index_path = cache_dir / self._index_filename(dataset_name, image_size)
        if not index_path.exists():
            raise FileNotFoundError(
                f"[image_cache] Index not found: {index_path}. "
                "Run build_image_cache.py first."
            )

        with open(index_path, "r") as f:
            meta = json.load(f)

        n_shards = meta["n_shards"]
        logger.info(
            "[image_cache] Loading %d shards for %s @ %dpx from %s …",
            n_shards, dataset_name, image_size, cache_dir,
        )

        shard_paths = [
            cache_dir / self._shard_filename(dataset_name, image_size, i, n_shards)
            for i in range(n_shards)
        ]

        missing = [p for p in shard_paths if not p.exists()]
        if missing:
            raise ValueError(
                f"[image_cache] {len(missing)} shard file(s) missing: {missing[:3]} …"
            )

        t0 = time.monotonic()

        def _load_shard(shard_path: Path) -> Dict[str, str]:
            with open(shard_path, "rb") as f:
                return pickle.load(f)

        try:
            from tqdm import tqdm as _tqdm
            _use_tqdm = True
        except ImportError:
            _use_tqdm = False

        # Load shards in parallel
        merged: Dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=min(num_workers, n_shards)) as exe:
            futures = {exe.submit(_load_shard, p): p for p in shard_paths}
            done_iter = as_completed(futures)
            if _use_tqdm:
                done_iter = _tqdm(done_iter, total=n_shards, desc="Loading cache shards", unit="shard")
            for fut in done_iter:
                try:
                    shard_data = fut.result()
                    merged.update(shard_data)
                except Exception as exc:
                    shard_path = futures[fut]
                    raise ValueError(f"[image_cache] Failed to load shard {shard_path}: {exc}") from exc

        elapsed = time.monotonic() - t0
        logger.info(
            "[image_cache] Loaded %d entries in %.2f s.", len(merged), elapsed,
        )

        # Reconstruct abs-path-keyed dict for drop-in compat
        result: dict = {}
        for rel_path, b64 in merged.items():
            abs_path_str = str(data_root / rel_path)
            result[(abs_path_str, image_size)] = b64

        return result


# ---------------------------------------------------------------------------
# Internal helper (defined after class so _encode_one closure works)
# ---------------------------------------------------------------------------

def _grey_placeholder(size: int, quality: int) -> str:
    """Return a grey placeholder image as a base64 JPEG string."""
    from PIL import Image

    img = Image.new("RGB", (size, size), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
