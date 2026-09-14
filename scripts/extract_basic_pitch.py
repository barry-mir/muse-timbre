#!/usr/bin/env python
"""Pre-compute Basic Pitch posteriorgrams for the training corpora.

Transcription is far too slow to run inside the training loop, so both datasets read
cached ``.npy`` arrays written here.  The work splits cleanly across processes with
``--shard`` / ``--num-shards``; run one per GPU (or per CPU core).

Rendered stems already carry an exact roll next to each wav (``render_midi.py`` writes
it), so only run the ``rendered`` split if you rendered without those rolls.

Usage::

    python scripts/extract_basic_pitch.py --split real --shard 0 --num-shards 4
    python scripts/extract_basic_pitch.py --split rendered --shard 0 --num-shards 4
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from musetimbre.config import get_config
from musetimbre.pitch.bp_extract import (
    WINDOW_SEC, posteriorgram, posteriorgram_window, real_cache_path, rendered_cache_path,
)


def rendered_jobs(cfg):
    stems = Path(cfg.data.rendered_stems_dir)
    jobs = []
    for wav in sorted(stems.rglob("*.wav")):
        jobs.append((wav, rendered_cache_path(wav, stems, cfg.data.bp_cache_rendered), None))
    return jobs


def real_jobs(cfg):
    from musetimbre.data import scan_real_audio
    samples = scan_real_audio(cfg.data.real_audio_dirs, cfg.data.audio_extensions,
                              cfg.data.scan_cache_dir)
    return [(Path(s["path"]),
             real_cache_path(s["stem_key"], s["window_idx"], cfg.data.bp_cache_real),
             s["window_idx"] * WINDOW_SEC)
            for s in samples]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["rendered", "real"], default="real")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--config", default=None, help="path to a YAML config file")
    args = ap.parse_args()

    cfg = get_config(args.config)
    jobs = rendered_jobs(cfg) if args.split == "rendered" else real_jobs(cfg)
    mine = [j for i, j in enumerate(jobs) if i % args.num_shards == args.shard]
    print(f"shard {args.shard}/{args.num_shards}: {len(mine)} of {len(jobs)} items", flush=True)

    done = skipped = failed = 0
    for k, (src, cache, start) in enumerate(mine):
        if cache.exists():
            skipped += 1
            continue
        try:
            pg = posteriorgram(src) if start is None else posteriorgram_window(src, start)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, pg)
            done += 1
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  failed on {src}: {e}", flush=True)
        if (k + 1) % 200 == 0:
            print(f"  {k + 1}/{len(mine)} (written {done}, cached {skipped}, failed {failed})",
                  flush=True)
    print(f"shard {args.shard} done: written {done}, cached {skipped}, failed {failed}", flush=True)


if __name__ == "__main__":
    main()
