#!/usr/bin/env python
"""Turn a training checkpoint into a release weights file.

A training checkpoint carries the optimizer state (roughly two extra copies of every
trainable tensor) and may carry tensors for modules that the released model does not
have.  This script keeps the model tensors only, drops those extra modules, and
writes either a ``.pt`` or a ``.safetensors`` file.

Usage::

    python scripts/export_weights.py --ckpt runs/<name>/checkpoints/checkpoint_latest.pt \
        --out models/musetimbre_v1.pt
"""

import argparse
import os
from collections import OrderedDict
from pathlib import Path

import torch

# Modules that exist in some training configurations but not in the released model.
DROP_PREFIXES = ("ip_attns.", "timbre_proj.")


def group_of(key):
    """Coarse module group of a state-dict key, for the summary printout."""
    return key.split(".")[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="training checkpoint (.pt)")
    ap.add_argument("--out", required=True, help="output file (.pt or .safetensors)")
    ap.add_argument("--fp16", action="store_true",
                    help="store floating-point tensors as float16 (halves the file size)")
    args = ap.parse_args()

    print(f"Loading {args.ckpt} ...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    step = ckpt.get("step") if isinstance(ckpt, dict) else None
    sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

    kept, dropped = OrderedDict(), []
    for k, v in sd.items():
        if k.startswith(DROP_PREFIXES):
            dropped.append(k)
            continue
        t = v.detach().cpu().contiguous()
        if args.fp16 and t.is_floating_point():
            t = t.half()
        kept[k] = t

    groups = {}
    for k, v in kept.items():
        g = groups.setdefault(group_of(k), [0, 0])
        g[0] += 1
        g[1] += v.numel()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".safetensors":
        from safetensors.torch import save_file
        meta = {"step": str(step)} if step is not None else {}
        save_file(kept, str(out), metadata=meta)
    else:
        torch.save({"model": kept, "step": step}, out)

    print(f"\nWrote {out} ({out.stat().st_size / 1e9:.2f} GB)")
    print(f"  source step: {step}")
    print(f"  kept {len(kept)} tensors, dropped {len(dropped)} "
          f"({sorted({k.split('.')[0] for k in dropped})})")
    print(f"  {'group':<22}{'tensors':>10}{'parameters':>16}")
    for g, (n, p) in sorted(groups.items(), key=lambda kv: -kv[1][1]):
        print(f"  {g:<22}{n:>10}{p:>16,}")
    print(f"  {'TOTAL':<22}{sum(n for n, _ in groups.values()):>10}"
          f"{sum(p for _, p in groups.values()):>16,}")


if __name__ == "__main__":
    main()
