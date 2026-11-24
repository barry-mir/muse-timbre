"""Real recordings with MuScriptor transcriptions as the pitch condition.

Same windowing as :class:`musetimbre.data.RealAudioDataset` (10 s windows, the louder 5 s
half is the target, the other half -- or another window with ``ref_mode="diffclip"`` -- is
the timbre reference).  Only the pitch source differs: instead of thresholded Basic Pitch
posteriorgrams, the roll is built from MuScriptor note events with
:func:`musetimbre.pitch.pianoroll.midi_to_pianoroll_176`, i.e. exactly the function used
for MIDI input at inference.  A 10 s window is rolled at 2 x 54 frames, so its head and
tail halves sit on the same frame grid as a 5 s MIDI roll.

The track list comes from a manifest (``MUSETIMBRE_REAL_MANIFEST``):
``[{"path", "source", "groups", ...}]``, with notes in ``MUSETIMBRE_NOTES_DIR/<md5(path)[:16]>.npy``
as float32 rows ``(onset_s, offset_s, midi_pitch)`` for the whole file.  Windows whose
halves hold no transcribed note are left out up front (they would be rejected as silent
anyway, at the cost of a USB read each).
"""

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from musetimbre.data import CLIP_SEC, PITCH_FRAMES, WINDOW_SEC, RealAudioDataset
from musetimbre.pitch.pianoroll import midi_to_pianoroll_176


def notes_key(path):
    return hashlib.md5(path.encode()).hexdigest()[:16]


def _sounding(notes, a, b):
    return bool(np.any((notes[:, 0] < b) & (notes[:, 1] > a))) if len(notes) else False


def scan_manifest(manifest, notes_dir, cache_dir=None):
    rows = json.load(open(manifest))
    tag = hashlib.md5((manifest + str(os.path.getmtime(manifest)) + notes_dir).encode()).hexdigest()[:12]
    cache_f = Path(cache_dir) / f"notes_scan_{tag}.json" if cache_dir else None
    if cache_f is not None and cache_f.exists():
        return json.load(open(cache_f))
    samples, missing = [], 0
    for r in rows:
        nf = Path(notes_dir) / f"{notes_key(r['path'])}.npy"
        if not nf.exists():
            missing += 1
            continue
        try:
            dur = sf.info(r["path"]).duration
        except Exception:
            continue
        notes = np.load(nf)
        key = f"{r['source']}__{notes_key(r['path'])}"
        for i in range(int(dur // WINDOW_SEC)):
            s = i * WINDOW_SEC
            if _sounding(notes, s, s + CLIP_SEC) and _sounding(notes, s + CLIP_SEC, s + WINDOW_SEC):
                samples.append({"path": r["path"], "stem_key": key, "source": r["source"],
                                "window_idx": i, "notes": str(nf)})
    print(f"scan_manifest: {len(samples)} windows, {missing} tracks without notes yet", flush=True)
    if cache_f is not None and missing == 0:
        cache_f.parent.mkdir(parents=True, exist_ok=True)
        json.dump(samples, open(cache_f, "w"))
    return samples


class RealNotesDataset(RealAudioDataset):
    """Real recordings + MuScriptor rolls; everything else as in RealAudioDataset."""

    def __init__(self, manifest=None, notes_dir=None, ref_mode="headtail", cache_dir=None):
        self.ref_mode = ref_mode
        manifest = manifest or os.environ["MUSETIMBRE_REAL_MANIFEST"]
        notes_dir = notes_dir or os.environ["MUSETIMBRE_NOTES_DIR"]
        self.samples = scan_manifest(manifest, notes_dir, cache_dir or os.environ.get("MUSETIMBRE_SCAN_CACHE"))
        self.stem_to_indices = defaultdict(list)
        for i, s in enumerate(self.samples):
            self.stem_to_indices[s["stem_key"]].append(i)
        self.source_of = [s["source"] for s in self.samples]
        self._notes_cache = {}
        print(f"RealNotesDataset: {len(self.samples)} windows over "
              f"{len(self.stem_to_indices)} recordings, ref_mode={self.ref_mode}")

    def _load_roll(self, sample):
        nf = sample["notes"]
        notes = self._notes_cache.get(nf)
        if notes is None:
            notes = np.load(nf)
            if len(self._notes_cache) > 512:
                self._notes_cache.clear()
            self._notes_cache[nf] = notes
        s = sample["window_idx"] * WINDOW_SEC
        sel = notes[(notes[:, 0] < s + WINDOW_SEC) & (notes[:, 1] > s)]
        rel = [(float(on - s), float(p), float(off - on)) for on, off, p in sel]
        return midi_to_pianoroll_176(rel, 2 * PITCH_FRAMES, WINDOW_SEC)
