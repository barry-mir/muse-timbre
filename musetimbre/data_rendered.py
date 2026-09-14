"""Rendered-MIDI half of the training mixture.

Every sample is one MIDI track rendered with one SoundFont (see
``scripts/render_midi.py``), which gives an exact ground-truth piano roll and, across
SoundFonts and GM programs, a very wide range of timbres for the same notes.

For each rendering the loader takes the two end windows of the clip: the louder one
is the target (reconstruction + pitch condition) and the other one is the timbre
reference.  Renderings are ~10 s and crops are 5 s, so the two windows are disjoint --
the reference has the identical timbre but different notes, and there is no content
to copy directly.
"""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from musetimbre.config import get_config
from musetimbre.pitch.bp_extract import LATENT_FPS, rendered_cache_path
from musetimbre.pitch.pianoroll import binarize_posteriorgram

SR = 44100
CLIP_SEC = 5.0
CLIP_SAMPLES = int(CLIP_SEC * SR)
PITCH_FRAMES = int(round(CLIP_SEC * LATENT_FPS))   # 54 frames per 5 s crop


def _is_silent(w, thr_db=-40.0):
    return 20.0 * np.log10(np.sqrt(np.mean(w ** 2) + 1e-12) + 1e-12) < thr_db


def _normalize(audio, target_db=-18.0):
    rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
    if rms <= 1e-8:
        return audio
    gain = 10 ** ((target_db - 20 * np.log10(rms)) / 20)
    return np.clip(audio * gain, -1.0, 1.0)


def _slice_window(roll, head, n_frames=PITCH_FRAMES):
    n = roll.shape[1]
    r = roll[:, 0:n_frames] if head else roll[:, max(0, n - n_frames):n]
    if r.shape[1] < n_frames:
        r = np.pad(r, ((0, 0), (0, n_frames - r.shape[1])))
    return r.astype(np.float32)


class RenderedStemDataset(Dataset):
    """Renderings of single MIDI tracks, indexed as (target crop, reference crop)."""

    # Non-pitched / heterogeneous GM families, identified by the track-directory suffix.
    DROP_INSTRUMENTS = frozenset({"percussive", "sfx", "synth_fx", "drums", "sound_effects"})

    def __init__(self, stems_dir=None, bp_cache=None, scan_cache_dir=None,
                 drop_instruments=None):
        cfg = get_config()
        self.stems_dir = Path(stems_dir or cfg.data.rendered_stems_dir)
        self.bp_cache = Path(bp_cache or cfg.data.bp_cache_rendered)
        scan_cache_dir = Path(scan_cache_dir or cfg.data.scan_cache_dir)
        self.drop_instruments = set(self.DROP_INSTRUMENTS if drop_instruments is None
                                    else drop_instruments)

        # Scanning durations over a large rendered corpus is slow, so the surviving
        # file list is cached on disk. Delete the cache file to force a rescan.
        key = hashlib.md5((str(self.stems_dir) + "|" + ",".join(sorted(self.drop_instruments))
                           + f"|{2 * CLIP_SEC:.0f}").encode()).hexdigest()[:12]
        cache_f = scan_cache_dir / f"rendered_{key}.json"
        self.samples = []
        n_skipped = 0
        if cache_f.exists():
            try:
                self.samples = [{"wav": w} for w in json.load(open(cache_f))]
                print(f"  loaded {len(self.samples)} renderings from {cache_f.name}")
            except Exception:
                self.samples = []
        if not self.samples:
            import soundfile as sf
            from concurrent.futures import ThreadPoolExecutor

            candidates = []
            for midi_dir_path in sorted(self.stems_dir.iterdir()):
                if not midi_dir_path.is_dir():
                    continue
                for track_dir in sorted(midi_dir_path.iterdir()):
                    if not track_dir.is_dir():
                        continue
                    parts = track_dir.name.split("_")
                    instr = "_".join(parts[2:]) if len(parts) > 2 else track_dir.name
                    if instr in self.drop_instruments:
                        continue
                    candidates.extend(str(w) for w in track_dir.glob("*.wav"))
            print(f"  scanning durations of {len(candidates)} renderings...")

            def long_enough(w):
                try:
                    info = sf.info(w)
                    return w if (info.frames / info.samplerate) >= (2 * CLIP_SEC) else None
                except Exception:
                    return None

            with ThreadPoolExecutor(max_workers=32) as ex:
                for r in ex.map(long_enough, candidates, chunksize=64):
                    if r is not None:
                        self.samples.append({"wav": r})
            n_skipped = len(candidates) - len(self.samples)
            try:
                scan_cache_dir.mkdir(parents=True, exist_ok=True)
                json.dump([s["wav"] for s in self.samples], open(cache_f, "w"))
            except OSError:
                pass

        def instrument_of(wav_path):
            parts = Path(wav_path).parent.name.split("_")
            return "_".join(parts[2:]) if len(parts) > 2 else Path(wav_path).parent.name

        self.instrument_of = [instrument_of(s["wav"]) for s in self.samples]
        self.instr_to_indices = defaultdict(list)
        for i, ins in enumerate(self.instrument_of):
            self.instr_to_indices[ins].append(i)

        print(f"RenderedStemDataset: {len(self.samples)} renderings >= {2 * CLIP_SEC:.0f}s "
              f"(skipped {n_skipped}), {len(self.instr_to_indices)} instrument families")

    def __len__(self):
        return len(self.samples)

    def _load_roll(self, wav_path):
        """Ground-truth roll written next to the rendering, else the cached posteriorgram."""
        rp = Path(str(wav_path)[:-4] + "_roll.npy")
        if rp.exists():
            try:
                return np.load(rp).astype(np.float32)
            except Exception:
                pass
        cp = rendered_cache_path(wav_path, self.stems_dir, self.bp_cache)
        if cp.exists():
            try:
                return binarize_posteriorgram(np.load(cp).astype(np.float32))
            except Exception:
                return None
        return None

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            full, _ = librosa.load(sample["wav"], sr=SR, mono=True)
        except Exception:
            full = np.zeros(CLIP_SAMPLES, dtype=np.float32)
        if len(full) < CLIP_SAMPLES:
            full = np.pad(full, (0, CLIP_SAMPLES - len(full)))

        max_start = len(full) - CLIP_SAMPLES
        head = full[0:CLIP_SAMPLES]
        tail = full[max_start:max_start + CLIP_SAMPLES]
        head_is_target = np.mean(head ** 2) >= np.mean(tail ** 2)
        audio_44k, ref_44k = (head.copy(), tail.copy()) if head_is_target else (tail.copy(), head.copy())

        # A silent target teaches the model to emit silence; a silent reference gives
        # the timbre encoder nothing to read. Resample instead.
        if _is_silent(audio_44k) or _is_silent(ref_44k):
            return self[random.randint(0, len(self.samples) - 1)]

        audio_44k = _normalize(audio_44k)

        roll = self._load_roll(sample["wav"])
        if roll is None:
            return self[random.randint(0, len(self.samples) - 1)]
        pitch_roll = _slice_window(roll, head_is_target)

        return {
            "audio_44k": torch.from_numpy(np.ascontiguousarray(audio_44k)).float(),
            "pitch_roll": torch.from_numpy(pitch_roll).float(),
            "timbre_wav": torch.from_numpy(np.ascontiguousarray(ref_44k)).float(),
        }


def build_rendered_dataloader(stems_dir=None, batch_size=2, num_workers=4,
                              distributed=False, rank=0, world_size=1):
    dataset = RenderedStemDataset(stems_dir)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if distributed else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None),
                      sampler=sampler, num_workers=num_workers, pin_memory=True,
                      drop_last=True, persistent_workers=(num_workers > 0))
