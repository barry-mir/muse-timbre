"""Real-recording half of the training mixture.

Every directory listed under ``data.real_audio_dirs`` in the config is scanned
recursively for audio files; each file is treated as one instrument and split into
consecutive 10 s windows.  As in the rendered loader, a window yields two 5 s crops:
the louder one is the target, the other is the timbre reference.

``ref_mode="diffclip"`` instead draws the reference from a *different* window of the
same recording, so the reference shares the instrument but none of the musical
content.  Pitch comes from the cached Basic Pitch posteriorgram of the window,
thresholded to the same binary roll the rendered loader produces.
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
from musetimbre.pitch.bp_extract import LATENT_FPS, real_cache_path
from musetimbre.pitch.pianoroll import binarize_posteriorgram

SR_AUDIO = 44100
CLIP_SEC = 5.0
WINDOW_SEC = 2 * CLIP_SEC
CLIP_SAMPLES_44K = int(CLIP_SEC * SR_AUDIO)
PITCH_FRAMES = int(round(CLIP_SEC * LATENT_FPS))   # 54 frames per 5 s crop

SILENCE_THRESHOLD_DB = -40.0
TARGET_DB = -18.0


def normalize_loudness(audio, target_db=TARGET_DB):
    """Scale to a target RMS level, then soft-clip."""
    rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
    if rms < 1e-8:
        return audio
    gain = 10 ** ((target_db - 20 * np.log10(rms)) / 20)
    return np.tanh(audio * gain)


def _is_silent(w, thr_db=SILENCE_THRESHOLD_DB):
    return 20.0 * np.log10(np.sqrt(np.mean(w ** 2) + 1e-12) + 1e-12) < thr_db


def _slice_window(roll, head, n_frames=PITCH_FRAMES):
    if roll is None:
        return None
    n = roll.shape[1]
    r = roll[:, 0:n_frames] if head else roll[:, max(0, n - n_frames):n]
    if r.shape[1] < n_frames:
        r = np.pad(r, ((0, 0), (0, n_frames - r.shape[1])))
    return r.astype(np.float32)


def stem_key(audio_path, root):
    """Stable, filesystem-safe identifier for one recording, used as the cache key."""
    rel = Path(audio_path).resolve().relative_to(Path(root).resolve()).with_suffix("")
    return f"{Path(root).name}__" + "__".join(rel.parts)


def scan_real_audio(roots, extensions, scan_cache_dir=None):
    """Index every recording under ``roots`` into 10 s windows.

    Returns a list of ``{"path", "stem_key", "source", "window_idx"}`` entries.
    """
    roots = [Path(r) for r in roots]
    key = hashlib.md5(("|".join(str(r) for r in roots) + "|"
                       + ",".join(extensions)).encode()).hexdigest()[:12]
    cache_f = Path(scan_cache_dir) / f"real_{key}.json" if scan_cache_dir else None
    if cache_f is not None and cache_f.exists():
        try:
            return json.load(open(cache_f))
        except Exception:
            pass

    import soundfile as sf
    from concurrent.futures import ThreadPoolExecutor

    files = []
    for root in roots:
        if not root.exists():
            print(f"  warning: {root} does not exist, skipping")
            continue
        for ext in extensions:
            files.extend((root, p) for p in sorted(root.rglob(f"*{ext}")))

    def windows_of(item):
        root, p = item
        try:
            info = sf.info(str(p))
            n_win = int((info.frames / info.samplerate) // WINDOW_SEC)
        except Exception:
            return []
        if n_win < 1:
            return []
        k = stem_key(p, root)
        return [{"path": str(p), "stem_key": k, "source": root.name, "window_idx": i}
                for i in range(n_win)]

    samples = []
    with ThreadPoolExecutor(max_workers=32) as ex:
        for out in ex.map(windows_of, files, chunksize=32):
            samples.extend(out)

    if cache_f is not None:
        try:
            cache_f.parent.mkdir(parents=True, exist_ok=True)
            json.dump(samples, open(cache_f, "w"))
        except OSError:
            pass
    return samples


class RealAudioDataset(Dataset):
    """Single-instrument recordings indexed as 10 s windows."""

    def __init__(self, audio_dirs=None, bp_cache=None, extensions=None,
                 scan_cache_dir=None, ref_mode="headtail"):
        cfg = get_config()
        self.audio_dirs = [Path(d) for d in (audio_dirs or cfg.data.real_audio_dirs)]
        self.bp_cache = Path(bp_cache or cfg.data.bp_cache_real)
        self.extensions = list(extensions or cfg.data.audio_extensions)
        self.ref_mode = ref_mode

        self.samples = scan_real_audio(self.audio_dirs, self.extensions,
                                       scan_cache_dir or cfg.data.scan_cache_dir)
        self.stem_to_indices = defaultdict(list)
        for i, s in enumerate(self.samples):
            self.stem_to_indices[s["stem_key"]].append(i)
        self.source_of = [s["source"] for s in self.samples]

        print(f"RealAudioDataset: {len(self.samples)} windows over "
              f"{len(self.stem_to_indices)} recordings, ref_mode={self.ref_mode}")

    def __len__(self):
        return len(self.samples)

    def _load_roll(self, sample):
        cp = real_cache_path(sample["stem_key"], sample["window_idx"], self.bp_cache)
        if cp.exists():
            try:
                return binarize_posteriorgram(np.load(cp).astype(np.float32))
            except Exception:
                return None
        return None

    def _load_window(self, sample):
        try:
            w, _ = librosa.load(sample["path"], sr=SR_AUDIO, mono=True,
                                offset=sample["window_idx"] * WINDOW_SEC, duration=WINDOW_SEC)
        except Exception:
            return None
        if len(w) < 2 * CLIP_SAMPLES_44K:
            w = np.pad(w, (0, 2 * CLIP_SAMPLES_44K - len(w)))
        return w

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_full = self._load_window(sample)
        if audio_full is None:
            return self[random.randint(0, len(self.samples) - 1)]

        max_start = len(audio_full) - CLIP_SAMPLES_44K
        head = audio_full[0:CLIP_SAMPLES_44K]
        tail = audio_full[max_start:max_start + CLIP_SAMPLES_44K]
        head_is_target = np.mean(head ** 2) >= np.mean(tail ** 2)
        audio_44k, ref_44k = (head, tail) if head_is_target else (tail, head)

        # Reference from a different window of the same recording: same instrument,
        # genuinely unrelated musical content. The target and its pitch stay put.
        if self.ref_mode == "diffclip":
            pool = [j for j in self.stem_to_indices[sample["stem_key"]] if j != idx]
            if pool:
                other = self.samples[random.choice(pool)]
                rf = self._load_window(other)
                if rf is not None:
                    h2, t2 = rf[0:CLIP_SAMPLES_44K], rf[-CLIP_SAMPLES_44K:]
                    ref_44k = h2 if np.mean(h2 ** 2) >= np.mean(t2 ** 2) else t2

        if _is_silent(audio_44k) or _is_silent(ref_44k):
            return self[random.randint(0, len(self.samples) - 1)]

        audio_44k = normalize_loudness(audio_44k)

        roll = self._load_roll(sample)
        if roll is None:
            return self[random.randint(0, len(self.samples) - 1)]
        pitch_roll = _slice_window(roll, head_is_target)

        return {
            "audio_44k": torch.from_numpy(np.ascontiguousarray(audio_44k)).float(),
            "pitch_roll": torch.from_numpy(pitch_roll).float(),
            "timbre_wav": torch.from_numpy(np.ascontiguousarray(ref_44k)).float(),
        }


def build_dataloader(audio_dirs=None, batch_size=4, num_workers=4, distributed=False,
                     rank=0, world_size=1, ref_mode="headtail"):
    dataset = RealAudioDataset(audio_dirs, ref_mode=ref_mode)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if distributed else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None),
                      sampler=sampler, num_workers=num_workers, pin_memory=True,
                      drop_last=True, persistent_workers=(num_workers > 0))
