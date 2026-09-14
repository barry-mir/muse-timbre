"""Basic Pitch posteriorgrams at the backbone's latent frame rate.

Basic Pitch annotates at ~86 frames/s; the backbone's latent runs at ~10.77 frames/s
(44100 / 4096).  Note activations are mean-pooled into the coarser frames and onsets
are max-pooled so a short onset spike survives.  The result is a ``(176, T)`` array
(88 note + 88 onset channels) that :mod:`musetimbre.pitch.pianoroll` then binarises.

Because transcription is far too slow to run inside the training loop, the training
datasets read cached ``.npy`` posteriorgrams written by ``scripts/extract_basic_pitch.py``.
"""

from pathlib import Path

import numpy as np

SR = 44100
LATENT_FPS = SR / 4096.0        # ~10.766 frames per second
BP_FPS = 22050 / 256.0          # Basic Pitch annotation rate, ~86.13
CLIP_SEC = 5.0
WINDOW_SEC = 2 * CLIP_SEC       # datasets index audio in 10 s windows

_BP_MODEL = None


def bp_model():
    global _BP_MODEL
    if _BP_MODEL is None:
        from basic_pitch.inference import Model
        from basic_pitch import ICASSP_2022_MODEL_PATH
        _BP_MODEL = Model(ICASSP_2022_MODEL_PATH)
    return _BP_MODEL


def posteriorgram(path):
    """(176, n) float16 posteriorgram for a whole audio file."""
    from basic_pitch.inference import predict
    model_output, _, _ = predict(str(path), bp_model())
    note = np.asarray(model_output["note"], np.float32)      # (T, 88) at ~86 fps
    onset = np.asarray(model_output["onset"], np.float32)    # (T, 88)
    T = note.shape[0]
    n = max(1, int(round((T / BP_FPS) * LATENT_FPS)))
    edges = np.linspace(0, T, n + 1).astype(int)
    out = np.zeros((176, n), np.float32)
    for i in range(n):
        a, b = edges[i], max(edges[i] + 1, edges[i + 1])
        out[0:88, i] = note[a:b].mean(0)
        out[88:176, i] = onset[a:b].max(0)
    return out.astype(np.float16)


def posteriorgram_from_audio(audio, sr=SR):
    """(176, n) posteriorgram for an in-memory mono waveform."""
    import os
    import tempfile
    import soundfile as sf
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(tmp, np.asarray(audio, dtype=np.float32), sr)
    try:
        return posteriorgram(tmp)
    finally:
        os.unlink(tmp)


def posteriorgram_window(audio_path, start_sec, dur=WINDOW_SEC):
    """Posteriorgram for a ``[start, start + dur]`` window of a longer file."""
    import librosa
    w, _ = librosa.load(str(audio_path), sr=SR, mono=True,
                        offset=float(start_sec), duration=float(dur))
    if len(w) < int(dur * SR):
        w = np.pad(w, (0, int(dur * SR) - len(w)))
    return posteriorgram_from_audio(w, SR)


def rendered_cache_path(wav_path, stems_root, cache_root):
    """Cache location for a rendered stem (mirrors the corpus directory layout)."""
    rel = Path(wav_path).resolve().relative_to(Path(stems_root).resolve()).with_suffix(".npy")
    return Path(cache_root) / rel


def real_cache_path(stem_key, window_idx, cache_root):
    """Cache location for one 10 s window of a real recording."""
    return Path(cache_root) / f"{stem_key}__{int(window_idx)}.npy"
