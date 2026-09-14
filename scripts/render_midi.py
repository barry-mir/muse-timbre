#!/usr/bin/env python
"""Render per-instrument stems from a MIDI corpus with several SoundFonts.

Each non-drum MIDI track is rendered on its own, once per SoundFont drawn for it and
once more for a handful of substituted GM programs.  That gives two kinds of pairing
for free: the same notes in many timbres, and the same timbre over many phrases.
Alongside each wav the exact 176-bin piano roll is written, aligned to the saved
window, so the rendered half of the training mixture needs no transcription at all.

The MIDI corpus, the SoundFont directory and the output directory all come from the
config (``render:`` section); see the README.

Requires fluidsynth (``pretty_midi.Instrument.fluidsynth``).

Usage::

    python scripts/render_midi.py --max-files 50000 --workers 16
"""

import argparse
import json
import logging
import random
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from musetimbre.config import get_config
from musetimbre.pitch.bp_extract import LATENT_FPS
from musetimbre.pitch.pianoroll import midi_to_pianoroll_176

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("render")

SR = 44100
MIN_NOTES = 10
MIN_DURATION = 3.0
CLIP_DURATION = 10.0            # seconds kept per rendering

GM_CLASSES = {
    range(0, 8): "piano",
    range(8, 16): "chromatic_percussion",
    range(16, 24): "organ",
    range(24, 32): "guitar",
    range(32, 40): "bass",
    range(40, 48): "strings",
    range(48, 56): "ensemble",
    range(56, 64): "brass",
    range(64, 72): "reed",
    range(72, 80): "pipe",
    range(80, 88): "synth_lead",
    range(88, 96): "synth_pad",
    range(96, 104): "synth_fx",
    range(104, 112): "ethnic",
    range(112, 120): "percussive",
    range(120, 128): "sfx",
}

# Programs each track is additionally re-rendered as, covering the major families.
CROSS_INSTRUMENT_PROGRAMS = [0, 25, 40, 56, 73, 65, 48, 19, 4, 30]


def get_gm_class(program):
    for r, name in GM_CLASSES.items():
        if program in r:
            return name
    return "unknown"


def render_one_midi(job):
    """Render every eligible track of one MIDI file. Returns the number of wavs written."""
    midi_path, output_base, sf_paths, n_soundfonts = job

    try:
        import pretty_midi
        import soundfile as sf
        midi = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception:
        return 0

    if midi.get_end_time() < MIN_DURATION:
        return 0

    midi_id = Path(midi_path).stem
    results = 0

    for i, inst in enumerate(midi.instruments):
        if inst.is_drum or len(inst.notes) < MIN_NOTES:
            continue
        duration = max(n.end for n in inst.notes) - min(n.start for n in inst.notes)
        if duration < MIN_DURATION:
            continue

        program = inst.program
        gm_class = get_gm_class(program)

        combos = [(program, gm_class, p)
                  for p in random.sample(sf_paths, min(n_soundfonts, len(sf_paths)))]
        others = [p for p in CROSS_INSTRUMENT_PROGRAMS if p != program]
        for cross_prog in random.sample(others, min(3, len(others))):
            combos.append((cross_prog, get_gm_class(cross_prog), random.choice(sf_paths)))

        for render_prog, render_class, sf_path in combos:
            sf_name = Path(sf_path).stem
            out_dir = Path(output_base) / midi_id / f"track_{i}_{render_class}"
            out_wav = out_dir / f"{sf_name}.wav"
            if out_wav.exists():
                results += 1
                continue

            try:
                solo_midi = pretty_midi.PrettyMIDI()
                solo_inst = pretty_midi.Instrument(program=render_prog)
                solo_inst.notes = inst.notes
                solo_inst.control_changes = inst.control_changes
                solo_midi.instruments.append(solo_inst)

                audio = solo_midi.fluidsynth(fs=SR, sf2_path=str(sf_path))
                if np.abs(audio).max() < 0.01:
                    continue

                # Trim to the active region (250 ms frames above 1% of the peak).
                frame_size = SR // 4
                n_frames = len(audio) // frame_size
                frame_rms = np.array([
                    np.sqrt(np.mean(audio[j * frame_size:(j + 1) * frame_size] ** 2))
                    for j in range(n_frames)])
                if n_frames == 0:
                    continue
                active = np.where(frame_rms > frame_rms.max() * 0.01)[0]
                if len(active) < 4:
                    continue
                start_frame = max(0, active[0] - 2)
                end_frame = min(n_frames, active[-1] + 2)
                audio = audio[start_frame * frame_size:end_frame * frame_size]

                if len(audio) < int(MIN_DURATION * SR):
                    continue
                if np.mean(np.abs(audio) < 0.005) > 0.3:
                    continue

                # Keep the loudest CLIP_DURATION window.
                clip_samples = int(CLIP_DURATION * SR)
                best_start = 0
                if len(audio) > clip_samples:
                    best_rms = 0.0
                    for k in range(0, len(audio) - clip_samples, SR):
                        rms = np.sqrt(np.mean(audio[k:k + clip_samples] ** 2))
                        if rms > best_rms:
                            best_rms, best_start = rms, k
                    audio = audio[best_start:best_start + clip_samples]

                rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
                if rms > 1e-8:
                    audio = np.clip(audio * 10 ** ((-18 - 20 * np.log10(rms)) / 20), -1.0, 1.0)

                out_dir.mkdir(parents=True, exist_ok=True)
                sf.write(str(out_wav), audio.astype(np.float32), SR, subtype="PCM_16")

                # Ground-truth roll for exactly the window that was saved: audio time 0
                # corresponds to MIDI time offset_sec.
                offset_sec = (start_frame * frame_size + best_start) / SR
                n_roll = max(1, int(round(len(audio) / SR * LATENT_FPS)))
                roll_notes = [(n.start - offset_sec, n.pitch, n.end - n.start) for n in inst.notes]
                roll = midi_to_pianoroll_176(roll_notes, n_roll, len(audio) / SR)
                np.save(out_dir / f"{sf_name}_roll.npy", roll.astype(np.float16))

                meta_path = out_dir / "meta.json"
                if not meta_path.exists():
                    with open(meta_path, "w") as f:
                        json.dump({"midi_id": midi_id, "track_idx": i,
                                   "original_program": program, "original_class": gm_class,
                                   "rendered_program": render_prog, "rendered_class": render_class,
                                   "n_notes": len(inst.notes), "duration": duration}, f)
                results += 1
            except Exception:
                continue

    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-files", type=int, default=50000,
                    help="cap on the number of MIDI files (0 = all)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n-soundfonts", type=int, default=3,
                    help="SoundFonts drawn per track for its original program")
    ap.add_argument("--output-dir", default=None, help="override render.output_dir")
    ap.add_argument("--midi-dir", default=None, help="override render.midi_dir")
    ap.add_argument("--config", default=None, help="path to a YAML config file")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = get_config(args.config)
    midi_dir = Path(args.midi_dir or cfg.render.midi_dir)
    out_dir = Path(args.output_dir or cfg.render.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sf_paths = [str(p) for p in cfg.soundfont_paths]
    missing = [p for p in sf_paths if not Path(p).exists()]
    if missing:
        logger.warning(f"{len(missing)} configured SoundFonts are missing, e.g. {missing[0]}")
    sf_paths = [p for p in sf_paths if Path(p).exists()]
    if not sf_paths:
        raise SystemExit(f"no SoundFonts found under {cfg.render.soundfont_dir}")
    logger.info(f"Using {len(sf_paths)} SoundFonts")

    midi_files = sorted(midi_dir.rglob("*.mid")) + sorted(midi_dir.rglob("*.midi"))
    logger.info(f"Found {len(midi_files)} MIDI files under {midi_dir}")
    if args.max_files > 0 and len(midi_files) > args.max_files:
        random.seed(args.seed)
        midi_files = random.sample(midi_files, args.max_files)
    logger.info(f"Rendering {len(midi_files)} files with {args.workers} workers")

    tasks = [(f, out_dir, sf_paths, args.n_soundfonts) for f in midi_files]
    total, done = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(render_one_midi, t) for t in tasks]
        for future in as_completed(futures):
            try:
                total += future.result()
            except Exception:
                pass
            done += 1
            if done % 500 == 0:
                logger.info(f"[{done}/{len(tasks)}] {total} stems so far")
    logger.info(f"Done. {total} stems from {done} MIDI files.")


if __name__ == "__main__":
    main()
