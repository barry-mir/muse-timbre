"""The 176-bin binary piano roll used as the pitch condition.

Layout: ``(176, T)``.  Rows ``0:88`` hold note activations (MIDI pitch 21..108 at
index ``pitch - 21``), rows ``88:176`` hold onsets.  Frames run at the backbone's
latent rate, ~10.77 frames/s, so a 5 s clip is 54 frames.

Exactly one representation is used everywhere -- rendered training data, real
training audio and both kinds of inference input:

* a MIDI score becomes a roll directly (:func:`midi_to_pianoroll_176`);
* audio is transcribed with Basic Pitch and the posteriorgram is thresholded
  (:func:`binarize_posteriorgram`).

Thresholding matters: the soft posteriorgram floor (harmonics, octave ghosts, decay
tails) is correlated with timbre, and leaving it in lets the pitch branch smuggle
timbre information past the reference encoder.  After binarisation the pitch channel
carries note content only.
"""

import numpy as np

NOTE_PEAK, ONSET_PEAK = 1.0, 1.0

# Measured against ground-truth MIDI on rendered stems: a 0.15 note threshold gives
# ~0.49 precision (half the detections are spurious harmonics), while 0.35 gives
# 0.91 precision / 0.82 recall and a clean roll.
NOTE_THRESHOLD, ONSET_THRESHOLD = 0.35, 0.50


def binarize_posteriorgram(pg, note_thr=NOTE_THRESHOLD, note_peak=NOTE_PEAK,
                           onset_thr=ONSET_THRESHOLD, onset_peak=ONSET_PEAK):
    """Soft Basic Pitch posteriorgram (176, T) -> binary roll of the same shape."""
    pg = np.asarray(pg, dtype=np.float32)
    out = np.zeros_like(pg, dtype=np.float32)
    note, onset = pg[:88], pg[88:176]
    out[:88][note >= note_thr] = note_peak
    out[88:176][onset >= onset_thr] = onset_peak
    return out


def midi_to_pianoroll_176(notes, n_frames, dur, note_peak=NOTE_PEAK, onset_peak=ONSET_PEAK):
    """Binary roll from symbolic notes.

    ``notes``: iterable of ``(onset_sec, midi_pitch, duration_sec)`` relative to the
    start of the window.  Returns ``(176, n_frames)``.
    """
    out = np.zeros((176, n_frames), np.float32)
    fdur = dur / n_frames
    for onset, pitch, ndur in notes:
        p = int(round(pitch)) - 21
        if not (0 <= p < 88):
            continue
        s, e = onset, onset + ndur
        if e <= 0 or s >= dur:
            continue
        fs = max(0, int(s / fdur))
        fe = min(n_frames, max(fs + 1, int(np.ceil(e / fdur))))
        out[p, fs:fe] = note_peak
        if 0 <= s < dur:
            out[88 + p, min(n_frames - 1, int(s / fdur))] = onset_peak
    return out


def notes_from_midi(midi_path):
    """Read every non-drum note of a MIDI file as ``(onset, pitch, duration)``."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    return [(nt.start, nt.pitch, nt.end - nt.start)
            for inst in pm.instruments if not inst.is_drum for nt in inst.notes]


def pianoroll_from_midi_file(midi_path, n_frames, dur, start_sec=0.0):
    """Binary roll for the ``[start_sec, start_sec + dur]`` window of a MIDI file."""
    notes = [(o - start_sec, p, d) for o, p, d in notes_from_midi(midi_path)]
    return midi_to_pianoroll_176(notes, n_frames, dur)
