# MuseTimbre

Play a phrase on one instrument, hand the model a few seconds of another instrument,
and get the same notes in that timbre. MuseTimbre adds two lightweight, separately
controllable conditions, **pitch** and **timbre**, to a frozen text-to-audio
diffusion backbone, so neither condition has to be described in words and the backbone
itself is never fine-tuned.

- **Backbone.** Stable Audio 3 Medium (base variant), a rectified-flow DiT over a
  latent audio space, together with its t5gemma text conditioner. Completely frozen.
- **Pitch.** A 176-channel binary piano roll (88 note-activation + 88 onset channels
  at ~10.77 frames/s) goes through a five-layer 1-D CNN and enters *every* DiT block
  through a decoupled cross-attention layer with rotary embeddings on Q, K and V. The
  output projection is zero-initialised, so the branch starts as a no-op. The roll
  comes from transcribed audio or straight from a MIDI score. The two are the same
  representation, which is why the model accepts either at inference.
- **Timbre.** The audio tower of LAION-CLAP (HTSAT-base) reads the raw 44.1 kHz
  reference clip and returns one 512-d embedding, fine-tuned end to end. A small MLP
  lifts it to the DiT width and a per-block zero-initialised linear layer turns it into
  an AdaLN scale/shift applied right after the pitch cross-attention.
- **Guidance.** During training the conditions are dropped in mutually exclusive draws
  (both dropped 10%, timbre alone 20%, pitch alone 20%), so multi-condition classifier-free guidance is available at
  sampling time:
  `v = v_u + λ_p (v_p − v_u) + λ_t (v_t − v_u)`, with λ_p = λ_t = 2 by default.

Only ~420 M parameters are trained; the backbone's ~2.3 B stay untouched.

## Install

```bash
conda create -n musetimbre python=3.10 && conda activate musetimbre
pip install -r requirements.txt          # install the CUDA build of torch that matches your GPU
pip install -e .                         # optional; or just run from the repo root
```

`scripts/render_midi.py` additionally needs **fluidsynth** (`apt install fluidsynth`
or `conda install -c conda-forge fluidsynth`). Nothing else in the repo requires it.

## Assets to download

Place these anywhere and point the config at them (see below). None of them are
included in this repository.

| Asset | Where | Notes |
| --- | --- | --- |
| Stable Audio 3 Medium, **base** variant | Stability AI on Hugging Face | Needs `model_config.json`, `model.safetensors` and the `t5gemma-b-b-ul2/` directory. Covered by the Stability AI Community License and the Gemma Terms of Use. Read and accept them; they restrict commercial use. |
| LAION-CLAP **music** checkpoint (HTSAT-base, `music_*.pt`) | LAION-AI/CLAP releases | Used to initialise the timbre encoder. Needed for training *and* inference, because the released weights store the fine-tuned encoder in the same layout. |
| MuseTimbre weights (`musetimbre_v1.pt`) | project release page | 1.7 GB, trainable modules only (pitch encoder, pitch cross-attention, timbre projection, AdaLN, timbre encoder). |

## Configuration

Every path lives in [`configs/default.yaml`](configs/default.yaml). Edit it in place,
or copy it and set `MUSETIMBRE_CONFIG=/path/to/your.yaml`. Relative paths resolve
against the repository root, so the defaults work once assets sit under `models/` and
`data/`. Individual entries can also be overridden with environment variables:

| Variable | Overrides |
| --- | --- |
| `MUSETIMBRE_CONFIG` | the config file itself |
| `MUSETIMBRE_SA3_DIR` | `paths.stable_audio_dir` |
| `MUSETIMBRE_CLAP_CKPT` | `paths.clap_checkpoint` |
| `MUSETIMBRE_CKPT` | `paths.model_checkpoint` |
| `MUSETIMBRE_RUN_DIR` | `paths.run_dir` |
| `MUSETIMBRE_RENDERED_DIR` | `data.rendered_stems_dir` |
| `MUSETIMBRE_REAL_DIRS` | `data.real_audio_dirs` (`:`-separated) |
| `MUSETIMBRE_BP_CACHE_RENDERED` / `MUSETIMBRE_BP_CACHE_REAL` | the posteriorgram caches |
| `MUSETIMBRE_SCAN_CACHE` | `data.scan_cache_dir` |
| `MUSETIMBRE_MIDI_DIR`, `MUSETIMBRE_SOUNDFONT_DIR`, `MUSETIMBRE_RENDER_OUT` | the `render:` section |

## Inference

```bash
# notes from a recording, timbre from another recording
python -m musetimbre.infer --source source.wav --reference reference.wav --out out.wav

# notes from a MIDI score instead
python -m musetimbre.infer --midi score.mid --midi-start 12.0 \
    --reference reference.wav --out out.wav

# stronger pitch adherence, weaker timbre transfer, more sampling steps
python -m musetimbre.infer --source source.wav --reference reference.wav --out out.wav \
    --lambda-p 3 --lambda-t 1.5 --steps 50 --seed 0

# add an acoustic-environment prompt through the frozen text conditioner
python -m musetimbre.infer --source source.wav --reference reference.wav --out out.wav \
    --text "in a large reverberant hall" --lambda-text 1.0
```

Notes:

- Output is a 5 s clip at 44.1 kHz, the crop length the model was trained on. Render
  longer material one window at a time.
- `--source` is transcribed with Basic Pitch and thresholded (note ≥ 0.35, onset ≥ 0.50)
  into the same binary roll a MIDI score produces.
- `--reference` is loudness-normalised and its loudest 5 s window is used.
- Training always used the empty text prompt, which is also the unconditional branch of
  the guidance formula. A `--text` prompt is therefore added as a *third*, separately
  guided branch rather than replacing the base condition; leave it empty to reproduce
  the default behaviour.
- The Python API mirrors the CLI:

```python
from musetimbre.infer import load_model, load_audio, pitch_roll_from_audio, generate

model, sa3 = load_model(device="cuda:0")
reference = load_audio("reference.wav")
roll = pitch_roll_from_audio(load_audio("source.wav"))
audio = generate(model, sa3, roll, reference, device="cuda:0", lambda_p=2, lambda_t=2)
```

## Data preparation

Training draws every batch from a mixture of two corpora.

**1. Rendered MIDI.** Single MIDI tracks rendered with many SoundFonts and re-rendered
under substituted GM programs. This gives exact ground-truth piano rolls and an
extremely wide timbre range for the same notes.

```bash
# point render.midi_dir and render.soundfont_dir at your MIDI corpus and .sf2 files
python scripts/render_midi.py --max-files 50000 --workers 16
```

Each rendering is a ~10 s wav with its aligned `*_roll.npy` beside it, so this half of
the mixture needs no transcription.

**2. Real recordings.** List one or more directories of single-instrument recordings
under `data.real_audio_dirs`. Each is scanned recursively, every file is split into
consecutive 10 s windows, and the directory name is used to balance the mixture. Each
corpus contributes equally regardless of size. Pitch comes from cached Basic Pitch
posteriorgrams:

```bash
# shard across processes; one per GPU is a good default
python scripts/extract_basic_pitch.py --split real --shard 0 --num-shards 4
```

Within a 10 s window the loader takes two disjoint 5 s crops: the louder one is the
target and the other is the timbre reference, so the reference shares the instrument
but none of the notes. `--ref-mode diffclip` goes further and draws the reference from a
different window of the same recording.

## Training

The released weights were trained with an effective batch of 16 (2 per GPU × 2 GPUs ×
4 accumulation steps) for 250 k steps: AdamW, lr 1e-4 for the adapters and 1e-5 for the
timbre encoder, weight decay 0.01, 1000 warmup steps then cosine decay, gradient
clipping at 1.0, and a 50/50 mixture of rendered and real data.

```bash
torchrun --nproc_per_node=2 -m musetimbre.train --distributed --name my_run
```

Single GPU:

```bash
python -m musetimbre.train --device cuda:0 --name my_run
```

Stem sub-directories named in `data.drop_categories` (drums, percussion, vocals, other by
default) are skipped, and `--exclude-sources DIRNAME ...` leaves whole corpora out, which
is how to keep an evaluation set out of training.

Useful flags: `--mix-ratio` (rendered fraction), `--ref-mode`, `--total-steps`,
`--batch-size`, `--grad-accum`, `--resume`. Checkpoints and TensorBoard logs land in
`<run_dir>/<name>/`. Checkpoints hold the trainable tensors only. The frozen backbone
is identical to the pretrained weights every run reloads, so storing it would add ~9 GB
per checkpoint for nothing.

To publish a run, strip the optimizer state:

```bash
python scripts/export_weights.py \
    --ckpt runs/my_run/checkpoints/checkpoint_latest.pt \
    --out models/musetimbre_v1.pt
```

## Repository layout

```
musetimbre/
  config.py          paths and environment-variable overrides
  model.py           pitch encoder, cross-attention, timbre encoder, full model
  data.py            real-recording dataset
  data_rendered.py   rendered-MIDI dataset
  data_mixed.py      weighted mixture of the two
  train.py           training entry point
  infer.py           sampling entry point and CLI
  pitch/
    pianoroll.py     the 176-bin roll: from MIDI, and from a posteriorgram
    bp_extract.py    Basic Pitch transcription at the latent frame rate
configs/default.yaml
scripts/
  render_midi.py        build the rendered corpus
  extract_basic_pitch.py  cache posteriorgrams for the real corpus
  export_weights.py       training checkpoint -> release weights
```

## Citation

```bibtex
@inproceedings{cheng2027musetimbre,
  title     = {MuseTimbre: Reference-Based Timbre Control for Music Generation},
  author    = {Cheng, Yuan-Chiao and Duan, Zhiyao},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2027}
}
```

## License

MIT for the code in this repository (see [LICENSE](LICENSE)). The Stable Audio 3 and
LAION-CLAP weights carry their own licenses and are not covered by it.
