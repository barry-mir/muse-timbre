"""Inference: take the notes from one clip and the timbre from another.

The pitch condition comes either from a source recording (transcribed with Basic
Pitch, then thresholded to a binary roll) or from a MIDI score.  The timbre condition
is a reference recording read by the CLAP encoder.  Sampling is 25 Euler steps of the
rectified flow on a sigmoid schedule over log-SNR -6..2, with multi-condition
classifier-free guidance::

    v = v_uncond + lambda_p (v_pitch - v_uncond) + lambda_t (v_timbre - v_uncond)

Each guidance branch is a separate forward pass through the frozen backbone, with the
condition that branch isolates switched on and the others zeroed -- exactly the
dropout pattern seen during training, so the "unconditional" branch is a condition
the model was actually trained on.

Examples::

    python -m musetimbre.infer --source src.wav --reference ref.wav --out out.wav
    python -m musetimbre.infer --midi score.mid --reference ref.wav --out out.wav
    python -m musetimbre.infer --source src.wav --reference ref.wav --out out.wav \
        --text "in a large reverberant hall" --lambda-p 2 --lambda-t 2 --steps 25
"""

import argparse
import json
import os

import librosa
import numpy as np
import torch

from musetimbre.config import get_config

SR = 44100
DURATION = 5.0                      # the clip length the model was trained on
SAMPLE_SIZE = int(DURATION * SR)


# ------------------------------------------------------------------ audio I/O

def normalize_loudness(audio, target_db=-18.0):
    """Scale to a target RMS level, then soft-clip."""
    rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
    if rms < 1e-8:
        return audio
    gain = 10 ** ((target_db - 20 * np.log10(rms)) / 20)
    return np.tanh(audio * gain)


def load_audio(path, sr=SR, duration=DURATION, offset=None):
    """Load one mono clip: the loudest window of the file, loudness-normalised."""
    wav_full, _ = librosa.load(path, sr=sr, mono=True)
    clip_samples = int(sr * duration)

    if offset is not None:
        start = int(offset * sr)
        wav = wav_full[start:start + clip_samples]
    elif len(wav_full) > clip_samples:
        hop = sr // 2
        best_start, best_rms = 0, -np.inf
        for i in range(0, len(wav_full) - clip_samples, hop):
            rms = np.sqrt(np.mean(wav_full[i:i + clip_samples] ** 2) + 1e-10)
            if rms > best_rms:
                best_rms, best_start = rms, i
        wav = wav_full[best_start:best_start + clip_samples]
    else:
        wav = wav_full

    if len(wav) < clip_samples:
        wav = np.pad(wav, (0, clip_samples - len(wav)))
    return normalize_loudness(wav[:clip_samples])


# --------------------------------------------------------------- pitch condition

def pitch_roll_from_audio(audio, sr=SR):
    """Transcribe a waveform and threshold it into the binary 176-bin roll."""
    from musetimbre.pitch.bp_extract import posteriorgram_from_audio
    from musetimbre.pitch.pianoroll import binarize_posteriorgram
    return binarize_posteriorgram(posteriorgram_from_audio(audio, sr).astype(np.float32))


def pitch_roll_from_midi(midi_path, start_sec=0.0, duration=DURATION):
    """Binary 176-bin roll for a window of a MIDI score."""
    from musetimbre.pitch.bp_extract import LATENT_FPS
    from musetimbre.pitch.pianoroll import pianoroll_from_midi_file
    n_frames = max(1, int(round(duration * LATENT_FPS)))
    return pianoroll_from_midi_file(midi_path, n_frames, duration, start_sec=start_sec)


# ------------------------------------------------------------------ model setup

def load_model(device="cuda", ckpt=None, sa3_dir=None, clap_ckpt=None, half=True):
    """Build the model and load the released weights. Returns ``(model, sa3)``."""
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import load_ckpt_state_dict
    from musetimbre.model import TimbreControlModel, load_clap_timbre_encoder, load_weights

    conf = get_config()
    sa3_dir = str(sa3_dir or conf.paths.stable_audio_dir)
    ckpt = str(ckpt or conf.paths.model_checkpoint)
    clap_ckpt = clap_ckpt or conf.paths.clap_checkpoint

    with open(f"{sa3_dir}/model_config.json") as f:
        model_config = json.load(f)
    sa3 = create_model_from_config(model_config)
    sd = load_ckpt_state_dict(f"{sa3_dir}/model.safetensors")
    sa3.load_state_dict(sd, strict=False)
    del sd
    sa3 = sa3.to(device)
    sa3.eval()
    sa3.requires_grad_(False)
    sa3.pretransform.enable_grad = True
    sa3.pretransform.model.pretransform.enable_grad = True
    if half:
        sa3.model.half()

    timbre_enc = load_clap_timbre_encoder(clap_ckpt, device=device)
    model = TimbreControlModel(sa3, timbre_enc, device=device).to(device)
    load_weights(model, ckpt)
    model.eval()
    return model, sa3


# --------------------------------------------------------------------- sampling

@torch.no_grad()
def generate(model, sa3, pitch_roll, reference_wav, device="cuda",
             lambda_p=2.0, lambda_t=2.0, steps=25, prompt="", lambda_text=0.0, seed=None):
    """Sample one 5 s clip with the given pitch roll and timbre reference."""
    model_dtype = next(sa3.model.parameters()).dtype
    io_channels = sa3.io_channels
    latent_size = SAMPLE_SIZE // sa3.pretransform.downsampling_ratio

    roll = torch.from_numpy(np.asarray(pitch_roll, dtype=np.float32)).unsqueeze(0).to(device)
    pitch_ctx = model.encode_pitch(roll, latent_size).to(model_dtype)
    null_pitch_ctx = torch.zeros_like(pitch_ctx)

    ref = torch.as_tensor(reference_wav, dtype=torch.float32).unsqueeze(0).to(device)
    timbre_vec = model.encode_timbre(ref).to(model_dtype)
    # Dropped conditions were zeroed during training, so zeros are the null condition.
    null_timbre_vec = torch.zeros_like(timbre_vec)

    def build_ci(text):
        ct = sa3.conditioner([{"prompt": text, "seconds_total": DURATION}], device)
        ct["inpaint_mask"] = [torch.zeros(1, 1, latent_size, device=device)]
        ct["inpaint_masked_input"] = [torch.zeros(1, io_channels, latent_size, device=device)]
        ci = sa3.get_conditioning_inputs(ct)
        return {k: (v.to(model_dtype) if v is not None else v) for k, v in ci.items()}

    use_text = bool(prompt) and lambda_text > 0.0
    base_ci = build_ci("")                       # the neutral prompt used in training
    text_ci = build_ci(prompt) if use_text else None

    if seed is not None:
        torch.manual_seed(seed)
    z = torch.randn(1, io_channels, latent_size, device=device, dtype=torch.float32)

    logsnr = torch.linspace(-6.0, 2.0, steps + 1)
    t_schedule = torch.sigmoid(-logsnr)
    t_schedule[0] = 1.0
    t_schedule[-1] = 0.0

    use_cfg = lambda_p != 1.0 or lambda_t != 1.0 or use_text

    def forward_with(p_ctx, t_vec, ci, t_tensor):
        model._pitch_context = p_ctx
        model._timbre_vec = t_vec
        v = sa3.model(z.to(model_dtype), t_tensor, **ci)
        model.clear_conditions()
        return v.float()

    for i in range(steps):
        t_curr, t_prev = t_schedule[i], t_schedule[i + 1]
        dt = t_prev - t_curr
        t_tensor = t_curr * torch.ones((1,), dtype=model_dtype, device=device)

        if use_cfg:
            v_uncond = forward_with(null_pitch_ctx, null_timbre_vec, base_ci, t_tensor)
            v_pitch = forward_with(pitch_ctx, null_timbre_vec, base_ci, t_tensor)
            v_timbre = forward_with(null_pitch_ctx, timbre_vec, base_ci, t_tensor)
            v = v_uncond + lambda_p * (v_pitch - v_uncond) + lambda_t * (v_timbre - v_uncond)
            if use_text:
                v_text = forward_with(null_pitch_ctx, null_timbre_vec, text_ci, t_tensor)
                v = v + lambda_text * (v_text - v_uncond)
        else:
            v = forward_with(pitch_ctx, timbre_vec, base_ci, t_tensor)

        z = z + dt * v

    audio = sa3.pretransform.decode(z)
    model.clear_conditions()
    return normalize_loudness(audio[0].cpu().float().numpy().mean(axis=0))


# -------------------------------------------------------------------------- CLI

def build_argparser():
    p = argparse.ArgumentParser(
        description="Generate audio with the pitch of a source clip or MIDI score and "
                    "the timbre of a reference recording.")
    p.add_argument("--source", default=None,
                   help="source audio whose notes drive the generation (transcribed "
                        "with Basic Pitch); omit when --midi is given")
    p.add_argument("--midi", default=None,
                   help="MIDI score to use as the pitch condition instead of --source")
    p.add_argument("--midi-start", type=float, default=0.0,
                   help="start time in seconds of the MIDI window to render")
    p.add_argument("--reference", required=True, help="reference audio supplying the timbre")
    p.add_argument("--out", required=True, help="output wav path")
    p.add_argument("--text", default="",
                   help="optional prompt for the frozen text conditioner, e.g. a room or "
                        "recording character; training used the empty prompt")
    p.add_argument("--lambda-text", type=float, default=1.0,
                   help="guidance weight for --text (ignored without a prompt)")
    p.add_argument("--lambda-p", type=float, default=2.0, help="pitch guidance weight")
    p.add_argument("--lambda-t", type=float, default=2.0, help="timbre guidance weight")
    p.add_argument("--steps", type=int, default=25, help="Euler sampling steps")
    p.add_argument("--seed", type=int, default=None, help="random seed")
    p.add_argument("--ckpt", default=None, help="model weights (default: from the config)")
    p.add_argument("--sa3-dir", default=None, help="backbone directory (default: from the config)")
    p.add_argument("--clap-ckpt", default=None,
                   help="CLAP checkpoint used to build the timbre encoder (default: from the config)")
    p.add_argument("--config", default=None, help="path to a YAML config file")
    p.add_argument("--device", default="cuda:0", help="device to run on")
    return p


def main():
    args = build_argparser().parse_args()
    if not args.source and not args.midi:
        raise SystemExit("give either --source (audio) or --midi (score) as the pitch condition")
    get_config(args.config)

    import soundfile as sf

    reference = load_audio(args.reference)
    if args.midi:
        roll = pitch_roll_from_midi(args.midi, start_sec=args.midi_start)
    else:
        roll = pitch_roll_from_audio(load_audio(args.source))
    print(f"Pitch roll: {roll.shape}, {int((roll[:88] > 0).sum())} active note frames")

    model, sa3 = load_model(device=args.device, ckpt=args.ckpt,
                            sa3_dir=args.sa3_dir, clap_ckpt=args.clap_ckpt)
    audio = generate(model, sa3, roll, reference, device=args.device,
                     lambda_p=args.lambda_p, lambda_t=args.lambda_t, steps=args.steps,
                     prompt=args.text, lambda_text=args.lambda_text, seed=args.seed)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    sf.write(out, audio, SR)
    print(f"Wrote {out} ({len(audio) / SR:.1f}s)")


if __name__ == "__main__":
    main()
