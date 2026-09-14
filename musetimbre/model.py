"""MuseTimbre: pitch- and timbre-controlled music generation on a frozen backbone.

Architecture
------------
* **Backbone** — Stable Audio 3 Medium (base variant), a rectified-flow DiT operating
  in a latent audio space.  It is loaded once and kept completely frozen, together
  with its t5gemma text conditioner.
* **Pitch branch** — a 176-channel binary piano roll (88 note-activation + 88 onset
  channels, ~10.77 frames/s) is encoded by a five-layer 1-D CNN and injected into
  *every* DiT block through a decoupled cross-attention layer with rotary position
  embeddings applied to Q, K and V.  The output projection is zero-initialised, so
  the branch starts as a no-op and the frozen backbone is never disturbed.
* **Timbre branch** — the audio tower of LAION-CLAP (HTSAT-base) reads the raw
  44.1 kHz reference clip and produces a single 512-d embedding.  It is fine-tuned
  end to end.  A small MLP lifts the embedding to the DiT width, and a per-block
  zero-initialised linear layer turns it into an AdaLN scale/shift that modulates the
  block output right after the pitch cross-attention.
* **Guidance** — during training pitch and timbre are dropped independently
  (both 10%, timbre 20%, pitch 20%), which makes multi-condition classifier-free
  guidance available at sampling time.

Only the pitch encoder, the pitch cross-attention layers, the timbre projection, the
per-block AdaLN layers and the CLAP encoder are trainable; everything else is frozen.
"""

import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

DIT_DIM = 1536
DIT_HEADS = 24
N_PITCH_BINS = 176


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


# ---------------------------------------------------------------- rotary embedding

class RotaryEmbedding(nn.Module):
    """Rotary position embedding over one attention head (dim = head_dim)."""

    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, position_ids):
        freqs = torch.einsum("bi,j->bij", position_ids.float(), self.inv_freq.to(x.device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
    x1, x2 = x.unbind(-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    return (x * cos.unsqueeze(1)) + (rotate_half(x) * sin.unsqueeze(1))


# ------------------------------------------------------- decoupled cross-attention

class IPCrossAttention(nn.Module):
    """Decoupled cross-attention added to one transformer block.

    The block's own queries are reused; only a fresh key/value projection and a
    zero-initialised output convolution are learned, so the layer contributes nothing
    at initialisation.  With ``use_rope=True`` rotary embeddings are applied to Q, K
    and V, which is what the pitch branch uses to stay time-aligned with the latent.
    """

    def __init__(self, dim, dim_kv, num_heads, differential=False, use_rope=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.differential = differential
        self.use_rope = use_rope

        if differential:
            self.to_kv_ip = nn.Linear(dim_kv, dim * 3, bias=False)
        else:
            self.to_kv_ip = nn.Linear(dim_kv, dim * 2, bias=False)

        self.conv_out = zero_module(nn.Conv1d(dim, dim, kernel_size=1, bias=False))

        if use_rope:
            self.rope = RotaryEmbedding(dim=self.head_dim)

    def init_from_pretrained(self, pretrained_to_kv_weight):
        """Warm-start the key/value projection from the block's frozen cross-attention."""
        with torch.no_grad():
            if pretrained_to_kv_weight.shape == self.to_kv_ip.weight.shape:
                self.to_kv_ip.weight.copy_(pretrained_to_kv_weight)

    def forward(self, x, context):
        B, N, C = x.shape
        H = self.num_heads
        D = self.head_dim
        T = context.shape[1]

        q = x.view(B, N, H, D).transpose(1, 2)

        kv = self.to_kv_ip(context)
        if self.differential:
            k, _, v = kv.chunk(3, dim=-1)
        else:
            k, v = kv.chunk(2, dim=-1)

        k = k.view(B, -1, H, D).transpose(1, 2)
        v = v.view(B, -1, H, D).transpose(1, 2)

        if self.use_rope:
            q_pos = torch.arange(N, device=x.device, dtype=torch.long) * (T / max(N, 1))
            q_pos = q_pos.unsqueeze(0).expand(B, -1)
            k_pos = torch.arange(T, device=x.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
            v_pos = k_pos

            orig_dtype = q.dtype

            q_cos, q_sin = self.rope(q, q_pos)
            q = apply_rope(q.float(), q_cos, q_sin).to(orig_dtype)

            k_cos, k_sin = self.rope(k, k_pos)
            k = apply_rope(k.float(), k_cos, k_sin).to(orig_dtype)

            v_cos, v_sin = self.rope(v, v_pos)
            v = apply_rope(v.float(), v_cos, v_sin).to(orig_dtype)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.conv_out(out.transpose(1, 2)).transpose(1, 2)
        return out


# ----------------------------------------------------------------- pitch encoder

class PitchEncoder(nn.Module):
    """176-channel binary piano roll -> DiT-width sequence.

    Input rows 0:88 are note activations, rows 88:176 are onsets, both binary and
    sampled at the backbone's latent frame rate.  The representation is produced
    either by transcribing the source audio (Basic Pitch, then thresholded) or
    directly from a MIDI score, so audio and symbolic input look identical here.
    """

    def __init__(self, n_bins=N_PITCH_BINS, dit_dim=DIT_DIM):
        super().__init__()
        self.n_bins = n_bins
        self.encoder = nn.Sequential(
            nn.Conv1d(n_bins, 256, 3, padding=0, stride=2), nn.SiLU(),
            nn.Conv1d(256, 256, 3, padding=1), nn.SiLU(),
            nn.Conv1d(256, 512, 3, padding=1, stride=2), nn.SiLU(),
            nn.Conv1d(512, 512, 3, padding=1), nn.SiLU(),
            nn.Conv1d(512, dit_dim, 3, padding=1),
        )

    def forward(self, x):
        """x: (B, 176, T) binary piano roll."""
        if x.shape[1] < self.n_bins:
            x = F.pad(x, (0, 0, 0, self.n_bins - x.shape[1]))
        return self.encoder(x)


# ----------------------------------------------------------------- timbre encoder

class CLAPTimbreEncoder(nn.Module):
    """LAION-CLAP audio tower (HTSAT-base) used as a fine-tunable timbre encoder.

    Takes the raw 44.1 kHz reference waveform and returns one global 512-d embedding.
    The text tower is frozen so distributed training sees no unused parameters.
    """

    def __init__(self, ckpt=None, amodel="HTSAT-base", device="cuda"):
        super().__init__()
        import laion_clap
        self.clap = laion_clap.CLAP_Module(enable_fusion=False, amodel=amodel, device=device)
        ck = str(ckpt) if ckpt else ""
        self.clap.load_ckpt(ck) if ck else self.clap.load_ckpt()
        for n, p in self.clap.model.named_parameters():
            if n.startswith("text_"):
                p.requires_grad_(False)
        self.output_dim = 512

    def forward_cls(self, wav_44k):
        """wav_44k: (B, T) at 44.1 kHz -> (B, 512), differentiable."""
        import torchaudio
        w48 = torchaudio.functional.resample(wav_44k, 44100, 48000)
        return self.clap.get_audio_embedding_from_data(x=w48, use_tensor=True)


def load_clap_timbre_encoder(ckpt=None, device="cuda"):
    """Build the timbre encoder, initialised from the LAION-CLAP music checkpoint."""
    if ckpt is None:
        ckpt = os.environ.get("MUSETIMBRE_CLAP_CKPT")
    if ckpt is None:
        from musetimbre.config import get_config
        ckpt = get_config().paths.clap_checkpoint
    return CLAPTimbreEncoder(ckpt=ckpt, device=device).to(device)


# -------------------------------------------------------------------- full model

class TimbreControlModel(nn.Module):
    """Frozen backbone + pitch cross-attention + AdaLN timbre modulation.

    ``sa3_model`` is a Stable Audio 3 conditioned diffusion wrapper; its DiT blocks
    are monkey-patched so that each one runs the pitch cross-attention and then the
    timbre AdaLN modulation after its own forward pass.  The conditions are passed
    through ``self._pitch_context`` / ``self._timbre_vec`` rather than as arguments,
    which keeps the backbone's own signature untouched and lets the sampler evaluate
    several guidance branches with one set of patched blocks.
    """

    def __init__(self, sa3_model, timbre_encoder, n_pitch_bins=N_PITCH_BINS, device="cuda"):
        super().__init__()
        self.sa3 = sa3_model
        self.timbre_encoder = timbre_encoder
        self.device = device
        self.n_pitch_bins = n_pitch_bins

        dit_dim = DIT_DIM
        num_heads = DIT_HEADS
        model_dtype = next(sa3_model.model.parameters()).dtype

        blocks_with_cross_attn = []
        for _, module in sa3_model.model.named_modules():
            if hasattr(module, "cross_attn") and hasattr(module.cross_attn, "to_kv"):
                blocks_with_cross_attn.append(module)

        # Pitch: encoder + one RoPE cross-attention per block.
        self.pitch_encoder = PitchEncoder(n_bins=n_pitch_bins, dit_dim=dit_dim)
        self.pitch_attns = nn.ModuleList()
        for block in blocks_with_cross_attn:
            pa = IPCrossAttention(dit_dim, dit_dim, num_heads,
                                  differential=block.cross_attn.differential, use_rope=True)
            pa.init_from_pretrained(block.cross_attn.to_kv.weight.data)
            self.pitch_attns.append(pa.to(device).to(model_dtype))

        # Timbre: embedding -> DiT width, then a zero-init AdaLN per block.
        tdim = timbre_encoder.output_dim
        self.timbre_single_proj = nn.Sequential(
            nn.Linear(tdim, dit_dim), nn.GELU(), nn.Linear(dit_dim, dit_dim), nn.LayerNorm(dit_dim)
        ).to(device)
        self.adaln = nn.ModuleList([
            zero_module(nn.Linear(dit_dim, 2 * dit_dim)).to(device).to(model_dtype)
            for _ in blocks_with_cross_attn
        ])

        print(f"Pitch cross-attention: {len(self.pitch_attns)} blocks (RoPE on Q, K, V)")
        print(f"Timbre AdaLN modulation: {len(self.adaln)} blocks")

        self._pitch_context = None
        self._timbre_vec = None
        self._patch_blocks(blocks_with_cross_attn)

        # The backbone is frozen, so activation checkpointing only wastes time here.
        import types
        ct = sa3_model.model.model.transformer
        orig_ct_forward = ct.forward

        def ct_forward_no_ckpt(self_ct, *args, **kwargs):
            kwargs["use_checkpointing"] = False
            return orig_ct_forward(*args, **kwargs)

        ct.forward = types.MethodType(ct_forward_no_ckpt, ct)

        # Classifier-free-guidance dropout rates used during training.
        self.p_drop_both = 0.10
        self.p_drop_timbre = 0.20
        self.p_drop_pitch = 0.20

    # -- block patching ------------------------------------------------------

    def _patch_blocks(self, blocks):
        import types
        model_ref = self
        for i, block in enumerate(blocks):
            pitch_attn = self.pitch_attns[i]
            adaln_layer = self.adaln[i]
            original_forward = (block.forward.__wrapped__
                                if hasattr(block.forward, "__wrapped__") else block.forward)

            def make_patched_forward(orig_fwd, pitch_layer, adaln):
                def patched_forward(self_block, x, context=None, **kwargs):
                    x = orig_fwd(x, context=context, **kwargs)
                    if model_ref._pitch_context is not None:
                        x = x + pitch_layer(x, model_ref._pitch_context)
                    if model_ref._timbre_vec is not None:
                        scale, shift = adaln(model_ref._timbre_vec).chunk(2, dim=-1)
                        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
                    return x
                return patched_forward

            block.forward = types.MethodType(
                make_patched_forward(original_forward, pitch_attn, adaln_layer), block)

    # -- conditions ----------------------------------------------------------

    def encode_pitch(self, pitch_roll, latent_size):
        """(B, 176, T) piano roll -> (B, latent_size, dit_dim) cross-attention context."""
        tokens = self.pitch_encoder(pitch_roll)
        tokens = F.interpolate(tokens, size=latent_size, mode="linear", align_corners=False)
        return tokens.transpose(1, 2)

    def encode_timbre(self, ref_wav_44k):
        """(B, T) reference waveform at 44.1 kHz -> (B, dit_dim) timbre latent."""
        return self.timbre_single_proj(self.timbre_encoder.forward_cls(ref_wav_44k))

    def clear_conditions(self):
        self._pitch_context = None
        self._timbre_vec = None

    # -- training ------------------------------------------------------------

    def forward_train(self, z, v_target, t, pitch_roll, timbre_wav, text_cond=None, training=True):
        """One rectified-flow step: predict the velocity and return the MSE loss."""
        B = z.shape[0]
        N = z.shape[2]
        model_dtype = next(self.sa3.model.parameters()).dtype

        pitch_tokens = self.pitch_encoder(pitch_roll)          # (B, dit_dim, T')
        single = self.encode_timbre(timbre_wav)                # (B, dit_dim), fp32

        def drop_pitch(i):
            """Drop the whole roll, or a random temporal segment of it."""
            if random.random() < 0.5:
                pitch_tokens[i] = 0.0
            else:
                T_pitch = pitch_tokens.shape[2]
                a, b = sorted(random.sample(range(T_pitch), 2))
                pitch_tokens[i, :, a:b] = 0.0

        if training:
            for i in range(B):
                r = random.random()
                if r < self.p_drop_both:
                    single[i] = 0.0
                    pitch_tokens[i] = 0.0
                elif r < self.p_drop_both + self.p_drop_timbre:
                    single[i] = 0.0
                elif r < self.p_drop_both + self.p_drop_timbre + self.p_drop_pitch:
                    drop_pitch(i)

        self._timbre_vec = single.to(model_dtype)
        pitch_interp = F.interpolate(pitch_tokens, size=N, mode="linear", align_corners=False)
        self._pitch_context = pitch_interp.transpose(1, 2).to(model_dtype)

        ci = text_cond or {}
        v_pred = self.sa3.model(z.to(model_dtype), t.to(model_dtype), **ci)

        self.clear_conditions()

        loss = F.mse_loss(v_pred.float(), v_target.float())
        return loss, {"loss": loss.item()}


# ------------------------------------------------------------------ weight I/O

# Prefixes of modules that exist in some training configurations but not in the
# released model; they are dropped silently when a checkpoint carries them.
_IGNORED_PREFIXES = ("ip_attns.", "timbre_proj.")


def load_weights(model, path, strict=False, verbose=True):
    """Load released or training weights into ``model``.

    Returns ``(missing, unexpected)``. ``missing`` lists trainable parameters that
    the checkpoint did not provide -- it should be empty. ``unexpected`` lists
    checkpoint tensors that do not fit any module of this model; tensors belonging
    to modules only used by other training configurations are filtered out first.
    """
    path = str(path)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        ckpt, sd = {}, load_file(path, device="cpu")
    else:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    sd = {k: v for k, v in sd.items() if not k.startswith(_IGNORED_PREFIXES)}

    own = model.state_dict()
    usable = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    incompatible = sorted(k for k in sd if k not in usable)
    result = model.load_state_dict(usable, strict=False)

    # Two kinds of "missing" key are expected and are filtered out of the report:
    # the frozen backbone, which no checkpoint stores because it always equals the
    # pretrained weights, and constant buffers such as the rotary frequency table,
    # which are recomputed whenever the module is built.
    trainable_prefixes = ("pitch_encoder.", "pitch_attns.", "timbre_single_proj.",
                          "adaln.", "timbre_encoder.")
    # Frozen tensors inside the timbre encoder (its text tower, the STFT and mel
    # filterbanks) are restored from the CLAP checkpoint when the encoder is built,
    # so they are absent from a training checkpoint by design.
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    missing = [k for k in result.missing_keys
               if k in trainable_names and k.startswith(trainable_prefixes)]

    if verbose:
        step = ckpt.get("step") if isinstance(ckpt, dict) else None
        print(f"Loaded {len(usable)} tensors from {path}"
              + (f" (step {step})" if step is not None else ""))
        if missing:
            print(f"  {len(missing)} trainable tensors missing from the checkpoint: "
                  f"{missing[:5]}{' ...' if len(missing) > 5 else ''}")
        if incompatible:
            print(f"  {len(incompatible)} checkpoint tensors ignored: "
                  f"{incompatible[:5]}{' ...' if len(incompatible) > 5 else ''}")
    if strict and (missing or incompatible):
        raise RuntimeError(f"checkpoint mismatch: {len(missing)} missing, "
                           f"{len(incompatible)} unexpected")
    return missing, incompatible
