"""Train MuseTimbre.

The Stable Audio 3 backbone and its text conditioner stay frozen; only the pitch
encoder, the per-block pitch cross-attention layers, the timbre projection, the
per-block AdaLN layers and the CLAP timbre encoder are updated.  Training is a plain
rectified-flow objective: sample ``t``, form ``z_t = (1 - t) z_0 + t eps``, and
regress the velocity ``eps - z_0``.

Text conditioning is held at the neutral empty prompt for every step, which is the
condition the sampler treats as unconditional.

Single GPU::

    python -m musetimbre.train --device cuda:0

Two GPUs (the configuration the released weights were trained with: batch 2 per GPU
x 2 GPUs x 4 accumulation steps = effective batch 16, 250k steps)::

    torchrun --nproc_per_node=2 -m musetimbre.train --distributed
"""

import os

# Transformers/HF are consulted while building the text conditioner; skip the network
# round-trips, the weights ship with the backbone.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import argparse
import json
import logging
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from musetimbre.config import get_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")

DEFAULT_CFG = {
    "lr": 1e-4,
    "lr_timbre_encoder": 1e-5,
    "weight_decay": 0.01,
    "warmup_steps": 1000,
    "total_steps": 250000,
    "batch_size": 2,        # per GPU
    "grad_accum": 4,
    "save_every": 5000,
    "log_every": 50,
}


def load_sa3(device, sa3_dir):
    """Load the frozen Stable Audio 3 backbone and return (model, config)."""
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.utils import load_ckpt_state_dict

    with open(f"{sa3_dir}/model_config.json") as f:
        model_config = json.load(f)
    model = create_model_from_config(model_config)
    sd = load_ckpt_state_dict(f"{sa3_dir}/model.safetensors")
    model.load_state_dict(sd, strict=False)
    del sd
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    # The autoencoder is used only to encode the target, but its graph must stay live
    # so the pretransform can run inside the training step.
    model.pretransform.enable_grad = True
    model.pretransform.model.pretransform.enable_grad = True
    return model, model_config


def get_lr(step, warmup_steps, total_steps, base_lr):
    """Linear warmup, then cosine decay."""
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Train the MuseTimbre pitch and timbre adapters on a frozen backbone.")
    parser.add_argument("--distributed", action="store_true",
                        help="run under torchrun with DistributedDataParallel")
    parser.add_argument("--device", default="cuda:0",
                        help="device for single-GPU training (ignored with --distributed)")
    parser.add_argument("--name", default=None, help="run name (default: timestamp)")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("--config", default=None, help="path to a YAML config file")
    parser.add_argument("--ref-mode", default="headtail", choices=["headtail", "diffclip"],
                        help="timbre reference for real audio: the other end of the same "
                             "window, or a different window of the same recording")
    parser.add_argument("--mix-ratio", type=float, default=0.5,
                        help="fraction of each batch drawn from the rendered corpus")
    parser.add_argument("--exclude-sources", nargs="*", default=[],
                        help="directory names under real_audio_dirs to leave out of training, "
                             "for example the corpus you evaluate on")
    parser.add_argument("--total-steps", type=int, default=None, help="override total steps")
    parser.add_argument("--batch-size", type=int, default=None, help="override per-GPU batch size")
    parser.add_argument("--grad-accum", type=int, default=None,
                        help="override gradient accumulation steps")
    parser.add_argument("--num-workers", type=int, default=4, help="dataloader workers per rank")
    return parser


def main():
    args = build_argparser().parse_args()
    conf = get_config(args.config)

    cfg = DEFAULT_CFG.copy()
    for key, value in (("total_steps", args.total_steps), ("batch_size", args.batch_size),
                       ("grad_accum", args.grad_accum)):
        if value is not None:
            cfg[key] = value

    if args.distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        rank, world_size = 0, 1
        device = torch.device(args.device)
    is_main = rank == 0

    run_name = args.name or f"musetimbre_{datetime.now().strftime('%m%d_%H%M')}"
    run_dir = Path(conf.paths.run_dir) / run_name
    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"

    writer = None
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        import yaml
        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(cfg, f)
        writer = SummaryWriter(str(log_dir))

    if is_main:
        logger.info("Loading the frozen backbone...")
    sa3_model, model_config = load_sa3(device, conf.paths.stable_audio_dir)
    sample_rate = model_config["sample_rate"]
    downsampling_ratio = sa3_model.pretransform.downsampling_ratio

    if is_main:
        logger.info("Loading the timbre encoder...")
    from musetimbre.model import TimbreControlModel, load_clap_timbre_encoder
    timbre_enc = load_clap_timbre_encoder(conf.paths.clap_checkpoint, device=device)

    model = TimbreControlModel(sa3_model, timbre_enc, device=device).to(device)

    adapter_params, encoder_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (encoder_params if "timbre_encoder" in name else adapter_params).append(p)

    optimizer = torch.optim.AdamW([
        {"params": adapter_params, "lr": cfg["lr"]},
        {"params": encoder_params, "lr": cfg["lr_timbre_encoder"]},
    ], weight_decay=cfg["weight_decay"])

    if is_main:
        logger.info(f"Trainable params: {sum(p.numel() for p in adapter_params) + sum(p.numel() for p in encoder_params):,} "
                    f"(adapters {sum(p.numel() for p in adapter_params):,}, "
                    f"timbre encoder {sum(p.numel() for p in encoder_params):,})")

    if args.distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        raw_model = model.module
    else:
        raw_model = model

    if is_main:
        logger.info("Building the mixed dataloader...")
    from musetimbre.data_mixed import build_mixed_dataloader
    dataloader = build_mixed_dataloader(
        batch_size=cfg["batch_size"], num_workers=args.num_workers,
        distributed=args.distributed, rank=rank, world_size=world_size,
        ref_mode=args.ref_mode, mix_ratio=args.mix_ratio,
        exclude_sources=args.exclude_sources,
    )

    # Neutral text condition, computed once and reused for every step.
    duration = 5.0
    latent_size = int(duration * sample_rate) // downsampling_ratio
    io_channels = sa3_model.io_channels
    with torch.no_grad():
        conditioning_tensors = sa3_model.conditioner(
            [{"prompt": "", "seconds_total": duration}], device)
        conditioning_tensors["inpaint_mask"] = [torch.zeros((1, 1, latent_size), device=device)]
        conditioning_tensors["inpaint_masked_input"] = [
            torch.zeros((1, io_channels, latent_size), device=device)]
        base_cond_inputs = sa3_model.get_conditioning_inputs(conditioning_tensors)
        model_dtype = next(sa3_model.model.parameters()).dtype
        base_cond_inputs = {k: (v.to(model_dtype) if v is not None else v)
                            for k, v in base_cond_inputs.items()}

    step = 0
    if args.resume:
        # Load onto the CPU first: mapping straight to the GPU would copy the whole
        # checkpoint on top of the already-resident model and can exhaust VRAM.
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        missing, _ = raw_model.load_state_dict(ckpt["model"], strict=False)
        if is_main and missing:
            logger.info(f"Resume: {len(missing)} keys not in the checkpoint")
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            for st in optimizer.state.values():
                for k, v in st.items():
                    if torch.is_tensor(v):
                        st[k] = v.to(device)
        except (ValueError, KeyError):
            if is_main:
                logger.info("Optimizer state did not match; starting a fresh optimizer")
        step = ckpt.get("step", 0)
        if is_main:
            logger.info(f"Resumed from step {step}")

    if is_main:
        logger.info(f"Training: {step} -> {cfg['total_steps']} steps, "
                    f"batch={cfg['batch_size']}x{world_size}x{cfg['grad_accum']}")

    model.train()
    data_iter = iter(dataloader)
    optimizer.zero_grad()
    accum_loss = 0.0
    grad_stats = None

    while step < cfg["total_steps"]:
        try:
            batch = next(data_iter)
        except StopIteration:
            if hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(step)
            data_iter = iter(dataloader)
            batch = next(data_iter)

        audio_44k = batch["audio_44k"].to(device)
        pitch_roll = batch["pitch_roll"].to(device)
        timbre_wav = batch["timbre_wav"].to(device)
        B = audio_44k.shape[0]

        with torch.no_grad():
            audio_stereo = audio_44k.unsqueeze(1).expand(B, 2, -1)
            z_0 = sa3_model.pretransform.encode(audio_stereo)

        t = torch.rand(B, device=device).clamp(0.001, 0.999)
        noise = torch.randn_like(z_0)
        t_expand = t[:, None, None]
        z_t = (1 - t_expand) * z_0 + t_expand * noise
        v_target = noise - z_0

        text_cond = {k: v.expand(B, *[-1] * (v.dim() - 1))
                     for k, v in base_cond_inputs.items() if v is not None}

        raw_m = model.module if args.distributed else model
        loss, metrics = raw_m.forward_train(
            z_t, v_target, t, pitch_roll, timbre_wav, text_cond=text_cond, training=True)

        loss = loss / cfg["grad_accum"]
        loss.backward()
        accum_loss += loss.item()

        if (step + 1) % cfg["grad_accum"] == 0:
            # Gradient norms must be read before the optimizer clears them.
            if is_main and (step + 1) % cfg["log_every"] == 0:
                with torch.no_grad():
                    def grad_norm(params):
                        grads = [p.grad for p in params if p.grad is not None]
                        if not grads:
                            return 0.0
                        return torch.cat([g.flatten() for g in grads]).norm().item()

                    grad_stats = {
                        "timbre_enc": grad_norm(list(raw_model.timbre_encoder.parameters())),
                        "timbre_proj": grad_norm(list(raw_model.timbre_single_proj.parameters())
                                                 + list(raw_model.adaln.parameters())),
                        "pitch_enc": grad_norm(list(raw_model.pitch_encoder.parameters())),
                        "pitch_attn": grad_norm(list(raw_model.pitch_attns.parameters())),
                    }
            else:
                grad_stats = None

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)

            lr = get_lr(step, cfg["warmup_steps"], cfg["total_steps"], cfg["lr"])
            for pg in optimizer.param_groups:
                pg["lr"] = (lr * (cfg["lr_timbre_encoder"] / cfg["lr"])
                            if pg is optimizer.param_groups[-1] else lr)

            optimizer.step()
            optimizer.zero_grad()

        step += 1

        if is_main and step % cfg["log_every"] == 0:
            avg_loss = accum_loss * cfg["grad_accum"] / cfg["log_every"]
            accum_loss = 0.0
            lr = optimizer.param_groups[0]["lr"]
            # The zero-init gates start at exactly 0 and grow as the adapters learn.
            gate = sum(pa.conv_out.weight.data.abs().mean().item()
                       for pa in raw_model.pitch_attns) / len(raw_model.pitch_attns)
            gn = grad_stats or {"timbre_enc": 0, "timbre_proj": 0, "pitch_enc": 0, "pitch_attn": 0}
            logger.info(f"[{step}/{cfg['total_steps']}] loss={avg_loss:.4f} lr={lr:.2e} "
                        f"gate={gate:.4f} density={pitch_roll.mean().item():.3f} "
                        f"g_tenc={gn['timbre_enc']:.2f} g_tproj={gn['timbre_proj']:.2f} "
                        f"g_penc={gn['pitch_enc']:.2f} g_pattn={gn['pitch_attn']:.2f}")
            if writer:
                writer.add_scalar("loss/total", avg_loss, step)
                writer.add_scalar("lr", lr, step)
                writer.add_scalar("pitch/gate_norm", gate, step)
                for k, v in gn.items():
                    writer.add_scalar(f"grad/{k}", v, step)

        if is_main and step % cfg["save_every"] == 0:
            # Save only the trainable tensors. The frozen backbone is byte-identical to
            # the pretrained weights every run reloads, so storing it would add ~9 GB
            # per checkpoint for nothing; resume and inference both load with
            # strict=False and leave those keys at their pretrained values.
            trainable_keys = {n for n, p in raw_model.named_parameters() if p.requires_grad}
            model_sd = {k: v for k, v in raw_model.state_dict().items() if k in trainable_keys}
            torch.save({"step": step, "model": model_sd,
                        "optimizer": optimizer.state_dict(), "config": cfg},
                       ckpt_dir / f"checkpoint_{step:06d}.pt")
            torch.save({"step": step, "model": model_sd,
                        "optimizer": optimizer.state_dict(), "config": cfg},
                       ckpt_dir / "checkpoint_latest.pt")
            logger.info(f"Saved step {step} ({len(model_sd)} trainable tensors)")

    if is_main:
        logger.info("Done.")
        if writer:
            writer.close()
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
