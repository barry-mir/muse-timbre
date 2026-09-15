"""Configuration for MuseTimbre.

All filesystem locations live in one YAML file (``configs/default.yaml``).  A user
can either edit that file, point ``MUSETIMBRE_CONFIG`` at a copy of it, or override
individual entries with environment variables:

======================================  ==========================================
Environment variable                    Overrides
======================================  ==========================================
``MUSETIMBRE_CONFIG``                   path of the YAML file itself
``MUSETIMBRE_SA3_DIR``                  ``paths.stable_audio_dir``
``MUSETIMBRE_CLAP_CKPT``                ``paths.clap_checkpoint``
``MUSETIMBRE_CKPT``                     ``paths.model_checkpoint``
``MUSETIMBRE_HF_REPO``                  ``paths.hf_repo``
``MUSETIMBRE_RUN_DIR``                  ``paths.run_dir``
``MUSETIMBRE_RENDERED_DIR``             ``data.rendered_stems_dir``
``MUSETIMBRE_REAL_DIRS``                ``data.real_audio_dirs`` (``:``-separated)
``MUSETIMBRE_BP_CACHE_RENDERED``        ``data.bp_cache_rendered``
``MUSETIMBRE_BP_CACHE_REAL``            ``data.bp_cache_real``
``MUSETIMBRE_SCAN_CACHE``               ``data.scan_cache_dir``
``MUSETIMBRE_MIDI_DIR``                 ``render.midi_dir``
``MUSETIMBRE_SOUNDFONT_DIR``            ``render.soundfont_dir``
``MUSETIMBRE_RENDER_OUT``               ``render.output_dir``
======================================  ==========================================

Relative paths are resolved against the repository root, so the defaults work from
any working directory once the assets have been placed under ``models/`` and
``data/``.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def _resolve(value) -> Path:
    p = Path(str(value)).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


@dataclass
class Paths:
    stable_audio_dir: Path
    clap_checkpoint: Path
    model_checkpoint: Path
    run_dir: Path
    hf_repo: str = "barry-mir/muse-timbre"
    hf_filename: str = "musetimbre_v1.pt"


@dataclass
class DataPaths:
    rendered_stems_dir: Path
    real_audio_dirs: List[Path]
    bp_cache_rendered: Path
    bp_cache_real: Path
    scan_cache_dir: Path
    audio_extensions: List[str] = field(default_factory=lambda: [".wav", ".flac"])
    drop_categories: List[str] = field(default_factory=lambda: ["drums", "percussion", "vocals", "other"])


@dataclass
class RenderPaths:
    midi_dir: Path
    soundfont_dir: Path
    soundfonts: List[str]
    output_dir: Path


@dataclass
class Config:
    paths: Paths
    data: DataPaths
    render: RenderPaths

    @property
    def soundfont_paths(self) -> List[Path]:
        return [self.render.soundfont_dir / name for name in self.render.soundfonts]


def load_config(path: Optional[str] = None) -> Config:
    """Read the YAML config, then apply environment-variable overrides."""
    cfg_path = Path(path or os.environ.get("MUSETIMBRE_CONFIG", DEFAULT_CONFIG_PATH))
    with open(cfg_path) as f:
        raw = yaml.safe_load(f) or {}

    p = raw.get("paths", {})
    d = raw.get("data", {})
    r = raw.get("render", {})

    def env(name, default):
        return os.environ.get(name, default)

    real_dirs_env = os.environ.get("MUSETIMBRE_REAL_DIRS")
    real_dirs = (real_dirs_env.split(os.pathsep) if real_dirs_env
                 else d.get("real_audio_dirs", []) or [])

    paths = Paths(
        stable_audio_dir=_resolve(env("MUSETIMBRE_SA3_DIR",
                                      p.get("stable_audio_dir", "models/stable-audio-3-medium-base"))),
        clap_checkpoint=_resolve(env("MUSETIMBRE_CLAP_CKPT",
                                     p.get("clap_checkpoint", "models/clap_music.pt"))),
        model_checkpoint=_resolve(env("MUSETIMBRE_CKPT",
                                      p.get("model_checkpoint", "models/musetimbre_v1.pt"))),
        run_dir=_resolve(env("MUSETIMBRE_RUN_DIR", p.get("run_dir", "runs"))),
        hf_repo=env("MUSETIMBRE_HF_REPO", p.get("hf_repo", "barry-mir/muse-timbre")),
        hf_filename=p.get("hf_filename", "musetimbre_v1.pt"),
    )
    data = DataPaths(
        rendered_stems_dir=_resolve(env("MUSETIMBRE_RENDERED_DIR",
                                        d.get("rendered_stems_dir", "data/rendered_stems"))),
        real_audio_dirs=[_resolve(x) for x in real_dirs],
        bp_cache_rendered=_resolve(env("MUSETIMBRE_BP_CACHE_RENDERED",
                                       d.get("bp_cache_rendered", "data/bp_cache/rendered"))),
        bp_cache_real=_resolve(env("MUSETIMBRE_BP_CACHE_REAL",
                                   d.get("bp_cache_real", "data/bp_cache/real"))),
        scan_cache_dir=_resolve(env("MUSETIMBRE_SCAN_CACHE",
                                    d.get("scan_cache_dir", "data/scan_cache"))),
        audio_extensions=list(d.get("audio_extensions", [".wav", ".flac"])),
        drop_categories=list(d.get("drop_categories", ["drums", "percussion", "vocals", "other"])),
    )
    render = RenderPaths(
        midi_dir=_resolve(env("MUSETIMBRE_MIDI_DIR", r.get("midi_dir", "data/midi"))),
        soundfont_dir=_resolve(env("MUSETIMBRE_SOUNDFONT_DIR",
                                   r.get("soundfont_dir", "data/soundfonts"))),
        soundfonts=list(r.get("soundfonts", [])),
        output_dir=_resolve(env("MUSETIMBRE_RENDER_OUT", r.get("output_dir", "data/rendered_stems"))),
    )
    return Config(paths=paths, data=data, render=render)


_CACHED: Optional[Config] = None


def get_config(path: Optional[str] = None) -> Config:
    """Process-wide cached config (reloaded when an explicit path is given)."""
    global _CACHED
    if path is not None:
        _CACHED = load_config(path)
    elif _CACHED is None:
        _CACHED = load_config()
    return _CACHED
