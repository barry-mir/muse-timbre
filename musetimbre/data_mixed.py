"""The single-stage mixed dataloader used for training.

Rather than pre-training on rendered MIDI and then fine-tuning on real audio, every
batch is drawn from a weighted mixture of both:

* ``mix_ratio`` of the probability mass goes to the rendered corpus, uniformly;
* the rest is split **equally** between the real corpora, uniformly within each, so a
  small collection is up-weighted to match a large one instead of being drowned out.
"""

import torch
from torch.utils.data import ConcatDataset, DataLoader, Sampler

from musetimbre.data import RealAudioDataset
from musetimbre.data_rendered import RenderedStemDataset


class MixedWeightedSampler(Sampler):
    """Weighted sampling with replacement, distributed-aware and reseedable.

    Each rank draws its own indices from a rank- and epoch-dependent generator, so
    ranks see different data and ``set_epoch`` reshuffles the stream.
    """

    def __init__(self, weights, rank=0, world_size=1, seed=0, num_samples=None):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.rank = rank
        self.world_size = max(1, world_size)
        self.seed = seed
        self.epoch = 0
        self.num_samples = (num_samples if num_samples is not None
                            else len(self.weights) // self.world_size)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.rank * 100003 + self.epoch * 1000003)
        idx = torch.multinomial(self.weights, self.num_samples, replacement=True, generator=g)
        return iter(idx.tolist())

    def __len__(self):
        return self.num_samples


def build_mixed_dataloader(batch_size=2, num_workers=4, distributed=False, rank=0,
                           world_size=1, ref_mode="headtail", mix_ratio=0.5, seed=0,
                           stems_dir=None, audio_dirs=None):
    rendered = RenderedStemDataset(stems_dir)
    # Rendered clips are a single take, so head/tail is already the cleanest pairing:
    # identical timbre, different notes. diffclip only helps for real recordings.
    real = RealAudioDataset(audio_dirs, ref_mode=ref_mode)

    n_rendered, n_real = len(rendered), len(real)
    if n_rendered == 0 or n_real == 0:
        raise RuntimeError(f"the mixed loader needs both datasets non-empty "
                           f"(rendered={n_rendered}, real={n_real})")

    w_rendered = [mix_ratio / n_rendered] * n_rendered

    real_mass = 1.0 - mix_ratio
    counts = {}
    for s in real.source_of:
        counts[s] = counts.get(s, 0) + 1
    per_source_mass = real_mass / len(counts)
    w_real = [per_source_mass / counts[s] for s in real.source_of]

    weights = w_rendered + w_real            # order must match the concat below
    concat = ConcatDataset([rendered, real])

    if rank == 0:
        print(f"[mixed] rendered={n_rendered} (mass {mix_ratio}), real={n_real} "
              f"across {len(counts)} sources {counts} (mass {real_mass:.2f})", flush=True)

    sampler = MixedWeightedSampler(weights, rank=rank, world_size=world_size, seed=seed,
                                   num_samples=len(concat) // max(1, world_size))
    return DataLoader(concat, batch_size=batch_size, sampler=sampler,
                      num_workers=num_workers, pin_memory=True, drop_last=True,
                      persistent_workers=(num_workers > 0))
