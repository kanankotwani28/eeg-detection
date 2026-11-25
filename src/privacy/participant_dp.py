"""Participant-level DP-SGD for one federated client.

At each local step:
  1. Poisson-sample each participant of the client independently with probability q.
  2. Per sampled participant, compute the gradient of the loss on ONE randomly chosen clean epoch
     (grad_mode=one_epoch) or the mean loss over all their epochs (grad_mode=mean_epochs).
  3. Clip each participant's gradient to L2 norm C.
  4. Sum, add N(0, sigma^2 C^2 I), divide by the expected batch size q * n_participants.
Per-participant gradients come from torch.func (vmap over participants); Opacus is used only for
accounting (src/privacy/accountant.py), never its per-sample PrivacyEngine.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, grad, vmap

from src.train.torch_core import PData


def clip_and_sum(per_grads: list[torch.Tensor], clip: float) -> tuple[list[torch.Tensor], torch.Tensor]:
    """per_grads: list over parameters of tensors (P, *shape). Returns (clipped sum per parameter,
    per-participant norms AFTER clipping)."""
    P = per_grads[0].shape[0]
    norms = torch.sqrt(sum(g.reshape(P, -1).pow(2).sum(1) for g in per_grads))
    factor = torch.clamp(clip / (norms + 1e-12), max=1.0)
    clipped = [g * factor.view(-1, *([1] * (g.dim() - 1))) for g in per_grads]
    post = torch.sqrt(sum(g.reshape(P, -1).pow(2).sum(1) for g in clipped))
    return [g.sum(0) for g in clipped], post


def add_noise(summed: list[torch.Tensor], sigma: float, clip: float, gen: torch.Generator) -> list[torch.Tensor]:
    return [s + sigma * clip * torch.randn(s.shape, generator=gen, dtype=s.dtype).to(s.device) for s in summed]


def _per_participant_grads(model: nn.Module, xb: torch.Tensor, yb: torch.Tensor, mb: torch.Tensor,
                           weights: torch.Tensor):
    """xb (P, E, ...), yb (P,), mb (P, E) mask of valid epochs. Loss per participant = masked mean CE."""
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def loss_one(p, x, y, m, w):
        out = functional_call(model, (p, buffers), (x,))
        ce = nn.functional.cross_entropy(out, y.expand(x.shape[0]), reduction="none")
        return w * (ce * m).sum() / m.sum().clamp(min=1)

    wb = weights[yb]
    g = vmap(grad(loss_one), in_dims=(None, 0, 0, 0, 0), randomness="different")(params, xb, yb, mb, wb)
    return [g[k] for k, _ in model.named_parameters()]


def dp_local_train(model: nn.Module, data: PData, steps: int, q: float, sigma: float, clip: float, lr: float,
                   seed: int, device, grad_mode: str = "one_epoch", weights: torch.Tensor | None = None,
                   chunk: int = 64, optimizer: str = "sgd") -> dict:
    """Run `steps` participant-level DP-SGD steps on one client's data. Returns diagnostics."""
    model.to(device).train()
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(int(rng.integers(2**31)))
    n = len(data)
    w = (weights if weights is not None else torch.ones(2)).to(device)
    # Adam/SGD only post-process the privatised gradient, so the guarantee is unchanged
    opt = torch.optim.Adam(model.parameters(), lr=lr) if optimizer == "adam" else torch.optim.SGD(model.parameters(), lr=lr)
    expected = max(q * n, 1e-12)
    sampled_counts = []
    for _ in range(steps):
        chosen = np.where(rng.random(n) < q)[0]
        sampled_counts.append(len(chosen))
        summed = [torch.zeros_like(p) for p in model.parameters()]
        for c0 in range(0, len(chosen), chunk):
            idx = chosen[c0:c0 + chunk]
            if grad_mode == "one_epoch":
                xs = [data.samples[i][rng.integers(len(data.samples[i]))][None] for i in idx]
                mask = np.ones((len(idx), 1), np.float32)
            else:
                E = max(len(data.samples[i]) for i in idx)
                xs, mask = [], np.zeros((len(idx), E), np.float32)
                for j, i in enumerate(idx):
                    s = data.samples[i]
                    pad = np.zeros((E - len(s),) + s.shape[1:], s.dtype)
                    xs.append(np.concatenate([s, pad]))
                    mask[j, : len(s)] = 1
            xb = torch.from_numpy(np.stack(xs).astype(np.float32)).to(device)
            yb = torch.from_numpy(data.y[idx]).to(device)
            pg = _per_participant_grads(model, xb, yb, torch.from_numpy(mask).to(device), w)
            s, _ = clip_and_sum(pg, clip)
            summed = [a + b for a, b in zip(summed, s, strict=True)]
        noisy = add_noise(summed, sigma, clip, gen)
        opt.zero_grad()
        for p, g in zip(model.parameters(), noisy, strict=True):
            p.grad = g / expected
        opt.step()
        model.apply_max_norm()  # post-processing: does not affect the guarantee
    return {"steps": steps, "mean_sampled": float(np.mean(sampled_counts)) if sampled_counts else 0.0}
