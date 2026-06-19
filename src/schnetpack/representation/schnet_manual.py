"""Manual fwd/bwd reimplementation of SchNetInteraction (energy-only).

The two "kernels" ``interaction_fwd`` / ``interaction_bwd`` are pure tensor
functions: no autograd, no ``ctx``, no ``requires_grad`` assumptions. They run
identically inside or outside ``torch.no_grad()`` and are the units to later
port to Triton. ``InteractionFn`` is a thin ``autograd.Function`` adapter for
dropping the block into a normal graph; ``ManualSchNetInteraction`` is the
stateful wrapper that lifts weights from a trained :class:`SchNetInteraction`.

Backward returns ``(grad_x, grad_f_ij, grad_rcut_ij)`` — the inputs through
which positions reach the block. Weights are frozen (inference), so no
parameter grads. ``f_ij``/``rcut_ij`` are shared across stacked blocks, so a
chaining caller must **accumulate** their grads across blocks.
"""

from typing import NamedTuple, Tuple

import torch

from schnetpack.nn.scatter import scatter_add
from schnetpack.representation.schnet import SchNetInteraction

__all__ = ["InteractionParams", "interaction_fwd", "interaction_bwd",
           "InteractionFn", "ManualSchNetInteraction"]


class InteractionParams(NamedTuple):
    """Raw weight tensors of one SchNetInteraction block (frozen)."""
    W_in: torch.Tensor       # in2f.weight            [F, B]
    Wf1: torch.Tensor        # filter_network[0].w    [F, R]
    bf1: torch.Tensor        #                        [F]
    Wf2: torch.Tensor        # filter_network[1].w    [F, F]
    bf2: torch.Tensor        #                        [F]
    W1: torch.Tensor         # f2out[0].weight        [B, F]
    b1: torch.Tensor         #                        [B]
    W2: torch.Tensor         # f2out[1].weight        [B, B]
    b2: torch.Tensor         #                        [B]


class Saved(NamedTuple):
    """Forward intermediates needed by the backward (flop-parity)."""
    x: torch.Tensor
    a: torch.Tensor
    idx_i: torch.Tensor
    idx_j: torch.Tensor
    p1: torch.Tensor
    Wij0: torch.Tensor
    rcut_ij: torch.Tensor
    Wij: torch.Tensor
    p2: torch.Tensor
    h: torch.Tensor


def _ssp(x: torch.Tensor) -> torch.Tensor:
    # shifted softplus; matches schnetpack.nn.activations.shifted_softplus
    return torch.nn.functional.softplus(x) - 0.6931471805599453  # log(2)


def interaction_fwd(
    x: torch.Tensor,
    f_ij: torch.Tensor,
    idx_i: torch.Tensor,
    idx_j: torch.Tensor,
    rcut_ij: torch.Tensor,
    p: InteractionParams,
) -> Tuple[torch.Tensor, Saved]:
    """Pure forward. Returns (x_out, saved). No autograd."""
    a = x @ p.W_in.t()                              # in2f (no bias, no act)

    p1 = f_ij @ p.Wf1.t() + p.bf1                   # filter layer 1 pre-act
    q1 = _ssp(p1)
    Wij0 = q1 @ p.Wf2.t() + p.bf2                   # filter layer 2 (no act)
    Wij = Wij0 * rcut_ij[:, None]

    m = a[idx_j] * Wij                              # continuous-filter conv
    s = scatter_add(m, idx_i, dim_size=x.shape[0])

    p2 = s @ p.W1.t() + p.b1                        # f2out layer 1 pre-act
    h = _ssp(p2)
    v = h @ p.W2.t() + p.b2                         # f2out layer 2 (no act)

    x_out = x + v
    return x_out, Saved(x, a, idx_i, idx_j, p1, Wij0, rcut_ij, Wij, p2, h)


def interaction_bwd(
    grad_x_out: torch.Tensor,
    s: Saved,
    p: InteractionParams,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure backward. Returns (grad_x, grad_f_ij, grad_rcut_ij). No autograd."""
    # f2out
    grad_h = grad_x_out @ p.W2
    grad_p2 = grad_h * torch.sigmoid(s.p2)
    grad_s = grad_p2 @ p.W1

    # scatter_add bwd is a gather; cfconv elementwise bwd
    grad_m = grad_s[s.idx_i]
    grad_Wij = grad_m * s.a[s.idx_j]
    grad_a_j = grad_m * s.Wij

    # grad into in2f input via scatter over neighbours
    grad_a = scatter_add(grad_a_j, s.idx_j, dim_size=s.x.shape[0])

    # cutoff envelope
    grad_rcut_ij = (grad_Wij * s.Wij0).sum(dim=1)
    grad_Wij0 = grad_Wij * s.rcut_ij[:, None]

    # filter network
    grad_q1 = grad_Wij0 @ p.Wf2
    grad_p1 = grad_q1 * torch.sigmoid(s.p1)
    grad_f_ij = grad_p1 @ p.Wf1

    # residual + in2f path
    grad_x = grad_x_out + grad_a @ p.W_in
    return grad_x, grad_f_ij, grad_rcut_ij


class InteractionFn(torch.autograd.Function):
    """Autograd adapter over the pure fwd/bwd. Param slots get no grad."""

    @staticmethod
    def forward(ctx, x, f_ij, idx_i, idx_j, rcut_ij, *params):
        p = InteractionParams(*params)
        x_out, saved = interaction_fwd(x, f_ij, idx_i, idx_j, rcut_ij, p)
        ctx.saved = saved
        ctx.params = p
        return x_out

    @staticmethod
    def backward(ctx, grad_x_out):
        gx, gf, grc = interaction_bwd(grad_x_out, ctx.saved, ctx.params)
        # x, f_ij, idx_i, idx_j, rcut_ij, then 9 params
        return (gx, gf, None, None, grc) + (None,) * 9


class ManualSchNetInteraction(torch.nn.Module):
    """Stateful wrapper holding frozen weights of one interaction block."""

    def __init__(self, params: InteractionParams):
        super().__init__()
        for name, t in params._asdict().items():
            self.register_buffer(name, t)

    @property
    def params(self) -> InteractionParams:
        return InteractionParams(*(getattr(self, n) for n in InteractionParams._fields))

    @classmethod
    def from_torch(cls, block: SchNetInteraction) -> "ManualSchNetInteraction":
        return cls(InteractionParams(
            W_in=block.in2f.weight.detach(),
            Wf1=block.filter_network[0].weight.detach(),
            bf1=block.filter_network[0].bias.detach(),
            Wf2=block.filter_network[1].weight.detach(),
            bf2=block.filter_network[1].bias.detach(),
            W1=block.f2out[0].weight.detach(),
            b1=block.f2out[0].bias.detach(),
            W2=block.f2out[1].weight.detach(),
            b2=block.f2out[1].bias.detach(),
        ))

    def forward(self, x, f_ij, idx_i, idx_j, rcut_ij):
        """Grad-enabled path (residual added to x, like SchNetInteraction... no:
        InteractionFn already returns x_out incl. residual)."""
        return InteractionFn.apply(x, f_ij, idx_i, idx_j, rcut_ij, *self.params)

    @torch.no_grad()
    def forward_nograd(self, x, f_ij, idx_i, idx_j, rcut_ij):
        """No-autograd path: returns (x_out, saved) for a manual bwd caller."""
        return interaction_fwd(x, f_ij, idx_i, idx_j, rcut_ij, self.params)
