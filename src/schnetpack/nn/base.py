from typing import Callable, Union, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_

from torch.nn.init import zeros_


__all__ = ["Dense"]


class _CallableAsModule(nn.Module):
    """Wrap a plain Python callable (e.g. F.silu, shifted_softplus) as a Module.

    TorchScript can only script attributes whose type it can infer; an
    `nn.Module` works, a bare callable doesn't. Wrapping at construction time
    keeps the Dense API permissive (`activation=F.silu`) while making the
    stored attribute uniformly typed for scripting.
    """

    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


def _as_module(act) -> nn.Module:
    if act is None:
        return nn.Identity()
    if isinstance(act, nn.Module):
        return act
    return _CallableAsModule(act)


class Dense(nn.Linear):
    r"""Fully connected linear layer with activation function.

    .. math::
       y = activation(x W^T + b)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Union[Callable, nn.Module] = None,
        weight_init: Callable = xavier_uniform_,
        bias_init: Callable = zeros_,
    ):
        """
        Args:
            in_features: number of input feature :math:`x`.
            out_features: number of output features :math:`y`.
            bias: If False, the layer will not adapt bias :math:`b`.
            activation: if None, no activation function is used. Plain callables
                (e.g. ``F.silu``) are wrapped as Modules so the layer is script-able.
            weight_init: weight initializer (used in ``reset_parameters`` only).
            bias_init: bias initializer (used in ``reset_parameters`` only).
        """
        # nn.Linear.__init__ calls reset_parameters, which needs these. Stash
        # them as private instance attrs (NOT script-visible: leading underscore
        # would still be discovered, so we delete after init below).
        self._weight_init = weight_init
        self._bias_init = bias_init
        super(Dense, self).__init__(in_features, out_features, bias)
        # Remove init refs — they're no longer needed and would block JIT scripting
        # because Python callables don't have an inferable TorchScript type.
        del self._weight_init
        del self._bias_init

        self.activation = _as_module(activation)

    def reset_parameters(self):
        # Called from nn.Linear.__init__ (and potentially externally on re-init).
        # Use stashed inits if present; otherwise fall back to defaults (this
        # branch matters for re-initialization of a Dense that was loaded from
        # a pickle, where _weight_init/_bias_init were stripped).
        weight_init = getattr(self, '_weight_init', xavier_uniform_)
        bias_init = getattr(self, '_bias_init', zeros_)
        weight_init(self.weight)
        if self.bias is not None:
            bias_init(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        y = F.linear(input, self.weight, self.bias)
        y = self.activation(y)
        return y
