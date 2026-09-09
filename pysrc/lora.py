"""minimal LoRA wrapper for pysrc.model.CastedLinear, used to give calibration
extra capacity in the last few transformer blocks without fully unfreezing
them (see hutter_run_log.md's fx2-vs-fx4 divergence table and
calibrate_prior_embedding.py's stage 1, which plateaued at a ~0.5% relative
BPB improvement -- consistent with prior_embedding alone being too small an
adapter to fix a different PPMD::Predict() implementation and R1 stage).

CastedLinear.forward(x, initialize_scales) already quantizes its own weight
on the fly; LoRALinear leaves that untouched (self.base stays frozen and is
called exactly as before) and adds a small trainable low-rank term
x @ A.T @ B.T * (alpha / rank) on top, so the adapter starts as an exact
no-op (B initialized to zero) and only ever adds a *correction* to the
frozen, verified-correct base prediction.
"""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import Module, Parameter, init

from pysrc.model import CastedLinear, InitializeScales


class LoRALinear(Module):
    def __init__(self, base: CastedLinear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        d_out, d_in = self.base.weight.shape
        dtype = self.base.weight.dtype
        self.lora_A = Parameter(torch.empty(rank, d_in, dtype=dtype))
        self.lora_B = Parameter(torch.zeros(d_out, rank, dtype=dtype))
        init.kaiming_uniform_(self.lora_A, a=5**0.5)
        self.scaling = alpha / rank

    def forward(
        self, x: Tensor, initialize_scales: InitializeScales
    ) -> Tensor:
        base_out = self.base(x, initialize_scales=initialize_scales)
        lora_x = x.to(self.lora_A.dtype)
        lora_out = (lora_x @ self.lora_A.T) @ self.lora_B.T
        return base_out + self.scaling * lora_out.to(base_out.dtype)


@dataclass(frozen=True, slots=True)
class LoRATarget:
    module_path: str  # dotted path to the CastedLinear, e.g. "blocks.10.attention.query_projection"


# the CastedLinear projections present in both Attention and
# KimiLinearAttention (see pysrc/model.py); skips the LowRankCastedLinear
# forget_gate_projection/output_gate_projection, beta_projection (raw fp32,
# not a CastedLinear) and the depthwise convolutions, to keep stage 2's scope
# to the main projections
_ATTENTION_PROJECTIONS = (
    "query_projection",
    "key_projection",
    "value_projection",
    "output_projection",
)
_MLP_PROJECTIONS = ("up", "down")


def wrap_lora_last_blocks(
    model: Module, n_blocks: int, rank: int, alpha: float
) -> list[Parameter]:
    """replaces the attention projections and MLP up/down of the last
    n_blocks transformer blocks with LoRALinear wrappers in place (mutates
    model.blocks), returning the newly created trainable LoRA parameters.
    every other parameter must already be frozen by the caller."""
    trainable: list[Parameter] = []
    n_layers = len(model.blocks)
    targets = range(n_layers - n_blocks, n_layers)

    for layer in targets:
        block = model.blocks[layer]

        attention = block.attention
        for name in _ATTENTION_PROJECTIONS:
            if not hasattr(attention, name):
                continue
            wrapped = LoRALinear(getattr(attention, name), rank=rank, alpha=alpha)
            setattr(attention, name, wrapped)
            trainable += [wrapped.lora_A, wrapped.lora_B]

        mlp = block.mlp
        for name in _MLP_PROJECTIONS:
            wrapped = LoRALinear(getattr(mlp, name), rank=rank, alpha=alpha)
            setattr(mlp, name, wrapped)
            trainable += [wrapped.lora_A, wrapped.lora_B]

    return trainable
