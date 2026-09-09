"""loads the deployed C++ checkpoint (cpp_infer/data/weights.bin, or a
weights_compress.py .c1/.c2 container -- models/transformer6m/6m-q4-fp32.tfwc2
is a .c2) back into a pysrc.model.Transformer for continued training.

export_weights.py's forward direction is: trained fp32 .weight parameter ->
QuantizeFunction (per-row scale, round-half-even, clamp [-7,7]) -> int8 ints +
bf16 row scale, written as two entries ("<key>.q", "<key>.scale") per
quantized weight, everything else (activation scales, raw fp32 tensors)
written as-is. This module inverts that: dequantized = ints.float32 * scale
is assigned directly as the state_dict's ".weight" entry, together with the
same scale under "<prefix>.quantize_weight.scale". This is an exact fixed
point of the quantizer (round(dequantized / scale) == ints again, since ints
is already integral), so the reloaded model reproduces bit-identical forward
passes to the exported checkpoint before any further training moves it.

rope.* and config.* entries are dropped: they are not part of the model's
state_dict (rope tables are recomputed by pysrc.model.make_rope_args; config
is redundant with the TransformerConfig passed to Transformer()).
"""

import struct

import numpy as np
import torch
from torch import Tensor

from pysrc.export_weights import (
    DTYPE_BF16_BITS,
    DTYPE_FLOAT32,
    MAGIC as MAGIC_UNCOMPRESSED,
    read_tensor_file as read_tensor_file_uncompressed,
)
from pysrc.weights_compress import MAGIC_V1, MAGIC_V2, _read_compressed
from pysrc.model import MatmulQuantization, Transformer, TransformerConfig
from pysrc.quantization import Quantization

_NON_STATE_DICT_PREFIXES = ("rope.", "config.")


def make_checkpoint_config() -> TransformerConfig:
    """export_weights.make_reference_config(), plus query_key_norm_gain=False:
    that field was added to pysrc/model.py's TransformerConfig after
    export_weights.py was written (training_recipes/train_6m_v2.py, a newer
    recipe, does set it), so export_weights.make_reference_config() now raises
    a missing-argument TypeError. The deployed checkpoint itself has no
    query_norm_gain/key_norm_gain tensors (checked directly against
    models/transformer6m/6m-q4-fp32.tfwc2's tensor names), confirming it
    predates that feature too -- so False is the value this checkpoint was
    actually exported with, not a guess."""
    weight_quantization = Quantization(
        buckets=15,
        block_size=None,
        scale_dtype=torch.bfloat16,
        arithmetic_dtype=torch.float32,
        scale_init="scaled_max",
        epsilon=None,
    )
    activation_quantization = Quantization(
        buckets=256,
        block_size=None,
        scale_dtype=torch.bfloat16,
        arithmetic_dtype=torch.float32,
        scale_init="scaled_mean",
        epsilon=None,
    )
    matmul_quantization = MatmulQuantization(
        weight=weight_quantization, activation=activation_quantization
    )

    return TransformerConfig(
        vocabulary_size=205,
        prior_embedding=True,
        prior_logit_mixing=False,
        max_variable_sequence_length=None,
        n_layers=12,
        attention_weight_sharing_pattern=list(range(12)),
        mlp_weight_sharing_pattern=list(range(12)),
        d_model=192,
        kimi_linear=[True, True, True, False] * 3,
        window_size=1024,
        window_size_multipliers=None,
        n_query_heads=3,
        n_key_value_heads=3,
        d_head=64,
        d_mla=None,
        rope_base=10_000,
        half_truncate_rope=False,
        query_key_norm=True,
        query_key_norm_gain=False,
        kimi_linear_d_head=64,
        kimi_linear_n_heads=3,
        kimi_linear_convolution_size=4,
        value_embedding_pattern=None,
        prior_value_embedding=False,
        attention_scale=None,
        d_mlp=768,
        activation_function="relu_squared",
        embedding_norm=True,
        prior_embedding_norm=True,
        skip_connections=True,
        token_embedding_connections=True,
        prior_embedding_connections=False,
        dropout=0.0,
        logit_softcap=15.0,
        prior_logprob_cap=-12.0,
        prior_embedding_on_logprobs=False,
        gradient_checkpointing=False,
        embedding_init="normal",
        prior_embedding_init="normal",
        up_init="kaiming_uniform",
        down_init="kaiming_uniform",
        unembedding_init="kaiming_uniform",
        convolution_init="kaiming_uniform",
        logit_mixing_head_init="zero",
        activation_dtype=torch.float32,
        attention_dtype=torch.float32,
        logit_dtype=torch.float32,
        weight_dtype=torch.float32,
        mlp_quantization=matmul_quantization,
        attention_full_rank_quantization=matmul_quantization,
        attention_low_rank_quantization=matmul_quantization,
        qkv_quantization=activation_quantization,
        kimi_linear_full_rank_quantization=matmul_quantization,
        kimi_linear_low_rank_quantization=matmul_quantization,
        kimi_linear_beta_projection_quantization=None,
        kimi_linear_convolution_quantization=None,
        embedding_quantization=weight_quantization,
        prior_embedding_quantization=matmul_quantization,
        unembedding_quantization=matmul_quantization,
        logit_mixing_head_quantization=None,
        adamw_quantization_scales=True,
        attention_implementation="flex_attention",
    )


def _bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    assert bits.dtype == np.uint16
    return (
        torch.from_numpy(bits.astype(np.uint16))
        .view(torch.bfloat16)
        .to(torch.float32)
        .numpy()
    )


def read_checkpoint_tensors(filename: str) -> dict[str, tuple[int, np.ndarray]]:
    """dispatches on the 8-byte magic to whichever of export_weights.py's
    uncompressed format or weights_compress.py's v1/v2 compressed formats the
    file is actually in"""
    with open(filename, "rb") as f:
        magic = f.read(8)
    if magic == MAGIC_UNCOMPRESSED:
        return read_tensor_file_uncompressed(filename)
    if magic in (MAGIC_V1, MAGIC_V2):
        return _read_compressed(filename)
    raise ValueError(f"{filename}: unrecognized magic {magic!r}")


def load_frozen_state_dict(filename: str) -> dict[str, Tensor]:
    tensors = read_checkpoint_tensors(filename)

    state_dict: dict[str, Tensor] = {}
    quantized_bases = {name[: -2] for name in tensors if name.endswith(".q")}

    for name, (dtype_code, array) in tensors.items():
        if name.startswith(_NON_STATE_DICT_PREFIXES):
            continue
        if name.endswith(".scale") and name[: -len(".scale")] in quantized_bases:
            continue  # consumed together with its ".q" pair below

        if name.endswith(".q"):
            base = name[: -2]  # "...weight"
            assert base.endswith(".weight"), base
            scale_dtype_code, scale_bits = tensors[base + ".scale"]
            assert scale_dtype_code == DTYPE_BF16_BITS, base
            scale = _bf16_bits_to_float32(scale_bits)  # (d_out,)
            assert scale.ndim == 1 and scale.shape[0] == array.shape[0], base

            dequantized = array.astype(np.float32) * scale[:, None]
            state_dict[base] = torch.from_numpy(dequantized)

            prefix = base[: -len(".weight")]
            state_dict[prefix + ".quantize_weight.scale"] = torch.from_numpy(scale)
            continue

        if dtype_code == DTYPE_BF16_BITS:
            state_dict[name] = torch.from_numpy(_bf16_bits_to_float32(array))
        elif dtype_code == DTYPE_FLOAT32:
            state_dict[name] = torch.from_numpy(array.astype(np.float32))
        else:
            raise ValueError(f"unexpected dtype_code {dtype_code} for {name}")

    return state_dict


def load_frozen_model(
    cfg: TransformerConfig, filename: str, device: torch.device
) -> Transformer:
    model = Transformer(cfg)
    state_dict = load_frozen_state_dict(filename)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert not unexpected, f"unexpected keys not in model: {unexpected}"
    assert not missing, f"model keys missing from checkpoint: {missing}"
    model = model.to(device)
    return model


def _verify_roundtrip(cfg: TransformerConfig, filename: str, device: torch.device) -> None:
    """loads the checkpoint, then re-quantizes every loaded weight with the
    model's own Quantize modules and checks the ints match the file exactly --
    the strongest available check that load_frozen_state_dict's dequantization
    was inverted correctly, without needing the original models/*.tch file"""
    from pysrc.model import InitializeScales

    model = load_frozen_model(cfg, filename, device)
    model.eval()

    tensors = read_checkpoint_tensors(filename)
    modules = dict(model.named_modules())
    n_checked = 0
    with torch.inference_mode():
        for name, (dtype_code, array) in tensors.items():
            if not name.endswith(".q"):
                continue
            base = name[:-2]
            prefix = base[: -len(".weight")]
            quantize_module = modules[prefix].quantize_weight
            weight = dict(model.named_parameters())[base]
            ints = quantize_module(
                weight, initialize_scales_steps=None, return_ints=True
            )
            expected = torch.from_numpy(array.astype(np.int64)).to(ints.device)
            assert torch.equal(ints, expected), f"requantization mismatch for {base}"
            n_checked += 1
    print(f"round-trip verified: {n_checked} quantized weights re-quantize to "
          f"exactly the ints stored in {filename}")


if __name__ == "__main__":
    import sys
    from dataclasses import replace

    filename = sys.argv[1] if len(sys.argv) > 1 else (
        "models/transformer6m/6m-q4-fp32.tfwc2"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = replace(make_checkpoint_config(), attention_implementation="sdpa")
    _verify_roundtrip(cfg, filename, device)
