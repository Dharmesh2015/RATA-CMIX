"""stage 2 calibration: on top of stage 1's calibrated prior_embedding, add
rank-r LoRA adapters (frozen base weight + trainable low-rank correction) to
the query/key/value/output projections and MLP up/down of the last N
transformer blocks, and continue training prior_embedding alongside them.

this follows the original incremental plan (prior-only -> prior+head -> LoRA
last 2 layers -> LoRA last 4 layers -> residual calibrator, stopping when
validation BPB stops improving): stage 1 (prior_embedding alone, ~39.5K
trainable parameters) plateaued at val BPB 1.16086 from a baseline of
1.16688, a ~0.5% relative improvement -- consistent with prior_embedding
being too small an adapter to fix a different PPMD::Predict() implementation
and R1 stage (see hutter_run_log.md's fx2-vs-fx4 divergence table), so this
stage gives calibration capacity in the deeper layers too.

data must already be split into disjoint train/val article sets with
pysrc.split_trace (run from fx4-cmix/):
    python -m pysrc.calibrate_stage2_lora --resume-from models/6m-prior-embedding-calibrated.tch
"""

import argparse
import math
from dataclasses import replace

import torch

from pysrc.calibrate_prior_embedding import evaluate_bpb, make_data_loader
from pysrc.load_frozen_checkpoint import load_frozen_model, make_checkpoint_config
from pysrc.lora import wrap_lora_last_blocks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="models/transformer6m/6m-q4-fp32.tfwc2"
    )
    parser.add_argument(
        "--resume-from",
        default="models/6m-prior-embedding-calibrated.tch",
        help="stage 1's calibrated checkpoint to build on (full model "
        "state_dict: frozen backbone + calibrated prior_embedding)",
    )
    parser.add_argument("--train-prefix", default="data/train")
    parser.add_argument("--val-prefix", default="data/val")
    parser.add_argument("--n-lora-blocks", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora-lr", type=float, default=1e-3)
    parser.add_argument("--prior-embedding-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--micro-batch-tokens", type=int, default=2048)
    parser.add_argument("--max-article-tokens", type=int, default=2048)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--output", default="models/6m-stage2-lora-calibrated.tch"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    cfg = replace(
        make_checkpoint_config(),
        attention_implementation="sdpa",
        gradient_checkpointing=True,
    )
    assert cfg.prior_embedding

    model = load_frozen_model(cfg, args.checkpoint, device)
    model.load_state_dict(torch.load(args.resume_from, map_location=device))
    print(f"loaded frozen checkpoint, resumed from {args.resume_from}")

    for p in model.parameters():
        p.requires_grad_(False)
    prior_embedding_params = list(model.prior_embedding.parameters())
    for p in prior_embedding_params:
        p.requires_grad_(True)

    lora_params = wrap_lora_last_blocks(
        model, n_blocks=args.n_lora_blocks, rank=args.lora_rank, alpha=args.lora_alpha
    )
    model = model.to(device)

    n_prior = sum(p.numel() for p in prior_embedding_params)
    n_lora = sum(p.numel() for p in lora_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"stage 2: training prior_embedding ({n_prior:,} params) + "
        f"rank-{args.lora_rank} LoRA on last {args.n_lora_blocks} blocks "
        f"({n_lora:,} params) = {n_prior + n_lora:,} of {n_total:,} total"
    )

    val_data = make_data_loader(
        args.val_prefix, epochs=1, micro_batch_tokens=args.micro_batch_tokens,
        max_article_tokens=args.max_article_tokens, device=device,
    )
    baseline_bpb = evaluate_bpb(model, val_data, "baseline (stage 1 checkpoint) val")

    train_data = make_data_loader(
        args.train_prefix, epochs=args.epochs,
        micro_batch_tokens=args.micro_batch_tokens,
        max_article_tokens=args.max_article_tokens, device=device,
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": prior_embedding_params, "lr": args.prior_embedding_lr},
            {"params": lora_params, "lr": args.lora_lr},
        ],
        weight_decay=args.weight_decay,
    )

    best_val_bpb = baseline_bpb
    model.train()
    step = 0
    total_steps = train_data.approximate_length()
    for batch in train_data:
        for micro_batch in batch.micro_batches:
            loss = model(
                input_tokens=micro_batch.input_tokens,
                priors=micro_batch.priors,
                loss_mask=micro_batch.loss_mask,
                output_tokens=micro_batch.output_tokens,
                cumulative_sequence_lengths=micro_batch.cumulative_sequence_lengths,
            )
            n_tokens = int(micro_batch.loss_mask.sum().item())
            (loss / n_tokens).backward()
            step_bpb = loss.item() / n_tokens / math.log(2)
        optimizer.step()
        optimizer.zero_grad()
        step += 1

        if step % args.log_every == 0:
            print(
                f"step {step}/{total_steps}, epoch {batch.epoch:.3f}: "
                f"train micro-batch BPB = {step_bpb:.5f}"
            )

        if batch.is_last or step % args.eval_every == 0:
            val_bpb = evaluate_bpb(model, val_data, f"val after step {step}")
            model.train()
            if val_bpb < best_val_bpb:
                best_val_bpb = val_bpb
                torch.save(model.state_dict(), args.output)
                print(f"new best val BPB {val_bpb:.5f}, saved to {args.output}")

    print(
        f"done. stage 1 val BPB = {baseline_bpb:.5f}, "
        f"best stage 2 val BPB = {best_val_bpb:.5f} "
        f"({'improved' if best_val_bpb < baseline_bpb else 'no improvement'} "
        f"by {abs(baseline_bpb - best_val_bpb):.5f})"
    )


if __name__ == "__main__":
    main()
