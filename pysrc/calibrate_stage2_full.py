"""stage 2 (full-rank variant): on top of stage 1's calibrated
prior_embedding, fully unfreeze the last N transformer blocks and the
unembedding head -- ordinary fine-tuning of the existing weights, not LoRA.
No new parameters are added: the exported model stays exactly ~5.9M
parameters / ~2.9MB after Q4 quantization, identical in shape to the
original checkpoint.

deep, already-well-trained blocks get a much lower learning rate than the
still-adapting prior_embedding/unembedding, since a high LR risks catastrophic
forgetting of FX2's original representations on our small (~9M token)
calibration trace -- the low LR is this approach's regularizer, in place of
LoRA's rank constraint.

data must already be split into disjoint train/val article sets with
pysrc.split_trace (run from fx4-cmix/):
    python -m pysrc.calibrate_stage2_full --resume-from models/6m-prior-embedding-calibrated.tch
"""

import argparse
import math
from dataclasses import replace

import torch

from pysrc.calibrate_prior_embedding import evaluate_bpb, make_data_loader
from pysrc.load_frozen_checkpoint import load_frozen_model, make_checkpoint_config


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
    parser.add_argument("--n-unfrozen-blocks", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--block-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--micro-batch-tokens", type=int, default=2048)
    parser.add_argument("--max-article-tokens", type=int, default=2048)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--output", default="models/6m-stage2-full-calibrated.tch"
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

    head_params = list(model.prior_embedding.parameters())
    head_params += list(model.unembedding.parameters())
    for p in head_params:
        p.requires_grad_(True)

    n_layers = len(model.blocks)
    unfrozen_layers = range(n_layers - args.n_unfrozen_blocks, n_layers)
    block_params: list = []
    for layer in unfrozen_layers:
        for p in model.blocks[layer].parameters():
            p.requires_grad_(True)
            block_params.append(p)

    n_head = sum(p.numel() for p in head_params)
    n_blocks = sum(p.numel() for p in block_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"stage 2 (full-rank): training prior_embedding + unembedding "
        f"({n_head:,} params) + blocks {list(unfrozen_layers)} "
        f"({n_blocks:,} params) = {n_head + n_blocks:,} of {n_total:,} total "
        f"({(n_head + n_blocks) / n_total:.1%}), no new parameters added"
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
            {"params": head_params, "lr": args.head_lr},
            {"params": block_params, "lr": args.block_lr},
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
