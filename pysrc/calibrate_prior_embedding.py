"""stage 1 calibration: freeze the entire fx2-trained 6M transformer backbone
and train only the prior_embedding module (CastedLinear, vocabulary_size=205
-> d_model=192) on fx4's own (article order, PPMD::Predict() priors) data, to
see whether that alone recovers some of the BPB fx4's real pipeline loses
relative to fx2's claimed result (see hutter_run_log.md's fx2-vs-fx4
divergence table: identical weights and dictionary, but a 98.15%-different
article order past line 3184 and a different PPMD::Predict() implementation
feeding the frozen transformer a distribution it wasn't trained on).

data must already be split into disjoint train/val article sets with
pysrc.split_trace (run from fx4-cmix/):
    python -m pysrc.split_trace data data/train data/val
    python -m pysrc.calibrate_prior_embedding
"""

import argparse
import math
from dataclasses import replace

import torch

from pysrc.data import CollatedArticlesDataLoader
from pysrc.load_frozen_checkpoint import load_frozen_model, make_checkpoint_config


def make_data_loader(
    prefix: str, epochs: int, micro_batch_tokens: int, max_article_tokens: int,
    device: torch.device,
) -> CollatedArticlesDataLoader:
    return CollatedArticlesDataLoader(
        tokens_filename=f"{prefix}.tokens.uint8",
        priors_filename=f"{prefix}.priors.float16",
        article_boundaries_filename=f"{prefix}.boundaries.int32",
        mmap=True,
        epochs=epochs,
        micro_batch_tokens=micro_batch_tokens,
        gradient_accumulation_steps=1,
        max_article_tokens=max_article_tokens,
        max_merge_tokens=None,
        device=device,
        rank=0,
        world_size=1,
    )


@torch.no_grad()
def evaluate_bpb(model, data: CollatedArticlesDataLoader, desc: str) -> float:
    model.eval()
    total_nats = 0.0
    total_tokens = 0
    for _articles, micro_batch in data.iter_ordered_micro_batches():
        loss = model(
            input_tokens=micro_batch.input_tokens,
            priors=micro_batch.priors,
            loss_mask=micro_batch.loss_mask,
            output_tokens=micro_batch.output_tokens,
            cumulative_sequence_lengths=micro_batch.cumulative_sequence_lengths,
        )
        total_nats += loss.item()
        total_tokens += int(micro_batch.loss_mask.sum().item())
    bpb = total_nats / total_tokens / math.log(2)
    print(f"{desc}: {total_tokens} tokens, BPB = {bpb:.5f}")
    return bpb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="models/transformer6m/6m-q4-fp32.tfwc2"
    )
    parser.add_argument("--train-prefix", default="data/train")
    parser.add_argument("--val-prefix", default="data/val")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--micro-batch-tokens", type=int, default=2048)
    parser.add_argument("--max-article-tokens", type=int, default=2048)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--output", default="models/6m-prior-embedding-calibrated.tch"
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="a previously-saved --output checkpoint (full model state_dict, "
        "frozen backbone + calibrated prior_embedding) to continue training "
        "from, instead of the original frozen checkpoint's untrained state",
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
    print(f"loaded frozen checkpoint from {args.checkpoint}")

    if args.resume_from is not None:
        model.load_state_dict(torch.load(args.resume_from, map_location=device))
        print(f"resumed calibrated prior_embedding weights from {args.resume_from}")

    for p in model.parameters():
        p.requires_grad_(False)
    trainable = list(model.prior_embedding.parameters())
    for p in trainable:
        p.requires_grad_(True)
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"stage 1: training {len(trainable)} tensors, {n_trainable:,} of "
        f"{n_total:,} parameters (prior_embedding only, backbone frozen)"
    )

    val_data = make_data_loader(
        args.val_prefix, epochs=1, micro_batch_tokens=args.micro_batch_tokens,
        max_article_tokens=args.max_article_tokens, device=device,
    )
    baseline_bpb = evaluate_bpb(model, val_data, "baseline (uncalibrated) val")

    train_data = make_data_loader(
        args.train_prefix, epochs=args.epochs,
        micro_batch_tokens=args.micro_batch_tokens,
        max_article_tokens=args.max_article_tokens, device=device,
    )
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
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
        f"done. baseline val BPB = {baseline_bpb:.5f}, "
        f"best val BPB = {best_val_bpb:.5f} "
        f"({'improved' if best_val_bpb < baseline_bpb else 'no improvement'} "
        f"by {abs(baseline_bpb - best_val_bpb):.5f})"
    )


if __name__ == "__main__":
    main()
