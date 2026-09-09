"""splits the fx4 PPMD trace (pysrc/data.py's CollatedArticlesDataLoader
format: raw uint8 tokens, raw float16 priors n_tokens x vocabulary_size, raw
int32 article start offsets) into disjoint train/val article sets, so
calibration can measure BPB on articles never seen during training.

every 10th article (by index in the trace, which is fx4's own article-
processing order) goes to validation and the rest to train, interleaving
instead of a single contiguous tail split -- our 10,000,000-position trace
only covers the first 4,938 articles of fx4's 172,277-article order, which
mixes the ~3,183 articles whose order matches fx2's training data with the
first ~1,755 diverged ones (see hutter_run_log.md), so a contiguous split
would risk one side being almost entirely one regime.

usage (from fx4-cmix/):
    python -m pysrc.split_trace data data/train data/val
"""

import sys

import numpy as np


def split_trace(input_prefix: str, train_prefix: str, val_prefix: str) -> None:
    tokens = np.fromfile(f"{input_prefix}/ppmd-bytes.uint8", dtype=np.uint8)
    n_tokens = len(tokens)

    boundaries = np.fromfile(
        f"{input_prefix}/article-boundaries.int32", dtype=np.int32
    ).astype(np.int64)
    assert boundaries[0] == 0 and boundaries[-1] == n_tokens
    assert (np.diff(boundaries) > 0).all()
    n_articles = len(boundaries) - 1

    import os

    float16_size = np.dtype(np.float16).itemsize
    priors_bytes = f"{input_prefix}/ppmd-probs.float16"
    n_priors_floats = os.path.getsize(priors_bytes) // float16_size
    assert n_priors_floats % n_tokens == 0
    vocabulary_size = n_priors_floats // n_tokens
    priors = np.memmap(
        priors_bytes, dtype=np.float16, mode="r", shape=(n_tokens, vocabulary_size)
    )

    val_articles = set(range(0, n_articles, 10))
    print(
        f"{n_articles} articles, {n_tokens} tokens: "
        f"{len(val_articles)} val articles, {n_articles - len(val_articles)} train articles"
    )

    for prefix, wanted in ((train_prefix, False), (val_prefix, True)):
        tokens_out = open(f"{prefix}.tokens.uint8", "wb")
        priors_out = open(f"{prefix}.priors.float16", "wb")
        boundaries_out = open(f"{prefix}.boundaries.int32", "wb")

        position = 0
        out_boundaries = [0]
        for article in range(n_articles):
            if (article in val_articles) != wanted:
                continue
            start, end = int(boundaries[article]), int(boundaries[article + 1])
            tokens_out.write(tokens[start:end].tobytes())
            priors_out.write(np.array(priors[start:end]).tobytes())
            position += end - start
            out_boundaries.append(position)

        tokens_out.close()
        priors_out.close()
        np.array(out_boundaries, dtype=np.int32).tofile(boundaries_out)
        boundaries_out.close()
        print(f"wrote {prefix}.* : {position} tokens, {len(out_boundaries) - 1} articles")


if __name__ == "__main__":
    input_prefix = sys.argv[1] if len(sys.argv) > 1 else "data"
    train_prefix = sys.argv[2] if len(sys.argv) > 2 else "data/train"
    val_prefix = sys.argv[3] if len(sys.argv) > 3 else "data/val"
    split_trace(input_prefix, train_prefix, val_prefix)
