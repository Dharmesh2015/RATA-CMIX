#!/usr/bin/env python3
"""Bounded donor/SCR2 crawler seeded by the historical 98,090 result.

The crawler is deliberately diagnostic. It tests canonical post-R1 1 MiB
regions from a dictionary-pretrained cold state, keeps every result, and emits
an F4CP-v3 candidate production plan containing only causal net winners. The
plan still needs one exact continuous full-prefix validation before release.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import mmap
import os
import shutil
import struct
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CHUNK_SIZE = 1 << 20
CANONICAL_SIZE = 587_138_826
CANONICAL_SHA256 = (
    "7826ff63dedd526c119dda08e6e044be8"
    "fa8f6e89a55f3d6b1f3447cdfc5c1ce"
)
LEGACY_REGION = 421
LEGACY_RAW_SHA256 = (
    "617d34c4987f60cad755c87f34d160dd"
    "83f5b490f466e00c9a0a01399213afb4"
)
LEGACY_BASELINE = 99_343
LEGACY_DONOR_ARCHIVE = 98_090
TRIAL_FIELDS = [
    "recipient_region",
    "recipient_offset",
    "mode",
    "profile_id",
    "donors",
    "archive_bytes",
    "side_bytes",
    "total_bytes",
    "bpb",
    "wall_seconds",
    "status",
]
DONOR_LENGTHS = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)


@dataclass(frozen=True, order=True)
class Window:
    offset: int
    length: int
    seed_gain: int = 0


@dataclass(frozen=True)
class Profile:
    windows: tuple[Window, ...]

    @property
    def id(self) -> str:
        if not self.windows:
            return "none"
        text = ";".join(f"{w.offset}:{w.length}" for w in self.windows)
        return hashlib.sha256(text.encode("ascii")).hexdigest()[:16]

    @property
    def text(self) -> str:
        return ";".join(f"{w.offset}:{w.length}" for w in self.windows) or "-"


@dataclass
class Trial:
    region: int
    mode: str
    profile: Profile
    archive_bytes: int
    side_bytes: int
    total_bytes: int
    bpb: float
    wall_seconds: float
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--cmix", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--seed-csv", type=Path, required=True)
    parser.add_argument("--proxy-csv", type=Path)
    parser.add_argument("--scr2-tool", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=46.0)
    parser.add_argument("--max-regions", type=int, default=48)
    parser.add_argument("--max-profiles-per-region", type=int, default=4)
    parser.add_argument("--profile-bank-size", type=int, default=8)
    parser.add_argument("--proxy-per-recipient", type=int, default=2)
    parser.add_argument("--trial-timeout", type=int, default=1800)
    parser.add_argument("--cpu", type=int, default=7)
    parser.add_argument("--start-region", type=int, default=0)
    parser.add_argument("--stop-region", type=int, default=0)
    parser.add_argument("--no-scr2", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_seed(path: Path) -> Profile:
    rows: list[tuple[int, Window]] = []
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if int(row["recipient_region"]) != LEGACY_REGION:
                continue
            rows.append(
                (
                    int(row["replay_order"]),
                    Window(
                        int(row["donor_offset"]),
                        int(row["donor_length"]),
                        int(row.get("gain_bytes") or 0),
                    ),
                )
            )
    rows.sort()
    profile = Profile(tuple(window for _order, window in rows))
    if len(profile.windows) != 7 or sum(w.length for w in profile.windows) != 22_528:
        raise SystemExit("seed CSV is not the exact seven-window 22,528-byte winner")
    return profile


def load_proxy(path: Path | None) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    if path is None:
        return result
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            region = int(row["recipient_region"])
            name = "donor_seed_offset" if "donor_seed_offset" in row else "donor_offset"
            result.setdefault(region, []).append(int(row[name]))
    return result


def varint_size(value: int) -> int:
    size = 1
    while value >= 128:
        value >>= 7
        size += 1
    return size


def block_signature(data: memoryview, start: int, end: int) -> int:
    best = 0xFFFFFFFF
    for local in range(start, min(end - 15, start + 256), 16):
        value = zlib.crc32(data[local : local + 16])
        if value < best:
            best = value
    return best


def window_signatures(data: memoryview, offset: int, length: int) -> set[int]:
    end = offset + length
    return {
        block_signature(data, local, min(local + 256, end))
        for local in range(offset, end - 15, 256)
    }


def profile_signatures(data: memoryview, profile: Profile) -> set[int]:
    result: set[int] = set()
    for window in profile.windows:
        result.update(window_signatures(data, window.offset, window.length))
    return result


def region_signatures(data: memoryview, region: int) -> set[int]:
    start = region * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, len(data))
    return {
        block_signature(data, local, min(local + 256, end))
        for local in range(start, end - 15, 256)
    }


def profile_score(signatures: set[int], target: set[int]) -> int:
    return len(signatures.intersection(target))


def rank_regions(
    data: memoryview,
    seed: Profile,
    start_region: int,
    stop_region: int,
) -> tuple[list[tuple[int, int]], dict[int, set[int]]]:
    seed_end = max(window.offset + window.length for window in seed.windows)
    first_causal = (seed_end + CHUNK_SIZE - 1) // CHUNK_SIZE
    complete = len(data) // CHUNK_SIZE
    first = max(first_causal, start_region)
    last = min(complete, stop_region or complete)
    seed_sig = profile_signatures(data, seed)
    signatures: dict[int, set[int]] = {}
    ranking: list[tuple[int, int]] = []
    for region in range(first, last):
        target = region_signatures(data, region)
        signatures[region] = target
        ranking.append((region, profile_score(seed_sig, target)))
    ranking.sort(key=lambda item: (-item[1], abs(item[0] - LEGACY_REGION), item[0]))
    for index, item in enumerate(ranking):
        if item[0] == LEGACY_REGION:
            ranking.insert(0, ranking.pop(index))
            break
    return ranking, signatures


def append_csv(path: Path, row: dict[str, object]) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=TRIAL_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        output.flush()
        os.fsync(output.fileno())


def load_trials(path: Path) -> dict[tuple[int, str, str], Trial]:
    trials: dict[tuple[int, str, str], Trial] = {}
    if not path.exists():
        return trials
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            windows: list[Window] = []
            if row["donors"] not in ("", "-"):
                for item in row["donors"].split(";"):
                    offset, length = item.split(":", 1)
                    windows.append(Window(int(offset), int(length)))
            trial = Trial(
                int(row["recipient_region"]),
                row["mode"],
                Profile(tuple(windows)),
                int(row["archive_bytes"] or 0),
                int(row["side_bytes"] or 0),
                int(row["total_bytes"] or 0),
                float(row["bpb"] or 0),
                float(row["wall_seconds"] or 0),
                row["status"],
            )
            trials[(trial.region, trial.mode, row["profile_id"])] = trial
    return trials


def write_bytes(path: Path, data: memoryview, start: int, length: int) -> None:
    temporary = path.with_suffix(path.suffix + ".new")
    with temporary.open("wb") as output:
        output.write(data[start : start + length])
    temporary.replace(path)


def write_bundle(path: Path, data: memoryview, profile: Profile) -> None:
    temporary = path.with_suffix(path.suffix + ".new")
    with temporary.open("wb") as output:
        for window in profile.windows:
            output.write(data[window.offset : window.offset + window.length])
    temporary.replace(path)


def run_process(command: list[str], cwd: Path, log_path: Path, timeout: int,
    environment: dict[str, str] | None = None) -> tuple[int, float]:
    started = time.monotonic()
    with log_path.open("wb") as log:
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            return result.returncode, time.monotonic() - started
        except subprocess.TimeoutExpired:
            return 124, time.monotonic() - started


def run_trial(
    args: argparse.Namespace,
    data: memoryview,
    trials: dict[tuple[int, str, str], Trial],
    ledger: Path,
    region: int,
    mode: str,
    profile: Profile,
    input_path: Path,
    side_bytes: int,
) -> Trial:
    key = (region, mode, profile.id)
    if key in trials:
        return trials[key]
    bundle = args.work_dir / f"profile_{profile.id}.bin"
    if profile.windows and not bundle.exists():
        write_bundle(bundle, data, profile)
    archive = args.work_dir / f"r{region}_{mode}_{profile.id}.fx4"
    log = args.work_dir / f"r{region}_{mode}_{profile.id}.log"
    archive.unlink(missing_ok=True)
    environment = os.environ.copy()
    if profile.windows:
        environment["FX4_RESEARCH_DONOR_BOOTSTRAP"] = str(bundle)
    else:
        environment.pop("FX4_RESEARCH_DONOR_BOOTSTRAP", None)
    command = [
        "taskset",
        "-c",
        str(args.cpu),
        str(args.cmix),
        "-n",
        str(args.dictionary),
        str(input_path),
        str(archive),
    ]
    status, elapsed = run_process(
        command, args.work_dir, log, args.trial_timeout, environment
    )
    archive_bytes = archive.stat().st_size if status == 0 and archive.exists() else 0
    total = archive_bytes + side_bytes if archive_bytes else 0
    trial = Trial(
        region,
        mode,
        profile,
        archive_bytes,
        side_bytes,
        total,
        total * 8.0 / CHUNK_SIZE if total else 0.0,
        elapsed,
        "ok" if status == 0 and archive_bytes else f"error_{status}",
    )
    append_csv(
        ledger,
        {
            "recipient_region": region,
            "recipient_offset": region * CHUNK_SIZE,
            "mode": mode,
            "profile_id": profile.id,
            "donors": profile.text,
            "archive_bytes": archive_bytes,
            "side_bytes": side_bytes,
            "total_bytes": total,
            "bpb": f"{trial.bpb:.9f}" if total else "",
            "wall_seconds": f"{elapsed:.2f}",
            "status": trial.status,
        },
    )
    trials[key] = trial
    archive.unlink(missing_ok=True)
    return trial


def prepare_scr2(
    args: argparse.Namespace, region: int, chunk: Path
) -> tuple[Path, int] | None:
    if args.no_scr2 or args.scr2_tool is None:
        return None
    prefix = args.work_dir / f"r{region}.scr2"
    transformed = prefix.with_suffix(".scr2.bin")
    metadata = prefix.with_suffix(".scr2.meta")
    if transformed.exists() and metadata.exists():
        return transformed, metadata.stat().st_size
    command = [
        sys.executable,
        str(args.scr2_tool),
        "run",
        str(chunk),
        "--prefix",
        str(prefix),
        "--max-macros",
        "128",
        "--min-total-saving",
        "1",
    ]
    status, _elapsed = run_process(
        command,
        args.work_dir,
        args.work_dir / f"r{region}.scr2.transform.log",
        min(args.trial_timeout, 900),
    )
    if status != 0 or not transformed.exists() or not metadata.exists():
        return None
    return transformed, metadata.stat().st_size


def unique_profiles(profiles: Iterable[Profile]) -> list[Profile]:
    result: list[Profile] = []
    seen: set[str] = set()
    for profile in profiles:
        if profile.id not in seen:
            seen.add(profile.id)
            result.append(profile)
    return result


def best_proxy_window(
    data: memoryview,
    target_sig: set[int],
    region: int,
    offsets: list[int],
    limit: int,
) -> list[Window]:
    result: list[tuple[int, Window]] = []
    recipient_start = region * CHUNK_SIZE
    for offset in offsets[: max(limit * 3, limit)]:
        for length in DONOR_LENGTHS:
            if offset & 255 or offset + length > recipient_start:
                continue
            window = Window(offset, length)
            score = profile_score(window_signatures(data, offset, length), target_sig)
            result.append((score, window))
    result.sort(key=lambda item: (-item[0], item[1].offset, item[1].length))
    selected: list[Window] = []
    seen_offsets: set[int] = set()
    for _score, window in result:
        if window.offset in seen_offsets:
            continue
        seen_offsets.add(window.offset)
        selected.append(window)
        if len(selected) >= limit:
            break
    return selected


def candidate_profiles(
    data: memoryview,
    region: int,
    target_sig: set[int],
    seed: Profile,
    bank: list[Profile],
    proxy_offsets: list[int],
    maximum: int,
    proxy_limit: int,
) -> list[Profile]:
    seed_sig = profile_signatures(data, seed)
    ranked_bank = sorted(
        bank,
        key=lambda profile: (
            -profile_score(profile_signatures(data, profile), target_sig),
            profile.id,
        ),
    )
    candidates: list[Profile] = [seed]
    candidates.extend(ranked_bank[:2])
    proxies = best_proxy_window(
        data, target_sig, region, proxy_offsets, proxy_limit
    )
    for proxy in proxies:
        candidates.append(Profile((proxy,)))
        for parent in ranked_bank[:2]:
            if len(parent.windows) < 8 and proxy not in parent.windows:
                candidates.append(Profile(parent.windows + (proxy,)))
            if parent.windows and proxy not in parent.windows:
                weakest = min(
                    range(len(parent.windows)),
                    key=lambda index: parent.windows[index].seed_gain,
                )
                replaced = list(parent.windows)
                replaced[weakest] = proxy
                candidates.append(Profile(tuple(replaced)))
    recipient_start = region * CHUNK_SIZE
    candidates = [
        profile
        for profile in unique_profiles(candidates)
        if all(window.offset + window.length <= recipient_start
               for window in profile.windows)
    ]
    candidates.sort(
        key=lambda profile: (
            0 if profile.id == seed.id else 1,
            -profile_score(profile_signatures(data, profile), target_sig),
            len(profile.windows),
            profile.id,
        )
    )
    return candidates[:maximum]


def write_rank(path: Path, ranking: list[tuple[int, int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["crawl_rank", "recipient_region", "recipient_offset", "seed_similarity"])
        for rank, (region, score) in enumerate(ranking):
            writer.writerow([rank, region, region * CHUNK_SIZE, score])


def load_crawl_order(path: Path, valid_regions: set[int]) -> list[int]:
    if not path.exists():
        return []
    result: list[int] = []
    seen: set[int] = set()
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            region = int(row["recipient_region"])
            if region in valid_regions and region not in seen:
                seen.add(region)
                result.append(region)
    return result


def append_crawl_order(
    path: Path, rank: int, region: int, score: int, profile_id: str
) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        if not exists:
            writer.writerow(
                [
                    "crawl_rank",
                    "recipient_region",
                    "recipient_offset",
                    "dynamic_similarity",
                    "source_profile_id",
                ]
            )
        writer.writerow([rank, region, region * CHUNK_SIZE, score, profile_id])
        output.flush()
        os.fsync(output.fileno())


def select_next_region(
    ranking: list[tuple[int, int]],
    signatures: dict[int, set[int]],
    selected: set[int],
    bank: list[Profile],
    profile_signature_cache: dict[str, set[int]],
) -> tuple[int, int, str] | None:
    rank_index = {region: index for index, (region, _score) in enumerate(ranking)}
    best: tuple[tuple[int, int, int], int, int, str] | None = None
    for region, seed_score in ranking:
        if region in selected:
            continue
        score = seed_score
        source_id = bank[0].id
        for profile in bank:
            candidate_score = profile_score(
                profile_signature_cache[profile.id], signatures[region]
            )
            if candidate_score > score:
                score = candidate_score
                source_id = profile.id
        key = (score, -rank_index[region], -abs(region - LEGACY_REGION))
        if best is None or key > best[0]:
            best = (key, region, score, source_id)
    if best is None:
        return None
    return best[1], best[2], best[3]

def write_outputs(
    args: argparse.Namespace,
    corpus_digest: str,
    seed: Profile,
    ranking: list[tuple[int, int]],
    trials: dict[tuple[int, str, str], Trial],
) -> None:
    baselines = {
        trial.region: trial
        for trial in trials.values()
        if trial.mode == "raw" and not trial.profile.windows and trial.status == "ok"
    }
    rows: list[tuple[int, Trial, Trial, Trial | None]] = []
    for region, baseline in sorted(baselines.items()):
        raw = [
            trial
            for trial in trials.values()
            if trial.region == region and trial.mode == "raw" and trial.status == "ok"
        ]
        scr2 = [
            trial
            for trial in trials.values()
            if trial.region == region and trial.mode == "scr2" and trial.status == "ok"
        ]
        winner = min(raw, key=lambda trial: trial.total_bytes)
        scr2_winner = min(scr2, key=lambda trial: trial.total_bytes) if scr2 else None
        rows.append((region, baseline, winner, scr2_winner))

    winners_path = args.results / "winners.csv"
    with winners_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "recipient_region",
                "baseline_bytes",
                "donor_bytes",
                "donor_gain_bytes",
                "donor_profile",
                "scr2_bytes",
                "scr2_gain_vs_donor",
                "selected_mode",
            ]
        )
        for region, baseline, winner, scr2_winner in rows:
            selected_mode = "raw"
            if winner.total_bytes < baseline.total_bytes:
                selected_mode = "donor"
            if scr2_winner and scr2_winner.total_bytes < winner.total_bytes:
                selected_mode = "scr2"
            writer.writerow(
                [
                    region,
                    baseline.total_bytes,
                    winner.total_bytes,
                    baseline.total_bytes - winner.total_bytes,
                    winner.profile.text,
                    scr2_winner.total_bytes if scr2_winner else "",
                    winner.total_bytes - scr2_winner.total_bytes if scr2_winner else "",
                    selected_mode,
                ]
            )

    assignments: list[tuple[int, Window, int]] = []
    selected_gain = 0
    for region, baseline, winner, _scr2 in rows:
        gain = baseline.total_bytes - winner.total_bytes
        metadata = 7 * len(winner.profile.windows)
        if not winner.profile.windows or gain <= metadata:
            continue
        if any(window.offset + window.length > region * CHUNK_SIZE
               for window in winner.profile.windows):
            continue
        selected_gain += gain
        for order, window in enumerate(winner.profile.windows):
            assignments.append((region, window, order))

    plan_path = args.results / "crawler_winners.f4cp"
    with plan_path.open("wb") as output:
        output.write(
            struct.pack(
                "<4sHHIIQH32s",
                b"F4CP",
                3,
                0,
                CHUNK_SIZE,
                max((window.length for _region, window, _order in assignments), default=0),
                args.corpus.stat().st_size,
                len(assignments),
                bytes.fromhex(corpus_digest),
            )
        )
        for region, window, order in assignments:
            output.write(struct.pack("<HIHB", region, window.offset, window.length, order))

    edge_path = args.results / "selected_edges.csv"
    with edge_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["recipient_region", "donor_offset", "donor_length", "replay_order"])
        for region, window, order in assignments:
            writer.writerow([region, window.offset, window.length, order])

    plan_cost = 3 + 7 * len(assignments)
    summary = {
        "format": "FX4 bounded donor/SCR2 crawler v1",
        "coordinate_domain": "canonical post-R1 predictor stream",
        "stream_bytes": args.corpus.stat().st_size,
        "stream_sha256": corpus_digest,
        "regions_ranked": len(ranking),
        "regions_completed": len(rows),
        "exact_trials_recorded": len(trials),
        "seed_profile_id": seed.id,
        "seed_windows": seed.text,
        "legacy_98090": {
            "raw_enwik9_region": LEGACY_REGION,
            "raw_region_sha256": LEGACY_RAW_SHA256,
            "baseline_bytes": LEGACY_BASELINE,
            "donor_archive_bytes": LEGACY_DONOR_ARCHIVE,
            "saving_bytes": LEGACY_BASELINE - LEGACY_DONOR_ARCHIVE,
            "warning": "calibration is raw-region isolated, not canonical post-R1 region 421",
        },
        "causal_plan_assignments": len(assignments),
        "causal_plan_recipient_regions": len({region for region, _window, _order in assignments}),
        "projected_gross_donor_gain_bytes": selected_gain,
        "archive_plan_bytes": plan_cost,
        "projected_net_donor_gain_bytes": selected_gain - plan_cost,
        "scr2_winner_regions": sum(
            1 for _region, _baseline, winner, scr2 in rows
            if scr2 and scr2.total_bytes < winner.total_bytes
        ),
        "validation": "isolated dictionary-pretrained screen; exact continuous full-prefix validation required",
    }
    (args.results / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    if (
        args.hours <= 0
        or args.max_regions <= 0
        or args.max_profiles_per_region <= 0
        or args.profile_bank_size <= 0
    ):
        raise SystemExit("time, region, profile, and bank limits must be positive")
    for path in (args.corpus, args.cmix, args.dictionary, args.seed_csv):
        if not path.is_file():
            raise SystemExit(f"missing required file: {path}")
    if args.corpus.stat().st_size != CANONICAL_SIZE:
        raise SystemExit(f"expected {CANONICAL_SIZE}-byte canonical post-R1 stream")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    corpus_digest = sha256_file(args.corpus)
    if corpus_digest != CANONICAL_SHA256:
        raise SystemExit(f"unexpected post-R1 SHA-256: {corpus_digest}")

    seed = load_seed(args.seed_csv)
    proxy = load_proxy(args.proxy_csv)
    ledger = args.results / "trials.csv"
    trials = load_trials(ledger)
    deadline = time.monotonic() + args.hours * 3600.0

    with args.corpus.open("rb") as source:
        mapped = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
        data = memoryview(mapped)
        try:
            ranking, signatures = rank_regions(
                data, seed, args.start_region, args.stop_region
            )
            write_rank(args.results / "region_ranking.csv", ranking)
            valid_regions = {region for region, _score in ranking}
            crawl_order_path = args.results / "crawl_order.csv"
            crawl_order = load_crawl_order(crawl_order_path, valid_regions)
            if not crawl_order and LEGACY_REGION in valid_regions:
                crawl_order.append(LEGACY_REGION)
                append_crawl_order(
                    crawl_order_path, 0, LEGACY_REGION,
                    dict(ranking)[LEGACY_REGION], seed.id
                )

            bank: list[Profile] = [seed]
            profile_signature_cache = {seed.id: profile_signatures(data, seed)}
            selected: set[int] = set()
            crawl_rank = 0
            while (
                len(selected) < args.max_regions
                and time.monotonic() < deadline
            ):
                if crawl_rank < len(crawl_order):
                    region = crawl_order[crawl_rank]
                    similarity = max(
                        profile_score(profile_signature_cache[profile.id], signatures[region])
                        for profile in bank
                    )
                else:
                    choice = select_next_region(
                        ranking,
                        signatures,
                        selected,
                        bank,
                        profile_signature_cache,
                    )
                    if choice is None:
                        break
                    region, similarity, source_profile_id = choice
                    crawl_order.append(region)
                    append_crawl_order(
                        crawl_order_path,
                        crawl_rank,
                        region,
                        similarity,
                        source_profile_id,
                    )
                if region in selected:
                    crawl_rank += 1
                    continue
                selected.add(region)

                chunk = args.work_dir / f"region_{region}.bin"
                if not chunk.exists() or chunk.stat().st_size != CHUNK_SIZE:
                    write_bytes(chunk, data, region * CHUNK_SIZE, CHUNK_SIZE)
                baseline = run_trial(
                    args, data, trials, ledger, region, "raw", Profile(()), chunk, 0
                )
                if baseline.status != "ok":
                    crawl_rank += 1
                    continue

                candidates = candidate_profiles(
                    data,
                    region,
                    signatures[region],
                    seed,
                    bank,
                    proxy.get(region, []),
                    args.max_profiles_per_region,
                    args.proxy_per_recipient,
                )
                raw_trials = [baseline]
                for profile in candidates:
                    if time.monotonic() >= deadline:
                        break
                    raw_trials.append(
                        run_trial(args, data, trials, ledger, region, "raw", profile, chunk, 0)
                    )
                valid_raw = [trial for trial in raw_trials if trial.status == "ok"]
                winner = min(valid_raw, key=lambda trial: trial.total_bytes)
                if (
                    winner.profile.windows
                    and winner.total_bytes + 7 * len(winner.profile.windows)
                    < baseline.total_bytes
                    and all(existing.id != winner.profile.id for existing in bank)
                    and args.profile_bank_size > 1
                ):
                    others = [profile for profile in bank if profile.id != seed.id]
                    others.append(winner.profile)
                    bank = [seed] + others[-(args.profile_bank_size - 1) :]
                    profile_signature_cache[winner.profile.id] = profile_signatures(
                        data, winner.profile
                    )

                if time.monotonic() < deadline:
                    scr2 = prepare_scr2(args, region, chunk)
                    if scr2 is not None:
                        transformed, metadata_size = scr2
                        run_trial(
                            args,
                            data,
                            trials,
                            ledger,
                            region,
                            "scr2",
                            winner.profile,
                            transformed,
                            metadata_size,
                        )
                write_outputs(args, corpus_digest, seed, ranking, trials)
                print(
                    f"rank={crawl_rank} region={region} similarity={similarity} "
                    f"baseline={baseline.total_bytes} best={winner.total_bytes} "
                    f"gain={baseline.total_bytes - winner.total_bytes} "
                    f"bank={len(bank)}",
                    flush=True,
                )
                crawl_rank += 1
        finally:
            data.release()
            mapped.close()

    write_outputs(args, corpus_digest, seed, ranking, trials)
    print((args.results / "summary.json").read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
