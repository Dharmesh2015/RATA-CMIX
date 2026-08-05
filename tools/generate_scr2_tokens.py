#!/usr/bin/env python3
"""Generate the fixed SCR2 token table used by the C++ transform and model."""

import argparse
import struct
import zlib
from pathlib import Path


MAGIC = b"SCR2M1\x00"


def read_uvarint(data: bytes, position: int):
    value = 0
    shift = 0
    while True:
        if position >= len(data) or shift > 63:
            raise ValueError("invalid SCR2 varint")
        byte = data[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, position
        shift += 7


def read_metadata(path: Path):
    blob = path.read_bytes()
    if not blob.startswith(MAGIC) or len(blob) < len(MAGIC) + 8:
        raise ValueError("invalid SCR2 metadata")
    raw_size, expected_crc = struct.unpack_from("<II", blob, len(MAGIC))
    payload = zlib.decompress(blob[len(MAGIC) + 8 :])
    if len(payload) != raw_size or zlib.crc32(payload) & 0xFFFFFFFF != expected_crc:
        raise ValueError("invalid SCR2 metadata payload")
    version, marker, original_size = struct.unpack_from("<BBQ", payload, 0)
    if version != 1:
        raise ValueError(f"unsupported SCR2 metadata version {version}")
    position = struct.calcsize("<BBQ") + 32
    count, position = read_uvarint(payload, position)
    patterns = []
    for _ in range(count):
        length, position = read_uvarint(payload, position)
        end = position + length
        if end > len(payload):
            raise ValueError("truncated SCR2 pattern")
        patterns.append(payload[position:end])
        position = end
    if position != len(payload) or len(patterns) > 255:
        raise ValueError("invalid SCR2 pattern table")
    return marker, original_size, patterns


def fnv1a(data: bytes, seed: int = 2166136261) -> int:
    value = seed
    for byte in data:
        value ^= byte
        value = value * 16777619 & 0xFFFFFFFF
    return value


def token_class(pattern: bytes) -> int:
    stripped = pattern.lstrip(b" \t")
    if stripped.startswith(b"Q "):
        return 1
    if stripped.startswith(b"L"):
        return 2
    if stripped.startswith(b"MM"):
        return 3
    if stripped.startswith(b"*"):
        return 4
    if stripped.startswith((b"[@", b"[[")):
        return 5
    if stripped.startswith(b"<"):
        return 6
    if stripped.startswith((b"D86", b"D99")):
        return 7
    return 8


def word_hash(pattern: bytes) -> int:
    normalized = bytearray()
    for byte in pattern:
        if 65 <= byte <= 90:
            normalized.append(byte + 32)
        elif 97 <= byte <= 122 or 48 <= byte <= 57 or byte >= 128:
            normalized.append(byte)
        elif normalized and normalized[-1] != 32:
            normalized.append(32)
    return fnv1a(bytes(normalized[-96:]), 0x9E3779B9)


def generate(marker: int, original_size: int, patterns, source: Path) -> str:
    flat = b"".join(patterns)
    rows = ["  {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},"]
    offset = 0
    for pattern in patterns:
        rows.append(
            "  {%du, %du, %du, %du, %du, %du, %du, %du, %du},"
            % (
                fnv1a(pattern),
                fnv1a(pattern[:24], 0xA5A5A5A5),
                fnv1a(pattern[-24:], 0xC3C3C3C3),
                word_hash(pattern),
                offset,
                len(pattern),
                token_class(pattern),
                pattern[0],
                pattern[-1],
            )
        )
        offset += len(pattern)
    byte_rows = []
    for start in range(0, len(flat), 16):
        byte_rows.append("  " + ", ".join(f"0x{x:02x}" for x in flat[start:start + 16]) + ",")
    return f'''#ifndef SCR2_TOKENS_H
#define SCR2_TOKENS_H

// Generated from {source.name} by tools/generate_scr2_tokens.py.
#include <cstddef>
#include <cstdint>

namespace scr2 {{

struct TokenInfo {{
  std::uint32_t hash;
  std::uint32_t prefix_hash;
  std::uint32_t suffix_hash;
  std::uint32_t word_hash;
  std::uint16_t offset;
  std::uint16_t length;
  std::uint8_t cls;
  std::uint8_t first;
  std::uint8_t last;
}};

inline constexpr std::uint8_t kMarker = {marker}u;
inline constexpr std::uint16_t kTokenCount = {len(patterns)}u;
inline constexpr std::uint16_t kMaxPatternLength = {max(map(len, patterns))}u;
inline constexpr std::uint64_t kLearnedPostR1Size = {original_size}ull;
inline constexpr std::uint8_t kPatternBytes[{len(flat)}] = {{
{chr(10).join(byte_rows)}
}};
inline constexpr TokenInfo kTokens[kTokenCount + 1] = {{
{chr(10).join(rows)}
}};

inline const std::uint8_t* PatternData(unsigned int code) {{
  return kPatternBytes + kTokens[code].offset;
}}

}}  // namespace scr2

#endif
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("meta")
    parser.add_argument("output")
    args = parser.parse_args()
    source = Path(args.meta)
    marker, original_size, patterns = read_metadata(source)
    text = generate(marker, original_size, patterns, source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(text)
    print(f"generated {output}: {len(patterns)} tokens, {sum(map(len, patterns))} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
