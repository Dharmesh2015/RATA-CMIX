#!/usr/bin/env bash
# Aggregate exact leave-one-out donor evidence. A donor receives credit only
# for the bytes lost when that donor is removed from a winning ordered profile.
# This avoids attributing an entire multi-donor win to every donor in it.
#
# usage:
#   find_multi_recipient_donors.sh WINNER_MARGINALS_CSV \
#       [TOTAL_RECIPIENTS] [MIN_RECIPIENTS]
#
# `explicit_net_upper_bound` subtracts a five-byte one-time donor registration
# and the ideal entropy of that donor's activation bitmap. It excludes archive
# framing and donor-ID interactions, so it is deliberately optimistic.
# `causal_no_id_upper_bound` assumes a decoder-visible selector reproduces all
# profitable activations with no recipient metadata. It is a target for the
# causal probability gate, not a measured archive saving.

set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 3 ]]; then
  echo "usage: $0 WINNER_MARGINALS_CSV [TOTAL_RECIPIENTS] [MIN_RECIPIENTS]" >&2
  exit 2
fi

readonly marginals="$1"
readonly total_recipients_arg="${2:-0}"
readonly min_recipients="${3:-2}"
readonly expected_header='recipient_region,recipient_offset,profile_donors,removed_donor,profile_donor_count,full_payload_bytes,without_payload_bytes,marginal_gain_bytes,status'

test -f "$marginals"
[[ "$(head -n 1 "$marginals")" == "$expected_header" ]]
[[ "$total_recipients_arg" =~ ^[0-9]+$ ]]
[[ "$min_recipients" =~ ^[1-9][0-9]*$ ]]

total_recipients="$total_recipients_arg"
if [[ "$total_recipients" == 0 ]]; then
  total_recipients="$(awk -F, 'NR>1 && $1+1>n {n=$1+1} END {print n+0}' "$marginals")"
fi
if [[ "$total_recipients" == 0 ]]; then
  echo "marginal ledger has no recipient rows" >&2
  exit 1
fi

printf '%s\n' 'donor_offset_length,recipients_helped,total_exact_marginal,registration_bytes,ideal_activation_bytes,explicit_net_upper_bound,causal_no_id_upper_bound,break_even_bytes_per_recipient'
awk -F, -v total="$total_recipients" -v minimum="$min_recipients" '
  function ceil_value(x) {
    return x == int(x) ? int(x) : int(x) + 1
  }
  NR > 1 && $9 == "helps" && $8 > 0 {
    donor = $4
    pair = donor SUBSEP $1
    gain_value = $8 + 0
    if (!(pair in best) || gain_value > best[pair]) {
      if (!(pair in best)) recipients[donor]++
      gains[donor] += gain_value - best[pair]
      best[pair] = gain_value
    }
  }
  END {
    log2 = log(2)
    registration = 5
    for (donor in recipients) {
      count = recipients[donor]
      if (count < minimum) continue
      p = count / total
      activation_bits = 0
      if (p > 0 && p < 1) {
        activation_bits = -total * (p * log(p) / log2 + (1 - p) * log(1 - p) / log2)
      }
      activation_bytes = ceil_value(activation_bits / 8)
      explicit_net = gains[donor] - registration - activation_bytes
      causal_upper = gains[donor] - registration
      break_even = (registration + activation_bytes) / count
      printf "%s,%d,%d,%d,%d,%d,%d,%.3f\n", donor, count,
          gains[donor], registration, activation_bytes, explicit_net,
          causal_upper, break_even
    }
  }
' "$marginals" | sort -t, -k7,7nr -k3,3nr