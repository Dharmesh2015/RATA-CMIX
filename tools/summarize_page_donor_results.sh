#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 && "$#" -ne 5 ]]; then
  echo "usage: $0 PAGE_RECIPIENTS_CSV WINNER_SELECTED_CSV [WINNER_TRIALS_CSV] OUT_DIR STOP_MIB" >&2
  exit 2
fi

readonly recipients="$1"
readonly selected="$2"
if [[ "$#" -eq 5 ]]; then
  readonly trials="$3"
  readonly out_dir="$4"
  readonly stop_mib="$5"
else
  readonly trials="${selected%.winner_selected.csv}.winner_trials.csv"
  readonly out_dir="$3"
  readonly stop_mib="$4"
fi
mkdir -p "$out_dir"

awk -F, 'BEGIN {
    OFS=",";
    print "post_r1_mib,pages,winner_pages,baseline_bytes,selected_bytes,net_gain_bytes"
  }
  NR > 1 {
    mib = int($2 / 1048576);
    pages[mib]++;
    winners[mib] += ($6 > 0);
    baseline[mib] += $3;
    selected[mib] += $4;
    net[mib] += $6;
  }
  END {
    for (mib = 0; mib < stop; ++mib)
      print mib,pages[mib]+0,winners[mib]+0,baseline[mib]+0,
            selected[mib]+0,net[mib]+0;
  }' stop="$stop_mib" "$selected" >"$out_dir/region_summary.csv"

awk -F, 'BEGIN { OFS="," }
  NR==FNR { if (FNR>1) name[$1]=$9; next }
  FNR==1 {
    print "class_name,pages,winner_pages,baseline_bytes,selected_bytes,net_gain_bytes";
    next
  }
  {
    c=name[$1];
    pages[c]++;
    winners[c]+=($6>0);
    baseline[c]+=$3;
    selected[c]+=$4;
    net[c]+=$6
  }
  END {
    for(c in pages)
      print c,pages[c],winners[c],baseline[c],selected[c],net[c]
  }' "$recipients" "$selected" | sort >"$out_dir/class_summary.csv"

if [[ ! -s "$trials" ]]; then
  echo "trial ledger not found: $trials" >&2
  exit 1
fi

awk -F, 'BEGIN {
    OFS=",";
    print "post_r1_mib,pages_tested,winner_pages,tie_pages,loss_pages," \
          "gross_gain_bytes,gross_loss_bytes,raw_net_bytes"
  }
  NR > 1 {
    mib = int($2 / 1048576);
    pages[mib]++;
    gain = $10 + 0;
    if (gain > 0) {
      winners[mib]++;
      gross_gain[mib] += gain;
    } else if (gain < 0) {
      losses[mib]++;
      gross_loss[mib] -= gain;
    } else {
      ties[mib]++;
    }
    net[mib] += gain;
  }
  END {
    for (mib = 0; mib < stop; ++mib)
      print mib,pages[mib]+0,winners[mib]+0,ties[mib]+0,losses[mib]+0,
            gross_gain[mib]+0,gross_loss[mib]+0,net[mib]+0;
  }' stop="$stop_mib" "$trials" >"$out_dir/raw_trial_region_summary.csv"

# Per-page evidence: every trial that beat baseline (already produced by the
# search as page.winner_positive.csv) versus every trial that lost to
# baseline, kept as two separate ledgers so a specific winning or losing
# page/offset can be looked up directly instead of only an aggregate.
head -n 1 "$trials" >"$out_dir/winner_trials_positive.csv"
awk -F, 'NR>1 && ($10+0)>0' "$trials" >>"$out_dir/winner_trials_positive.csv"
head -n 1 "$trials" >"$out_dir/winner_trials_negative.csv"
awk -F, 'NR>1 && ($10+0)<0' "$trials" >>"$out_dir/winner_trials_negative.csv"

awk -F, 'BEGIN { OFS="," }
  NR==FNR { if (FNR>1) name[$1]=$9; next }
  FNR==1 {
    print "class_name,pages_tested,winner_pages,tie_pages,loss_pages," \
          "gross_gain_bytes,gross_loss_bytes,raw_net_bytes";
    next
  }
  {
    c=name[$1];
    pages[c]++;
    gain=$10+0;
    if (gain>0) {
      winners[c]++;
      gross_gain[c]+=gain;
    } else if (gain<0) {
      losses[c]++;
      gross_loss[c]-=gain;
    } else {
      ties[c]++;
    }
    net[c]+=gain;
  }
  END {
    for(c in pages)
      print c,pages[c],winners[c],ties[c],losses[c],gross_gain[c],
            gross_loss[c],net[c]
  }' "$recipients" "$trials" | sort >"$out_dir/raw_trial_class_summary.csv"
