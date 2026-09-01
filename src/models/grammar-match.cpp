#include "grammar-match.h"

#include <stdio.h>  // GM-CENSUS fprintf (transitively present on libc++, NOT libstdc++)

namespace {
#ifdef GM_REVTS
// Non-negative decimal image (no leading zeros), returns digit count.
int GmDecEnc(long long v, char* buf) {
  if (v < 0) v = 0;
  if (v == 0) { buf[0] = '0'; return 1; }
  char tmp[24];
  int n = 0;
  while (v > 0 && n < 20) { tmp[n++] = (char)('0' + (int)(v % 10)); v /= 10; }
  for (int i = 0; i < n; ++i) buf[i] = tmp[n - 1 - i];
  return n;
}
#endif
}  // namespace

GrammarMatch::GrammarMatch(const unsigned int& bit_context, int limit,
    float delta) : bit_context_(bit_context), limit_(limit), delta_(delta),
    divisor_(1.0 / (limit + delta)) {
  // Flat 0.5 init: the channel is born silent everywhere and only speaks
  // once a state's confidence has moved off neutral (unlike Match's ramped
  // init, which encodes a match-length prior we do not have).
  predictions_.fill(0.5f);
  counts_.fill(0);
#ifdef FX4_GRAMMAR_CENSUS
  census_arms_.fill(0);
  census_hits_.fill(0);
#endif
  tbuf_.fill(0);
  title_.fill(0);
  fail_.fill(0);
  tgt_.fill(0);
}

GrammarMatch::~GrammarMatch() {
#ifdef FX4_GRAMMAR_CENSUS
  // GM-CENSUS (stderr-only): true per-state armed-bit / matched-bit totals
  // plus the learned confidence — every full run doubles as a per-family
  // contribution census at scale. Families decode via the STEP-0 base table.
  unsigned long long ta = 0, th = 0;
  for (int s = 0; s < 256; ++s) { ta += census_arms_[s]; th += census_hits_[s]; }
  fprintf(stderr, "GM-CENSUS total armed-bits=%llu matched=%llu\n", ta, th);
  for (int s = 0; s < 256; ++s) {
    if (census_arms_[s] > 0) {
      fprintf(stderr, "GM-CENSUS state=%d arms=%llu hits=%llu conf=%.4f\n",
          s, census_arms_[s], census_hits_[s], (double)predictions_[s]);
    }
  }
#endif
}


const std::valarray<float>& GrammarMatch::Predict() const {
  if (!have_expectation_) {
    outputs_[0] = 0.5f;  // exact: SetInput stretches this to 0
    return outputs_;
  }
  if (expected_byte_ & bit_pos_) outputs_[0] = predictions_[state_];
  else outputs_[0] = 1 - predictions_[state_];
  return outputs_;
}

void GrammarMatch::Perceive(int bit) {
  if (!have_expectation_) return;  // idle is silence: no table update either
  int match = 0;
  if (bit == ((expected_byte_ & bit_pos_) != 0)) match = 1;
  bit_pos_ /= 2;
#ifdef FX4_GRAMMAR_CENSUS
  ++census_arms_[state_];
  census_hits_[state_] += match;
#endif

  float divisor = divisor_;
  if (counts_[state_] < limit_) {
    ++counts_[state_];
    divisor = 1.0 / (counts_[state_] + delta_);
  }
  predictions_[state_] += (match - predictions_[state_]) * divisor;

  if (!match) {
    // The grammar's byte is wrong at this site: go soft (silent) for the
    // rest of the byte; ByteUpdate sees the actual byte and re-derives the
    // byte-level state (divergence -> scan/idle) at the boundary.
    have_expectation_ = false;
  }
}

void GrammarMatch::ByteUpdate() {
  ParseByte(static_cast<unsigned char>(bit_context_));
  bit_pos_ = 128;
}

void GrammarMatch::ParseByte(unsigned char c) {
  recent_ = (recent_ << 8) | c;

  // --- title capture ---------------------------------------------------
  if (!in_title_) {
    if ((recent_ & 0xFFFFFFFFULL) == kTitleOpen) {
      // A new page begins: the previous title (and its echoes) are stale.
      in_title_ = true;
      tbuf_len_ = 0;
      title_len_ = 0;
      echo_k_ = 0;
    }
  } else {
    if ((recent_ & 0xFFFFFFFFFFULL) == kTitleClose) {
      FinishTitle();
    } else if (tbuf_len_ >= kTitleCap) {
      in_title_ = false;  // oversize title: no echo family this article
      tbuf_len_ = 0;
    } else {
      tbuf_[tbuf_len_++] = c;
    }
  }

  BracketAdvance(c);
  if (!in_title_ && title_len_ > 0) EchoAdvance(c);
#ifdef GM_REVTS
  ++pos_;
#endif
#ifdef GM_REVTS
  RevTsAdvance(c);
#endif
  ExportExpectation();
}

void GrammarMatch::BracketAdvance(unsigned char c) {
  if (c == '[') {  // a fresh bracket restarts target capture in every state
    br_state_ = kBrTarget;
    tgt_len_ = 0;
    return;
  }
  switch (br_state_) {
    case kBrIdle:
      break;
    case kBrTarget:
      if (c == 'Q') {  // 'Q' is '|': literal uppercase never survives WRT
        if (tgt_len_ > 0) {
          br_state_ = kBrLabel;
          label_j_ = (tgt_[0] == 0x40 && tgt_len_ > 1) ? 1 : 0;  // CAP-strip
          label_pos_ = 0;
        } else {
          br_state_ = kBrIdle;
        }
      } else if (c == ']' || AbortClass(c)) {
        br_state_ = kBrIdle;  // unpiped link / class violation
      } else if (tgt_len_ >= kTargetCap) {
        br_state_ = kBrIdle;  // overflow
      } else {
        tgt_[tgt_len_++] = c;
      }
      break;
    case kBrLabel:
      if (c == ']') {
        br_state_ = kBrIdle;  // label ended (early ']' = PREFIX site over)
      } else if (c == 'Q') {
        br_state_ = kBrScan;  // multi-parameter ([image:...Q...Q...])
      } else if (AbortClass(c)) {
        br_state_ = kBrIdle;
      } else if (label_j_ < tgt_len_ && c == tgt_[label_j_]) {
        ++label_j_;
        if (label_j_ >= tgt_len_) br_state_ = kBrPostCopy;
      } else if (label_pos_ == 0 && label_j_ == 1 && c == 0x40) {
        // Label carries its own cap flag ([@xyz Q @xyz...]): the first
        // label byte repeats 0x40; keep comparing from tgt_[1].
      } else {
        br_state_ = kBrScan;  // diverged (e.g. codeword lead in the label)
      }
      ++label_pos_;
      break;
    case kBrPostCopy:
      if (c == ']') {
        br_state_ = kBrIdle;  // full-copy site completed
      } else if (AbortClass(c)) {
        br_state_ = kBrIdle;
      } else {
        br_state_ = kBrScan;  // EXTENDS suffix running; idle until ']'
      }
      ++label_pos_;
      break;
    case kBrScan:
      if (c == ']' || AbortClass(c)) {
        br_state_ = kBrIdle;
      } else if (++label_pos_ > kTargetCap) {
        br_state_ = kBrIdle;  // runaway label region
      }
      break;
  }
}

void GrammarMatch::EchoAdvance(unsigned char c) {
  while (echo_k_ > 0 && c != title_[echo_k_]) echo_k_ = fail_[echo_k_ - 1];
  if (c == title_[echo_k_]) ++echo_k_;
  if (echo_k_ >= title_len_) echo_k_ = fail_[title_len_ - 1];  // full echo
}

void GrammarMatch::FinishTitle() {
  in_title_ = false;
  // tbuf_ holds the title body plus the already-appended first four bytes
  // of the close anchor ('L' '/' 0xDF 0x9B); strip them.
  title_len_ = tbuf_len_ >= 4 ? tbuf_len_ - 4 : 0;
  echo_k_ = 0;
  if (title_len_ <= 0) return;
  for (int i = 0; i < title_len_; ++i) title_[i] = tbuf_[i];
  fail_[0] = 0;
  for (int i = 1; i < title_len_; ++i) {
    int k = fail_[i - 1];
    while (k > 0 && title_[i] != title_[k]) k = fail_[k - 1];
    if (title_[i] == title_[k]) ++k;
    fail_[i] = k;
  }
}

void GrammarMatch::ExportExpectation() {
  have_expectation_ = false;
  // Priority: bracket family > echo (bracket knowledge is sharper);
  // exactly one expected byte is exported per byte.
  if (br_state_ == kBrLabel && label_j_ < tgt_len_
      ) {
    expected_byte_ = tgt_[label_j_];
    state_ = kFamilyPiped * 32 + (label_j_ < 31 ? label_j_ : 31);
    have_expectation_ = true;
  } else if (br_state_ == kBrPostCopy) {
    expected_byte_ = ']';
    state_ = kFamilyPipedPostCopy * 32;
    have_expectation_ = true;
  } else if (!in_title_ && echo_k_ >= kEchoArm && echo_k_ < title_len_
             ) {
    expected_byte_ = title_[echo_k_];
    state_ = kFamilyEcho * 32 + (echo_k_ < 31 ? echo_k_ : 31);
    have_expectation_ = true;
  }

  // Stage-4 families fire only if the bracket/echo chain above left the
  // channel idle (bracket/echo knowledge is sharper and lives in the main
  // stream; the tail families are regime-disjoint from it). Each is behind
  // its own gate, so with the gates off this whole block vanishes and the
  // object is identical to Stage 3.
#ifdef GM_REVTS
  if (!have_expectation_ && rev_phase_ != 0 && rev_can_pred_) {
    if (rev_phase_ == 1) {                 // YY (always 2 digits)
      if (rev_dpos_ < 2) {
        expected_byte_ = (unsigned char)rev_pred_yy_[rev_dpos_];
        state_ = kStateRevTs + rev_dpos_;
        have_expectation_ = true;
      }
    } else if (rev_phase_ == 2) {          // DDD (variable) then 'J'
      if (rev_dpos_ < rev_pred_ddd_len_) {
        expected_byte_ = (unsigned char)rev_pred_ddd_[rev_dpos_];
        state_ = kStateRevTs + 2 + (rev_dpos_ < 2 ? rev_dpos_ : 2);
        have_expectation_ = true;
      } else {
        expected_byte_ = 'J';
        state_ = kStateRevTs + 5;
        have_expectation_ = true;
      }
    } else {                               // SEC (variable) then '\n'
      if (rev_dpos_ < rev_pred_sec_len_) {
        expected_byte_ = (unsigned char)rev_pred_sec_[rev_dpos_];
        state_ = kStateRevTs + 6 + (rev_dpos_ < 4 ? rev_dpos_ : 4);
        have_expectation_ = true;
      } else {
        expected_byte_ = '\n';
        state_ = kStateRevTs + 11;
        have_expectation_ = true;
      }
    }
  }
#endif
}



#ifdef GM_REVTS
// REVTS: regime-1 revision timestamps. Anchor 0xDF 0xCD 0x4E ("timestamp>"),
// then the phda9 image YY DDD 'J' seconds '\n' (phda9_preprocess.h:804). YY
// and DDD are copied from the previous block (run locality); seconds run
// through a deterministic fixed-point affine tracker keyed on the rev-id
// delta (the digit run that precedes the anchor). Per-position confidence
// carries the (high) seconds uncertainty.
void GrammarMatch::RevTsAdvance(unsigned char c) {
  // rev-id capture: value of the most recent complete ASCII digit run.
  if (c >= '0' && c <= '9') {
    if (!in_digits_) { in_digits_ = true; last_int_ = 0; }
    if (last_int_ < 100000000000LL) last_int_ = last_int_ * 10 + (c - '0');
  } else {
    in_digits_ = false;
  }

  if (rev_phase_ == 0) {
    const bool in_window = (pos_ >= revts_lo_ && pos_ <= revts_hi_);
    if (in_window && (recent_ & 0xFFFFFFULL) == 0xDFCD4EULL) {
      rev_revid_ = last_int_;
      rev_phase_ = 1; rev_dpos_ = 0;
      rev_act_yy_ = rev_act_ddd_ = rev_act_sec_ = 0;
      rev_can_pred_ = have_prev_rev_;
      if (rev_can_pred_) {
        rev_pred_yy_[0] = (char)('0' + (int)((prev_yy_ / 10) % 10));
        rev_pred_yy_[1] = (char)('0' + (int)(prev_yy_ % 10));
        rev_pred_ddd_len_ = GmDecEnc(prev_ddd_, rev_pred_ddd_);
        const long long drev = rev_revid_ - prev_revid_;
        long long psec = prev_sec_ + ((alpha_q16_ * drev) >> 16);
        if (psec < 0) psec = 0; else if (psec > 86399) psec = 86399;
        rev_pred_sec_len_ = GmDecEnc(psec, rev_pred_sec_);
      }
    }
    return;
  }

  if (c == '\n') {  // block complete: commit + update the affine tracker
    const long long drev = rev_revid_ - prev_revid_;
    if (have_prev_rev_ && drev != 0) {
      const long long obs = ((rev_act_sec_ - prev_sec_) << 16) / drev;
      if (obs >= 0 && obs <= (8LL << 16)) {
        alpha_q16_ += (obs - alpha_q16_) >> 5;  // EMA, lr 1/32
        if (alpha_q16_ < 0) alpha_q16_ = 0;
        else if (alpha_q16_ > (8LL << 16)) alpha_q16_ = 8LL << 16;
      }
    }
    prev_yy_ = rev_act_yy_; prev_ddd_ = rev_act_ddd_;
    prev_sec_ = rev_act_sec_; prev_revid_ = rev_revid_;
    have_prev_rev_ = true;
    rev_phase_ = 0; rev_can_pred_ = false;
    return;
  }
  if (c >= '0' && c <= '9') {
    const int d = c - '0';
    if (rev_phase_ == 1) {
      rev_act_yy_ = rev_act_yy_ * 10 + d;
      if (++rev_dpos_ >= 2) { rev_phase_ = 2; rev_dpos_ = 0; }
    } else if (rev_phase_ == 2) {
      // caps keep the decimal image within rev_pred_ddd_/sec_ (12 bytes)
      // even if a false-fire lands on a pathological digit run.
      if (rev_act_ddd_ < 100000000LL) rev_act_ddd_ = rev_act_ddd_ * 10 + d;
      ++rev_dpos_;
    } else {
      if (rev_act_sec_ < 100000000LL) rev_act_sec_ = rev_act_sec_ * 10 + d;
      ++rev_dpos_;
    }
  } else if (c == 'J' && rev_phase_ == 2) {
    rev_phase_ = 3; rev_dpos_ = 0;  // DDD -> seconds delimiter
  }
}
#endif

