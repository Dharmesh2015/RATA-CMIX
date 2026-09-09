#include "scr2-match.h"

namespace {

const float kLearningRate = FX2_SCR2_MATCH_LR_E3 * 1e-3f;

// The coder must never be handed 0 or 1.
const float kFloor = 0.0001f;
const float kCeil = 0.9999f;

float Clamp(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Bit position within the current byte: c0 is a leading 1 followed by the
// bits decoded so far, so its bit length minus one is how many bits are in.
int BitPos(unsigned int c0) {
  int bpos = 0;
  for (unsigned int t = c0; t > 1u; t >>= 1) ++bpos;
  return bpos > 7 ? 7 : bpos;
}

}  // namespace

Scr2Match::Scr2Match(const unsigned int& bit_context)
    : Model(1), bit_context_(bit_context) {
  matched_.fill(0);
  // 0.5 is "no opinion". The first observation in a cell moves it, and until
  // then the channel says nothing the mixer can act on.
  confidence_.fill(0.5f);
  seen_.fill(0);
}

bool Scr2Match::StillLive(int bpos) const {
  // If the bits decoded so far already contradict the expectation it cannot
  // be right, and saying so is worth more than hedging.
  if (bpos == 0) return true;
  const unsigned int seen_bits = bit_context_ & ((1u << bpos) - 1u);
  const unsigned int want_bits =
      static_cast<unsigned int>(expected_byte_) >> (8 - bpos);
  return seen_bits == want_bits;
}

const std::valarray<float>& Scr2Match::Predict() {
  if (!have_expectation_) {
    // Exactly 0.5 so the predictor can SetZero this input: an idle byte then
    // costs the mixer nothing at all, rather than a weight it must learn to
    // ignore. Most bytes are idle.
    outputs_[0] = 0.5f;
    return outputs_;
  }

  const int bpos = BitPos(bit_context_);
  if (!StillLive(bpos)) {
    outputs_[0] = 0.5f;
    return outputs_;
  }

  const int bit = (expected_byte_ >> (7 - bpos)) & 1;
  const float c = confidence_[ConfidenceIndex()];
  p_ = Clamp(bit ? c : 1.0f - c, kFloor, kCeil);
  outputs_[0] = p_;
  return outputs_;
}

void Scr2Match::Perceive(int bit) {
  if (!have_expectation_) return;

  const int bpos = BitPos(bit_context_);
  if (!StillLive(bpos)) return;

  // Only the cell that actually spoke gets updated, and only while the
  // expectation is still live for this byte.
  const int want = (expected_byte_ >> (7 - bpos)) & 1;
  const int idx = ConfidenceIndex();
  const float target = (bit == want) ? 1.0f : 0.0f;
  confidence_[idx] += kLearningRate * (target - confidence_[idx]);
  confidence_[idx] = Clamp(confidence_[idx], 0.01f, 0.99f);
  if (seen_[idx] < 0xFFFFFFFFu) ++seen_[idx];
}

void Scr2Match::ByteUpdate() {
  // bit_context_ holds the byte that just completed, exactly as it does for
  // the other byte-level models at this point in Predictor::Perceive.
  const unsigned char byte = static_cast<unsigned char>(bit_context_);

  // Advance every pattern's match length by this byte, then take the longest
  // live match. O(patterns) per byte with no history buffer and no rescan:
  // the match length IS the state.
  int best_len = 0;
  int best_idx = -1;
  int agree = 0;
  bool conflict = false;
  unsigned char want = 0;

  for (int i = 0; i < kPatterns; ++i) {
    const std::uint8_t* pat = scr2::kBlob + scr2::kOffset[i];
    const int len = scr2::kLength[i];
    int m = matched_[i];

    if (m < len && pat[m] == byte) {
      ++m;
    } else {
      m = (pat[0] == byte) ? 1 : 0;
    }
    // A pattern matched to its end has said everything it can; restart it so
    // it can fire again on the next occurrence.
    if (m >= len) m = (pat[0] == byte) ? 1 : 0;
    matched_[i] = static_cast<std::uint8_t>(m);

    if (m >= kMinContext && m < len) {
      const unsigned char next = pat[m];
      if (m > best_len) {
        best_len = m;
        best_idx = i;
        want = next;
        agree = 1;
        conflict = false;
      } else if (m == best_len) {
        // Patterns tied at the longest match. They only disagree if they
        // predict different bytes -- and this table is full of patterns
        // sharing a prefix, so most ties agree: 2,833 of 3,191.
        if (next == want) {
          ++agree;
        } else {
          conflict = true;
        }
      }
    }
  }

  // Silence only on genuine disagreement. Silencing every tie, as the first
  // version did, threw away the model's most confident predictions -- worth
  // 3 bytes at 1 MB, which is the difference between -10 and -13.
  if (best_idx < 0 || conflict) {
    have_expectation_ = false;
    best_pattern_ = -1;
    return;
  }

  best_pattern_ = best_idx;
  expected_byte_ = want;
  have_expectation_ = true;
  agree_bucket_ = (agree > 1) ? 1 : 0;

  // The table is sorted by occurrence count, so index buckets are monotone in
  // how much traffic each carries and the mixer sees a meaningful ordering.
  const int pb = (best_idx * kPatternBuckets) / kPatterns;
  int lb = best_len - kMinContext;
  if (lb < 0) lb = 0;
  if (lb >= kLengthBuckets) lb = kLengthBuckets - 1;
  state_ = pb * kLengthBuckets + lb;
}
