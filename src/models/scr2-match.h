#ifndef SCR2_MATCH_H
#define SCR2_MATCH_H

#include "model.h"
#include "scr2-table.h"

#include <array>
#include <cstdint>

// Scr2Match: SCR2's pattern table used as a MODEL, never as a transform.
//
// WHY NOT THE TRANSFORM. SCR2 substitutes its patterns with marker and escape
// bytes taken from the values the stream does not use. Our 1% prefix uses 203
// of 256; the transform takes all 51 that are free, and the frozen 6M
// transformer -- fixed at a 205-symbol vocabulary -- refuses the result
// outright:
//
//   cmix error: the transformer was trained on the enwik9 vocabulary of 205
//   bytes, but this input has a vocabulary of 254 bytes
//
// No SCR2 parameter avoids that: the mechanism REQUIRES symbols outside the
// source alphabet. So the transform cannot sit in front of a frozen model at
// any setting, and the only coherent place left for the pattern knowledge is
// AFTER the transformer -- at the mixer, alongside its prediction, rather
// than in the bytes before it.
//
// This is that. The stream is untouched, so the alphabet stays 203, the
// transformer sees exactly what it was trained on, the other 571 models keep
// the byte context they rely on, and nothing needs retraining. The 1,232-byte
// derivation shrinks to a 3,389-byte compiled table (see scr2-table.h), which
// ships once rather than per occurrence.
//
// SHAPE, copied from GrammarMatch deliberately. That model is worth -13 on
// this branch and won by being exactly this: an additive channel plus its own
// layer-0 mixer context keyed on which structure fired, never rewriting the
// coded probability. Everything that rewrote working output lost -- ESN/NLMS
// +20, APM +36, DirectState +12.
//
// HOW IT PREDICTS. It tracks, for every pattern, how many of its leading bytes
// match the tail of already-decoded history. When the longest such match is at
// least kMinContext bytes and the patterns tied at that length agree on the
// next byte, that byte is an expectation; the per-bit probability is that
// byte's bit, scaled by a confidence learned per (pattern bucket x matched
// length x whether several patterns agreed). Idle emits exactly 0.5 so the
// predictor can SetZero it and an idle byte costs the mixer nothing.
//
// MEASURED, at 1,048,576 bytes against the same stack without it:
//
//   silencing every tie (the first version)          -10 bytes
//   speaking when the tied patterns agree            -13 bytes   <- this
//   ... plus KMP failure links on the fallback       -10 bytes
//   ... plus confidence split by bit position         -4 bytes
//
// So agreement is the whole win and the other two repairs were removed.
// Ties are not noise: on the real table 2,833 of 3,191 tied positions have
// every tied pattern predicting the SAME next byte, because the table is full
// of patterns sharing a prefix. Agreement across patterns is evidence FOR an
// expectation, not against it.
//
// The two that were dropped, so they are not retried blind. KMP failure links
// recover overlapping prefixes the plain fallback discards, and they do change
// the coded stream -- a different archive of identical length -- but the byte
// count did not move, and they cost kBlobBytes + kPatterns of RAM and a prefix
// table. Splitting confidence by bit position measured actively harmful: it
// multiplies the confidence cells by 8, only about 1% of bytes arm, and the
// cells never fill enough to pay the dilution back. That is the same failure
// GM_BPCTX had, for the same reason.

// Leading bytes that must match before a pattern is allowed to speak. Too low
// and every 8-byte pattern fires on noise; too high and the long patterns
// never arm.
#define FX2_SCR2_MATCH_MIN_CONTEXT 4

// Confidence learning rate, in 1e-3 units.
#define FX2_SCR2_MATCH_LR_E3 20

class Scr2Match : public Model {
 public:
  explicit Scr2Match(const unsigned int& bit_context);

  const std::valarray<float>& Predict();
  void Perceive(int bit);
  void ByteUpdate();

  bool HaveExpectation() const { return have_expectation_; }
  unsigned char ExpectedByte() const { return expected_byte_; }

  // 0 when idle, else 1 + state, with one extra bit for whether several
  // patterns agreed. Idle must stay 0 so the mixer keeps one bucket for
  // "this model is silent", which is most bytes. 129 values in all --
  // comparable to GrammarMatch's 96 families, which is the shape that won.
  unsigned long long MixerContext() const {
    if (!have_expectation_) return 0ULL;
    return 1ULL + static_cast<unsigned long long>(
        state_ + agree_bucket_ * kStates);
  }

  static const int kPatterns = scr2::kPatternCount;
  static const int kMinContext = FX2_SCR2_MATCH_MIN_CONTEXT;

  // 8 pattern buckets x 8 matched-length buckets. Kept small on purpose:
  // GM_BPCTX lost by splitting a sparse channel too thin, and only about 1%
  // of bytes arm here.
  static const int kPatternBuckets = 8;
  static const int kLengthBuckets = 8;
  static const int kStates = kPatternBuckets * kLengthBuckets;
  // One cell for a lone pattern, one for several agreeing.
  static const int kAgreeBuckets = 2;
  static const int kConfidenceCells = kStates * kAgreeBuckets;

  static_assert(kMinContext >= 2 && kMinContext <= 16,
                "FX2_SCR2_MATCH_MIN_CONTEXT must be between 2 and 16");

 private:
  // Index into confidence_/seen_ for the cell that is speaking this bit.
  int ConfidenceIndex() const { return state_ * kAgreeBuckets + agree_bucket_; }
  // False once the bits decoded so far contradict the expectation.
  bool StillLive(int bpos) const;

  const unsigned int& bit_context_;

  // matched_[i] is how many leading bytes of pattern i currently match the
  // tail of history. Advanced one byte at a time; no rescan, no history
  // buffer -- the state IS the match length.
  std::array<std::uint8_t, kPatterns> matched_;

  // Learned P(the expectation is right) per cell.
  std::array<float, kConfidenceCells> confidence_;
  std::array<std::uint32_t, kConfidenceCells> seen_;

  bool have_expectation_ = false;
  unsigned char expected_byte_ = 0;
  int state_ = 0;
  int agree_bucket_ = 0;
  int best_pattern_ = -1;

  float p_ = 0.5f;
};

#endif  // SCR2_MATCH_H
