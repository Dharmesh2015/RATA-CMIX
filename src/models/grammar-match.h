// Adapted from the supplied fx-deepmix CPU-only GrammarMatch model.
// Distributed under the project GPL; see THIRD_PARTY_NOTICES.txt.
#ifndef GRAMMAR_MATCH_H
#define GRAMMAR_MATCH_H

#include "model.h"
#include "../fx4_config.h"

#include <array>

// ECHO arm threshold knob: -DGM_ARM=<1|2|3>. Default 2 keeps Stage-2
// behavior (bit-identical object at the default value).
#ifndef GM_ARM
#define GM_ARM 2
#endif

// GrammarMatch: the markup-bundle reframe (GRAMMAR_MATCH_DESIGN.md).
// match.cpp predicts continuation of previously-seen bytes found by hashing;
// GrammarMatch predicts continuation of previously-seen bytes found by
// parsing the post-WRT modeled stream. Families (Stage-0 verified):
//
//   PIPED - piped links image as '[' <target> 'Q' <label> ']' post-WRT
//           ('|' -> 'Q' via the encode_text remap). When the label's byte
//           image literally extends the target's image (cap flag 0x40
//           stripped), the label bytes are derivable; when the first label
//           byte diverges (e.g. codeword lead), the channel goes idle and
//           the learned confidence prices the rest.
//   ECHO  - in-article recurrences of the page title, whose post-WRT image
//           is byte-identical at every recurrence. Title anchors (verified
//           against the real modeled stream, english.dic):
//           <title>  ->  'L' 0xDF 0x9B 'N'      (0x4C 0xDF 0x9B 0x4E)
//           </title> ->  'L' '/' 0xDF 0x9B 'N'  (0x4C 0x2F 0xDF 0x9B 0x4E)
// The byte->bit bridge, the count-limited learned confidence table and the
// ByteUpdate hook mirror Match line for line; the state index is
// (family x position-bucket) instead of match_length_. Idle emits exactly
// 0.5, which the predictor wiring maps to an exact stretched 0 via
// MixerInput::SetZero (Logit(0.5) itself is not reliably 0 under fast-math),
// so an idle channel contributes nothing to any mixer dot product and
// nothing to any weight update.
// All state is bounded and scalar; the parser consumes only completed bytes
// (bit_context at ByteUpdate time), so encode/decode symmetry is Match's.
class GrammarMatch : public Model {
 public:
  GrammarMatch(const unsigned int& bit_context, int limit, float delta);
  // Attribution census (stderr-only, output-neutral): the confidence table
  // already holds per-(family x position) observation counts and learned
  // accuracies — dump them at teardown so every full run doubles as a
  // per-family contribution census at scale.
  ~GrammarMatch();
  const std::valarray<float>& Predict() const;
  void Perceive(int bit);
  void ByteUpdate();

  // Introspection for the offline mini-harness / debugging; never called on
  // the compressor's hot path.
  bool HaveExpectation() const { return have_expectation_; }
  unsigned char ExpectedByte() const { return expected_byte_; }
  int StateIndex() const { return state_; }

  // Layer-0 mixer context (the design's Stage-4 amplifier): 0 when idle,
  // else 1 + state index (family x position bucket). Always < 256.
  unsigned long long MixerContext() const {
    return have_expectation_ ? 1ULL + state_ : 0ULL;
  }

#ifdef GM_REVTS
  // Regime windows (byte-offset gate in the modeled stream). Defaults are
  // the full range so the offline harness can feed a regime-specific tail
  // file directly; production narrows these to the r1 / r2 tail offsets.
  // The families also self-gate on the anchor shape + run context, so a
  // coarse window suffices (documented per family in the .cpp).
  void SetRevTsRange(unsigned long long lo, unsigned long long hi) {
#ifdef GM_REVTS
    revts_lo_ = lo; revts_hi_ = hi;
#else
    (void)lo; (void)hi;
#endif
  }
  void SetIwLangRange(unsigned long long lo, unsigned long long hi) {
    (void)lo; (void)hi;
  }
#endif


 private:
  enum Family { kFamilyPiped = 0, kFamilyPipedPostCopy = 1, kFamilyEcho = 2
  };
  enum BracketState { kBrIdle, kBrTarget, kBrLabel, kBrPostCopy, kBrScan
  };

  void ParseByte(unsigned char c);
  void BracketAdvance(unsigned char c);
  void EchoAdvance(unsigned char c);
  void FinishTitle();
  void ExportExpectation();
#ifdef GM_REVTS
  void RevTsAdvance(unsigned char c);
#endif

  // --- STEP 0: learned-confidence state-budget audit (blocking) ----------
  // predictions_/counts_ are 256-wide; the confidence index (state_) MUST
  // stay < 256 with EVERY family gate on. The base families keep the
  // family*32 scheme; RevTs uses an explicit cumulative base:
  //
  //   family            base   width   states        gate
  //   Piped               0      32    0..31         (always)
  //   PipedPostCopy      32       1    32            (always)
  //   Echo               64      32    64..95        (always)
  //   RevTs             169      12    169..180      GM_REVTS
  //
  // Max confidence index = 180 < 256 => PASS. (The dead Stage-3/4 families
  // that occupied 96..168 and 181..204 were removed in the 2026-07-21
  // cleanup — see REMOVED.md; their bases are historical, RevTs keeps 169.)

  // Post-WRT alien/marker bytes end any capture: port of mk_bundle's EX
  // class minus 0x06/0x07/0x0C, which post-WRT are the legitimate WRT case
  // flags / escape and occur inside word images.
  static bool AbortClass(unsigned char c) {
    return c <= 0x05 || c == 0x08 || c == '\n' ||
           (c >= 0x0E && c <= 0x10);
  }

  static const int kTitleCap = 256;   // raw cap was 200 cps
  static const int kTargetCap = 512;  // raw caps: 120 cps target
  static const int kEchoArm = GM_ARM;  // k>=ARM before ECHO exports
  static const unsigned long long kTitleOpen = 0x4CDF9B4EULL;
  static const unsigned long long kTitleClose = 0x4C2FDF9B4EULL;

  // Stage-4 confidence-table bases (see the STEP 0 audit above).
#ifdef GM_REVTS
  static const int kStateRevTs = 169;      // width 12
  // "timestamp>" post-WRT codeword: 0xDF 0xCD 0x4E (verified 243,425 sites
  // on r1_lex/r1_pre). YY/DDD/seconds image follows per phda9_preprocess.h.
#endif


  const unsigned int& bit_context_;

#ifdef FX4_GRAMMAR_CENSUS
  // Optional research-only attribution census. Production builds omit its
  // state and stderr output.
  std::array<unsigned long long, 256> census_arms_;
  std::array<unsigned long long, 256> census_hits_;
#endif

  // byte->bit bridge (Match's pattern)
  unsigned char expected_byte_ = 0, bit_pos_ = 128;
  bool have_expectation_ = false;
  int state_ = 0;
  int limit_;
  float delta_, divisor_;
  std::array<float, 256> predictions_;
  std::array<int, 256> counts_;

  // title tracking (ECHO)
  unsigned long long recent_ = 0;  // last 8 bytes, newest in the low byte
  bool in_title_ = false;
  int tbuf_len_ = 0;
  int title_len_ = 0;
  int echo_k_ = 0;  // KMP prefix-match length against title_
  std::array<unsigned char, kTitleCap> tbuf_;
  std::array<unsigned char, kTitleCap> title_;
  std::array<int, kTitleCap> fail_;

  // bracket tracking (PIPED)
  int br_state_ = kBrIdle;
  int tgt_len_ = 0;
  int label_j_ = 0;    // next target byte the label is expected to copy
  int label_pos_ = 0;  // bytes seen since 'Q'
  std::array<unsigned char, kTargetCap> tgt_;



#ifdef GM_REVTS
  unsigned long long pos_ = 0;  // byte offset in the modeled stream
#endif



#ifdef GM_REVTS
  unsigned long long revts_lo_ = 0, revts_hi_ = ~0ULL;
  int rev_phase_ = 0;   // 0 off, 1 YY, 2 DDD, 3 SEC
  int rev_dpos_ = 0;    // digit position within the current phase
  bool rev_can_pred_ = false;
  // predicted decimal images (from the previous block + affine tracker).
  // Buffers are padded well past the real widths (DDD<=3, seconds<=5); the
  // accumulators below are capped so a malformed run can never overflow.
  char rev_pred_yy_[4] = {0};
  char rev_pred_ddd_[12] = {0}; int rev_pred_ddd_len_ = 0;
  char rev_pred_sec_[12] = {0}; int rev_pred_sec_len_ = 0;
  // actual accumulators for the current block
  long long rev_act_yy_ = 0, rev_act_ddd_ = 0, rev_act_sec_ = 0;
  // previous block + affine tracker (fixed point Q16 -> deterministic)
  long long prev_yy_ = 0, prev_ddd_ = 0, prev_sec_ = 0, prev_revid_ = 0;
  bool have_prev_rev_ = false;
  long long alpha_q16_ = 32768;  // seconds ~= prev + alpha*(rev_id delta)
  // last complete run of ASCII digits (the rev-id precedes the anchor)
  long long last_int_ = 0; bool in_digits_ = false;
  long long rev_revid_ = 0;  // rev-id captured at the current anchor
#endif


};

#endif
