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

  // --- STEP 0: learned-confidence state-budget audit (blocking) ----------
  // predictions_/counts_ are 256-wide; the confidence index (state_) MUST
  // stay < 256 with EVERY family gate on. The base families keep the
  // family*32 scheme; RevTs uses an explicit cumulative base:
  //
  //   family            base   width   states        gate
  //   Piped               0      32    0..31         (always)
  //   PipedPostCopy      32       1    32            (always)
  //   Echo               64      32    64..95        (always)
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


};

#endif
