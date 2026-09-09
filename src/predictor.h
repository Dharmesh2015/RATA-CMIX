#ifndef PREDICTOR_H
#define PREDICTOR_H

#include "mixer/sigmoid.h"
#include "mixer/mixer-input.h"
#include "mixer/mixer.h"
#include "mixer/byte-mixer.h"
#include "mixer/sse.h"
#include "models/model.h"
#include "models/byte-model.h"
#include "context-manager.h"
#include "models/direct.h"
#include "models/direct-hash.h"
#include "models/indirect.h"
#include "models/match.h"
#include "models/ppmd.h"
#include "models/bracket.h"
#include "models/fxcmv1.h"
#include "models/grammar-match.h"
#include "models/scr2-match.h"
#include "models/morphology-match.h"
#include "models/causal-donor.h"
#include "mixer/lstm.h"
#include "contexts/context-hash.h"
#include "contexts/bracket-context.h"
#include "contexts/sparse.h"
#include "contexts/indirect-hash.h"
#include "contexts/interval.h"
#include "contexts/interval-hash.h"
#include "contexts/bit-context.h"
#include "contexts/combined-context.h"

#include "ds/SmallVector.h"
#include "ds/emhash_set.hpp"

#include "../cpp_infer/src/opt/model_opt.h"

#include <vector>
#include <set>
#include <memory>
// PPMD's sub-allocator, in MEGABYTES. StartSubAllocator does SASize << 20,
// so the default 14000 reserves 14 GB. It is plain new[] rather than an
// mmap -- ppmd.cpp has mmap_to_disk = false -- so pages are committed
// lazily as the model fills, and a 1% run touches only a fraction of it.
//
// TWO REASONS THIS NUMBER MATTERS MORE THAN IT LOOKS.
//
// The Hutter Prize caps the decompressor at 10 GB. 14000 is over it.
// fx2-cmix-transformer reports 9663 MB compressing and 9547 MB
// decompressing, so the reference entry fits and this does not.
//
// And it decides whether a full run is compute-bound or I/O-bound. Matt
// Mahoney's Intel result for fx2-cmix-transformer -- 695,558 ns/byte, about
// 8 days -- against 131,561 on AMD is attributed to PPM paging under a
// 12 GB WSL2 cap, a 5.3x penalty from the SAME binary. PPM access is
// irregular, so once the working set exceeds RAM every lookup risks a
// fault, and no amount of predictor optimisation touches that.
//
// The 1% benchmark cannot see any of this: at 5.87 MB of input the model
// has not filled, the profile shows no kernel time, and swap stays at zero.
// So 1% timings rank arms correctly against each other and must NOT be
// multiplied by 100 to project a full run.
// Fixed for the release: order 25 over a 14,000 MB sub-allocator, held on
// disk. StartSubAllocator does SASize << 20, so this is 14 GB of address
// space; ppmd.cpp maps it to ppm.temp and keeps VmRSS down with a
// MADV_DONTNEED cadence, which is what fits the 10 GB judging limit.
#define FX2_PPMD_MEMORY_MB 14000
#define FX2_PPMD_ORDER 25

// FX2_LSTM_EXPERT adds an online LSTM as an ADDITIVE outer-mixer expert.
// Default 0: builds are byte-identical to the measured v521 configuration.
// FX2_LSTM_EXPERT_LAYERS selects 1x200 (T1) or 2x200 stacked (T2).
#ifndef FX2_LSTM_EXPERT
#define FX2_LSTM_EXPERT 1
#endif
//
// MEASURED, AND THE MIDDLE IS NOT EMPTY. A single 256-cell layer -- 72%
// of the multiply-adds, 770 us/byte against 1160 -- opens +6 against
// EMA96 and CROSSES OVER at about 786 KB, reaching -8 by 1.11 MB and
// still widening. 1x170 never crossed: it was byte-identical to EMA96
// out to 655 KB. So width is not irrelevant; 170 is simply below the
// threshold and 256 is above it.
//
// Do not read a comparison in this slot before 1 MB. Both wrong calls on
// this question came from checkpoints under 400 KB, where every capacity
// looks flat because the expert has not finished learning. 1x256 was
// +1 to +2 against no expert at 327 KB and -11 against it at 1.11 MB.
#ifndef FX2_LSTM_EXPERT_CELLS
#define FX2_LSTM_EXPERT_CELLS 200
#endif
#ifndef FX2_LSTM_EXPERT_LAYERS
#define FX2_LSTM_EXPERT_LAYERS 2
#endif

// BPTT depth. The backward pass fires every FX2_LSTM_EXPERT_HORIZON bytes
// and then walks all HORIZON stored timesteps, so the AMORTISED cost is
// one epoch of backward work per byte NO MATTER WHAT THIS IS SET TO:
// horizon iterations divided by horizon bytes. The previous claim here --
// that it trades gradient quality against cost roughly linearly -- was
// wrong, and would have sent anyone looking for a speed lever to the one
// knob that cannot provide it.
//
// What this does change: the length of credit assignment, and the size of
// the per-epoch output_layer_ ring buffer, which is 128 x 205 x 400 floats
// at the default -- about 42 MB, which is why the copy in Lstm::Perceive
// was worth fusing. Halving the horizon halves that buffer, so it can
// still be faster through cache locality, just not through less work.
//
// To actually reduce the 15.3% the backward pass costs, the passes have to
// be SKIPPED rather than shortened. See FX2_LSTM_BPTT_STRIDE in mixer/
// lstm.h.
#ifndef FX2_LSTM_EXPERT_HORIZON
#define FX2_LSTM_EXPERT_HORIZON 128
#endif

// T2-E: expected-byte hint. When the LSTM's predicted byte still agrees with
// the bits decoded so far, its next bit becomes an extra outer expert. This is
// deliberately weak (0.25/0.75, not the raw probability): it says "the LSTM
// expects this bit" without asserting confidence the LSTM has not earned.
// SetInput takes a probability and applies Logit itself.
#ifndef FX2_LSTM_EXPERT_HINT
#define FX2_LSTM_EXPERT_HINT 1
#endif

// T2-D: disagreement expert. The transformer is frozen corpus knowledge and the
// LSTM is live adaptation, so the LSTM is most likely to add something exactly
// where the two disagree. Abstains (0.5) when they broadly agree.
#ifndef FX2_LSTM_EXPERT_DISAGREE
#define FX2_LSTM_EXPERT_DISAGREE 1
#endif

// T2-G: uncertainty gate. Voice the LSTM only where the transformer is itself
// unsure; abstain where it is confident.
#ifndef FX2_LSTM_EXPERT_GATE
#define FX2_LSTM_EXPERT_GATE 1
#endif

// FX2_GRAMMAR_MATCH adds Halvor Yttredal's GrammarMatch (fx-deepmix), ported
// here unchanged from fx4-cmix release/google-cloud-hutter, as ONE additive
// outer-mixer channel. It predicts continuation of previously-seen bytes found
// by PARSING the post-WRT stream -- piped links '[' target 'Q' label ']' and
// in-article recurrences of the page title -- where match.cpp finds them by
// hashing, so it is orthogonal to every existing context model. Idle emits
// exactly 0.5, which the wiring below maps to a stretched hard zero via
// MixerInput::SetZero, so a silent channel contributes nothing to any mixer
// dot product and nothing to any weight update.
// ON by default: measured -1 byte and holding across every checkpoint from
// 786 KB on, against the same stack without it. -DFX2_GRAMMAR_MATCH=0 is the
// t2edg_stat ablation.
#ifndef FX2_GRAMMAR_MATCH
#define FX2_GRAMMAR_MATCH 1
#endif

// FX2_GRAMMAR_MIXER is fx4-cmix's Stage-4 amplifier: a layer-0 mixer keyed
// on GrammarMatch's own state (0 when idle, else 1 + family x position
// bucket, always < 256). Without it a single global weight has to serve
// every state at once, and the census shows those states are not alike --
// accuracies run from 0.74 to 0.996, and one ECHO bucket alone carries
// 18,064 of the 33k armed bits. Needs FX2_GRAMMAR_MATCH.
// ON by default: measured -10 to -13 bytes over the first 327 KB against
// the identical stack without it, where GrammarMatch alone was flat at +0
// over the same range. The channel was never the weak part; sharing one
// mixer weight across 29 unlike states was.
#ifndef FX2_GRAMMAR_MIXER
#define FX2_GRAMMAR_MIXER 1
#endif


// Give the channel its own layer-0 mixer, keyed on the head's confidence.
// GrammarMatch was worth -3 bytes as a bare input and -13 once it had a
// context, so this is tested as the second step rather than assumed.
#include <optional>
#include <string>
#include <cstdint>
#include <cstdio>

// Prints an error message to stderr and exits with a nonzero status.
[[noreturn]] void Fail(const char* fmt, ...);

struct PredictorOptions {
  // Run only the ppmd model. Predict() returns the ppmd's bit prediction and
  // no other model is constructed or updated.
  bool ppmd_only = false;
  // Run only the ppmd and the transformer it feeds (requires
  // transformer_weights). Like ppmd_only, no other model is constructed or
  // updated and Predict() returns the ppmd's bit prediction, but the
  // transformer is still stepped on every byte, so its distributions can be
  // dumped with save_transformer_probs and its loss is printed.
  bool transformer_only = false;
  // If nonempty, every probability distribution the ppmd outputs is appended
  // to this file: for each processed byte, the probabilities of the bytes in
  // the lstm's vocabulary (in vocabulary order), rounded to float16.
  std::string save_ppmd_probs;
  // If nonempty, the ppmd and the transformer are not run. The byte-level
  // probability distributions the transformer would have produced are read
  // from this file instead (same format as save_ppmd_probs). Requires
  // transformer_weights to be set: the loaded distributions replace the
  // transformer's output, so running the lstm instead is an error.
  std::string load_transformer_probs;
  // Number of bytes the predictor will process. Used to validate the size of
  // the load_transformer_probs file.
  unsigned long long num_input_bytes = 0;
  // If nonempty, the lstm is replaced by the pretrained transformer whose
  // weights (FX2TFW01 or FX2TFWC1/2 format) are loaded from this path. The
  // transformer is not trained; it is fed the ppmd's distribution (rounded
  // to float16 exactly as --save-ppmd-probs writes it) and the completed
  // byte, and its output distribution goes to the code downstream of the
  // lstm through the same float16 rounding and zero guard the
  // --save-ppmd-probs / --load-transformer-probs file pipeline applies.
  // Requires the enwik9 vocabulary (205 bytes). The transformer's loss is
  // always printed.
  std::string transformer_weights;
  // Testing option (requires transformer_weights): if nonempty, every
  // distribution passed downstream in place of the lstm's output (the
  // transformer's float16-rounded rows, and the ppmd's float16 rows at the
  // article-start tokens the transformer cannot predict) is appended to
  // this file, in the format of save_ppmd_probs.
  std::string save_transformer_probs;
};

// Buffered writer converting floats to float16 (IEEE half precision).
class HalfFileWriter {
 public:
  explicit HalfFileWriter(const std::string& path);
  ~HalfFileWriter();
  void Write(const float* values, size_t n);
  // Appends values that are already float16.
  void WriteHalves(const uint16_t* values, size_t n);

 private:
  void Flush();
  std::string path_;
  std::vector<uint16_t> buffer_;
  size_t used_ = 0;
  FILE* file_;
};

// Buffered reader converting float16 back to floats. Checks on construction
// that the file holds exactly the expected number of distributions.
class HalfFileReader {
 public:
  HalfFileReader(const std::string& path,
      unsigned long long num_distributions, unsigned int vocab_size);
  ~HalfFileReader();
  void Read(float* values, size_t n);

 private:
  std::string path_;
  std::vector<uint16_t> buffer_;
  size_t pos_ = 0, available_ = 0;
  FILE* file_;
};

class Predictor {
 public:
  Predictor(const std::vector<bool>& vocab,
      const PredictorOptions& options = PredictorOptions());
  float Predict();
  void Perceive(int bit);
  void Pretrain(int bit);

 private:
  unsigned long long GetNumModels();
  void AddMixer(int layer, const unsigned long long& context,
      float learning_rate);
  void AddAuxiliary();
  void AddPPMD();
  void AddBracket();
  void AddWord();
  void AddDirect();
  void AddMatch();
  void AddDoubleIndirect();
  void AddMixers();
  void WritePpmdProbs();
  void LoadTransformerProbs();
  void AccumulateTransformerLoss();
  void TransformerByteUpdate();

  llvm::SmallVector<Indirect<Nonstationary>, 30-7> indirect_ns_models_; // non-stationary
  llvm::SmallVector<Indirect<RunMap>, 1> indirect_r_models_; // run map
  llvm::SmallVector<Direct, 1> direct_models_;
  llvm::SmallVector<Match, 10> match_models_;
  
  std::optional<Bracket> bracket_model_;
#if FX2_GRAMMAR_MATCH
  std::optional<GrammarMatch> grammar_model_;
#endif
  std::optional<Scr2Match> scr2_model_;
  std::optional<MorphologyMatch> morphology_model_;
  std::optional<CausalDonor> causal_donor_model_;
// One definition, guarded to match exactly the places that write and
// read it. Several experts want the previous bit's coded probability and
// the earlier overlapping conditions were one -D away from either a
// redefinition or an undeclared identifier.
  size_t auxiliary_size_ = 2; // 0 -> fxcm, 1 -> byte_mixer
  SSE sse_;
  llvm::SmallVector<MixerInput,2> layers_;
  llvm::SmallVector<Mixer, 23> mixer_0_;
  llvm::SmallVector<Mixer, 1> mixer_1_;
  std::vector<unsigned int> auxiliary_;
  ContextManager manager_;
  Sigmoid sigmoid_;
  std::optional<PPMD::PPMD> byte_model_;
  std::optional<ByteMixer> byte_mixer_;
#if FX2_LSTM_EXPERT
  // Additive online-LSTM expert. byte_mixer_ is owned by the transformer, so
  // this is a SECOND mixer fed the same PPMd distribution the LSTM used to get
  // before the transformer displaced it. It contributes one extra outer-mixer
  // input and touches nothing else: lstmpr/lstmex still come from byte_mixer_,
  // so FXCM's bridge and the v521 result are unaffected.
  std::optional<ByteMixer> lstm_expert_;
#if FX2_LSTM_EXPERT_HINT
  float LstmExpectedByteHint() const;
#endif
#if FX2_LSTM_EXPERT_DISAGREE
  float LstmDisagreementExpert() const;
#endif
#if FX2_LSTM_EXPERT_GATE
  float LstmUncertaintyExpert() const;
#endif
  float lstm_expert_output_ = 0.5f;
#endif
  std::vector<bool> vocab_;
  std::optional<FXCM> fxcm_model_;
  bool ppmd_only_ = false;
  bool transformer_only_ = false;
  unsigned int vocab_size_ = 0;
  std::vector<int> vocab_bytes_;  // byte values in the vocabulary, ascending
  std::vector<float> probs_scratch_;
  std::unique_ptr<HalfFileWriter> ppmd_probs_writer_;
  std::unique_ptr<HalfFileWriter> transformer_probs_writer_;
  std::unique_ptr<HalfFileReader> transformer_probs_reader_;
  bool print_transformer_loss_ = false;
  unsigned long long num_input_bytes_ = 0;
  double transformer_loss_sum_ = 0;
  unsigned long long transformer_tokens_ = 0;

  // Pretrained transformer replacing the lstm (see
  // PredictorOptions::transformer_weights).
  std::unique_ptr<fx2::opt::TransformerOpt> transformer_;
  int byte_to_index_[256];
  std::vector<uint16_t> half_scratch_;   // float16-rounded distributions
  std::vector<float> transformer_probs_; // the transformer's output row
  // Vocabulary indices of the last 15 processed bytes; an article ends
  // exactly where this window matches the encoded article separator.
  unsigned char separator_window_[15];
  // Tokens of the current article piece processed so far. Articles are cut
  // into pieces of at most kMaxArticleTokens tokens, exactly like the
  // training data loader's split_article_lengths; each piece is a fresh
  // transformer context.
  unsigned long long article_tokens_ = 0;
};

#endif

