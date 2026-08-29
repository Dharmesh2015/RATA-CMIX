#ifndef PPMD_H
#define PPMD_H

#include "byte-model.h"

#include <array>
#include <memory>
#include <vector>

namespace PPMD {

struct ppmd_Model;

class PPMD : public ByteModel {
 public:
  PPMD(int order, int memory, const unsigned int& bit_context,
      const std::vector<bool>& vocab);
  ~PPMD();
  std::valarray<float>& Predict();
  const std::array<float, 4>& PredictOrderBands();
  float ByteProbability(unsigned int byte) const;
  // Building the 4 order-band trees inside ppmd_PrepareByte() (4x
  // ConvertShadowSQ calls, each with a 3KB memset + 255-node tree build)
  // is pure overhead unless a post-R1 expert actually consumes
  // PredictOrderBands(). Those trees are entirely separate from
  // tree_zero_/tree_total_/ConvertSQ() (which feed the accepted PPMd
  // probability), so skipping them has zero effect on p_base or archive
  // bytes -- verified by construction, and by exact roundtrip tests.
  // Defaults to true (matches prior always-on behavior); callers that
  // never touch this get identical behavior to before this change.
  void SetOrderBandsNeeded(bool needed);
  unsigned int EffectiveOrder() const;
  unsigned int LastEscapeDepth() const;
  float RecentEscapeRate() const;
  void Perceive(int bit);
  void ByteUpdate();
 private:
  const unsigned int& byte_;
  std::unique_ptr<ppmd_Model> ppmd_model_;
  std::valarray<int> byte_map_;
  std::array<unsigned int, 256> tree_zero_;
  std::array<unsigned int, 256> tree_total_;
  std::array<std::array<unsigned int, 256>, 4> band_tree_zero_{};
  std::array<std::array<unsigned int, 256>, 4> band_tree_total_{};
  std::array<float, 4> band_outputs_{{0.5f, 0.5f, 0.5f, 0.5f}};
  std::vector<unsigned char> disabled_bytes_;
  unsigned int tree_context_ = 1;
  bool vocab_full_ = false;
};

} // namespace PPMD

#endif
