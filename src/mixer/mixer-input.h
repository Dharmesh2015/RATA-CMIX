#ifndef MIXER_INPUT_H
#define MIXER_INPUT_H

#include "sigmoid.h"

#include <cstdlib>
#include <memory>

class MixerInput {
 public:
  MixerInput(const Sigmoid& sigmoid, float eps);
  void SetNumModels(int num_models);
  void SetExtraInputSize(size_t size);
  // Identical clamp + logit-table lookup the out-of-line SetInput always
  // did; inline so per-bit marshaling keeps the limits and table pointer in
  // registers.
  void SetInput(int index, float p) { inputs_[index] = StretchChecked(p); }
  // Batched form of SetInput(): convert a gathered run of model
  // probabilities into consecutive input slots in one tight loop.
  void SetInputsChecked(int base, const float* p, int count) {
    float* dst = inputs_.get() + base;
    for (int i = 0; i < count; ++i) dst[i] = StretchChecked(p[i]);
  }
  void SetStretchedInput(int index, float p) {
    if (p > stretched_max_) p = stretched_max_;
    else if (p < stretched_min_) p = stretched_min_;
    inputs_[index] = p;
  }
  void SetStretchedInputUnchecked(int index, float p) { inputs_[index] = p; }
  void SetZero(int index) { inputs_[index] = 0.0f; }
  void SetExtraInput(size_t index, float p) {
    if (p > stretched_max_) p = stretched_max_;
    else if (p < stretched_min_) p = stretched_min_;
    extra_inputs_[index] = p;
  }
  // The layer input rows are plain 64-byte-aligned flat arrays; the mixers
  // read them through raw pointers (no valarray indirection per bit).
  const float* Inputs() const { return inputs_.get(); }
  unsigned int NumInputs() const { return num_models_; }
  const float* ExtraInputs() const { return extra_inputs_.get(); }

 private:
  float StretchChecked(float p) const {
    if (p < min_) p = min_;
    else if (p > max_) p = max_;
    int index = p * logit_size_;
    if (index >= logit_size_) index = logit_size_ - 1;
    else if (index < 0) index = 0;
    return logit_table_[index];
  }
  struct FreeDeleter { void operator()(void* p) const { std::free(p); } };
  std::unique_ptr<float[], FreeDeleter> inputs_;
  std::unique_ptr<float[], FreeDeleter> extra_inputs_;
  unsigned int num_models_ = 0;
  const float* logit_table_;
  int logit_size_;
  float min_, max_, stretched_min_, stretched_max_;
};

#endif
