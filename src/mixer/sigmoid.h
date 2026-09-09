#ifndef SIGMOID_H
#define SIGMOID_H

#include <vector>

class Sigmoid {
 public:
  Sigmoid(int logit_size);
  float Logit(float p) const;
  static float Logistic(float p);
  static float FastLogistic(float p);
  const float* Table() const { return logit_table_.data(); }
  int TableSize() const { return logit_size_; }

 private:
  float SlowLogit(float p);
  int logit_size_;
  std::vector<float> logit_table_;
};

#endif
