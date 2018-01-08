/*
 * poisson_regression.h
 * Copyright (C) 2017 zhangyule <zhangyule01@baidu.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef POISSON_REGRESSION_H
#define POISSON_REGRESSION_H

#include <cstdio>

namespace zdsp {

struct KeyValueData {
  float prediction_value;
  float label;
};

struct Parameters {
  float alpha;
  float beta;
  float learning_rate;
};

struct ShrinkageMethod;

class PoissonRegressionUpdater {
public:
  virtual void UpdateByData(const KeyValueData& kv, Parameters* parameters) = 0;
  void SetRegularizationCoefficient(double Lk_coeffcient) {
    this->lk_coefficent_ = Lk_coeffcient;
  }
  void SetThrinkageMethod(ShrinkageMethod* method) { this->thrinkage_ = method; }

  void Destroy();
protected:
  double lk_coefficent_;
  ShrinkageMethod* thrinkage_;

};

class ShrinkageMethod {
public:
  virtual void apply(const KeyValueData& kv, Parameters* parameters) = 0;
};

class ConstantShrinkage : public ShrinkageMethod {
public:
  void apply(const KeyValueData& kv, Parameters* parameters);
};

class PoissonRegression_L2_SGD_Impl : public PoissonRegressionUpdater {
public:
  void UpdateByData(const KeyValueData& kv, Parameters* parameters);
};

//class PoissionRegression_L2_NEWTON_Impl : public PoissionRegressionUpdater {
//public:
//  void UpdateByData(const KeyValueData& kv, Parameters* parameters);
//};

class PoissonRegressionPredictor {
public:
  virtual double PredictData(const Parameters& parameters, const KeyValueData& kv) const = 0;
};

class PoissonRegressionPredictor_Impl : public PoissonRegressionPredictor {
public:
  double PredictData(const Parameters& parameters, const KeyValueData& kv) const;
};


struct PackageManager {
  PoissonRegressionUpdater* updater;
  PoissonRegressionPredictor* predictor;
};

#if 0
class PoissonRegression_L1 : PoissonRegressionUpdater {
public:
  void UpdateByData(const KeyValueData& kv, Parameters* parameters);
};
#endif

}

#endif /* !POISSON_REGRESSION_H */
