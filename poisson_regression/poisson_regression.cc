/*
 * poisson_regression.c
 * Copyright (C) 2017 zhangyule <zhangyule01@baidu.com>
 *
 * Distributed under terms of the MIT license.
 */

#include <cstdlib>
#include <cmath>

#include "poisson_regression.h"


namespace zdsp {
// theta^t * x = alpha * x + beta
void PoissonRegression_L2_SGD_Impl::UpdateByData(const KeyValueData& kv, Parameters* parameters) {
  double h_theta_x = exp((parameters->alpha + parameters->beta * kv.prediction_value));
  parameters->alpha += -(-kv.label + h_theta_x) * 0.01;
  parameters->beta += -(-kv.label + h_theta_x) * kv.prediction_value * 0.01;
}

void PoissonRegressionUpdater::Destroy() {
  if (thrinkage_ == NULL) {
    delete thrinkage_;
    thrinkage_ = NULL;
  }
}

void ConstantShrinkage::apply(const KeyValueData& kv, Parameters* parameters) {
}

double PoissonRegressionPredictor_Impl::PredictData(const Parameters& param, const KeyValueData& kv) const {
  double tmp = exp(param.alpha + kv.prediction_value * param.beta);
  return kv.label ? exp(1-tmp) : exp(-tmp);
}

}
