/*
 * model_exe.cc
 * Copyright (C) 2017 zhangyule <zhangyule01@baidu.com>
 *
 * Distributed under terms of the MIT license.
 */

#include <cmath>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <climits>
#include <unistd.h>
#include "poisson_regression.h"

using namespace zdsp;

PackageManager manager;

void set_env() {
  manager.updater = new PoissonRegression_L2_SGD_Impl;
  manager.predictor = new PoissonRegressionPredictor_Impl;
  manager.updater->SetThrinkageMethod(new ConstantShrinkage);
}

void destroy_env() {
  manager.updater->Destroy();
  delete manager.updater;
  delete manager.predictor;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "Usage: cat datafile | %s "
        "alpha:beta:learning_rate:regulation, LINE:%d, FILE:%s\n",
        argv[0], __LINE__, __FILE__);
    return -1;
  }
  set_env();
  std::vector<KeyValueData> kv_data;
  kv_data.reserve(1000);
  float regulation_coef;
  Parameters param;
  int ret = sscanf(argv[1], "%f:%f:%f:%f",
      &param.alpha, &param.beta, &param.learning_rate, &regulation_coef);

  if (ret != 4) {
    fprintf(stderr, "Usage: cat datafile | %s "
        "alpha:beta:learning_rate:regulation, LINE:%d, FILE:%s\n",
        argv[0], __LINE__, __FILE__);
    destroy_env();
    return -1;
  }

  KeyValueData data;
  char buf[255];
  manager.updater->SetRegularizationCoefficient(regulation_coef);
  while (fgets(buf, 255, stdin)) {
    sscanf(buf, "%f %f\n", &data.prediction_value, &data.label);
    printf("%f %f\n", data.prediction_value, data.label);
    manager.updater->UpdateByData(data, &param);
    kv_data.push_back(data);
  }
  float mse = 0;
  kv_data.shrink_to_fit();
  for (auto& dat : kv_data) {
    double result = manager.predictor->PredictData(param, dat);
    mse += fabs(result - dat.label);
  }
  mse /= kv_data.size();

  printf("Training result: alpah:%f\tbeta:%f\tloss:%f\nDone.\n",
      param.alpha, param.beta, mse);

  destroy_env();
  return 0;
}
