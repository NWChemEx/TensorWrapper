#pragma once
#include <tensorwrapper/tensor/data_evaluator/element_evaluator.hpp>
#include <tensorwrapper/tensor/data_evaluator/tile_evaluator.hpp>

namespace testing {

/** The various Evaluators are pure virtual, so these are simple derived
 *  classes for use in testing. They also serve as examples for the minimal
 *  requirements for new evaluators.
 */

struct TestScalarElementEval
  : public tensorwrapper::tensor::data_evaluator::ScalarElementEvaluator {};

struct TestTensorElementEval
  : public tensorwrapper::tensor::data_evaluator::TensorElementEvaluator {};

struct TestScalarTileEval
  : public tensorwrapper::tensor::data_evaluator::ScalarTileEvaluator {};

struct TestTensorTileEval
  : public tensorwrapper::tensor::data_evaluator::TensorTileEvaluator {};

} // namespace testing