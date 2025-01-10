#pragma once
#include <tensorwrapper/buffer/buffer_base.hpp>

namespace tensorwrapper::buffer {

/** @brief Relatively template-free API for Eigen tensor contraction.
 *
 *  Eigen's tensor library relies on a heavy amount of template meta-programming
 *  to implement contract. TensorWrapper strives to do things at runtime.
 *  Ultimately, to have it both ways we need to create contraction dispatch
 *  instantiations for every combination of template parameters that Eigen may
 *  end up seeing, that's what the functions in this header do.
 *
//  *  The entry point into this infrastructure is currently the return_rank
 *  method, which kicks the process off by working out the rank of the tensor
 *  which will
 *
 */
template<typename FloatType>
BufferBase::dsl_reference eigen_contraction(
  BufferBase::base_reference rv, BufferBase::const_base_reference lhs,
  BufferBase::const_base_reference rhs,
  const std::vector<std::pair<unsigned short, unsigned short>>& sum_modes);

} // namespace tensorwrapper::buffer