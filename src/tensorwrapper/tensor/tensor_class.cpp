#include "detail_/tensor_pimpl.hpp"

namespace tensorwrapper {

// -- Ctors, assignment, and dtor

Tensor::Tensor() noexcept = default;

Tensor::~Tensor() noexcept = default;

// -- Private methods

bool Tensor::has_pimpl_() const noexcept { return static_cast<bool>(m_pimpl_); }

} // namespace tensorwrapper
