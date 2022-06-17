#include "detail_/add.hpp"
#include "detail_/labeled.hpp>"
#include <tensorwrapper/tensor/expression/expression.hpp>

namespace tensorwrapper::tenosr::expression {

Expression::Expression(labeled_tensor t) :
  m_pimpl_(std::make_unique<Labeled>(std::move(t))) {}

Expression::Expression(labeled_tot t) :
  m_pimpl_(std::make_unique<Labeled>(std::move(t))) {}

Expression Expression::operator+(Expression rhs) const {
    auto pimpl = std::make_unique<Add>(*this, std::move(rhs));
    return Expression(std::move(pimpl));
}

Expression Expression::operator*(Expression rhs) const {
    auto pimpl = std::make_unique<Times>(*this, std::move(rhs));
    return Expression(std::move(pimpl));
}

} // namespace tensorwrapper::tenosr::expression
