#pragma once
#include <memory>

namespace tensorwrapper {
namespace detail_ {
class TensorPIMPL;
}

class Tensor {
public:
    /// Type of the object implementing *this
    using pimpl_type = detail_::TensorPIMPL;

    /// Type of a pointer to an object of type pimpl_type
    using pimpl_pointer = std::unique_ptr<pimpl_type>;

    Tensor() noexcept;

    /// Defaulted no-throw dtor
    ~Tensor() noexcept;

private:
    /// Does *this have a PIMPL?
    bool has_pimpl_() const noexcept;

    /// Object actually implementing *this
    pimpl_pointer m_pimpl_;
};

} // namespace tensorwrapper
