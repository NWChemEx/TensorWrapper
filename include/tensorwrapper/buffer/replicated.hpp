#pragma once
#include <tensorwrapper/buffer/local.hpp>

namespace tensorwrapper::buffer {

class Replicated : public Local {
private:
    /// Type *this derives from
    using base_type = Local;

public:
    // Pull in base's ctors
    using base_type::base_type;
};

} // namespace tensorwrapper::buffer
