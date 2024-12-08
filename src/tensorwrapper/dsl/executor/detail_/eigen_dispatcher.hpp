#pragma once
#include "../../../allocator/detail_/eigen_buffer_unwrapper.hpp"
#include <tensorwrapper/tensor/tensor_class.hpp>
namespace tensorwrapper::dsl::executor::detail_ {

template<typename Functor>
class EigenDispatcher {
private:
    using rank_type = unsigned short;
    using unwrapper = allocator::detail_::EigenBufferUnwrapper;

    template<typename LHSType, typename RHSType>
    auto dispatch(LHSType&& lhs, RHSType&& rhs) {
        auto l = unwrapper::downcast(std::forward<LHSType>(lhs));
        auto r = unwrapper::downcast(std::forward<RHSType>(rhs));

        return std::visit(
          [](auto&& l_lambda) {
              return std::visit(
                [&l_lambda](auto&& r_lambda) {
                    Functor f;
                    return f(l_lambda, r_lambda);
                },
                r);
          },
          l);
    }
};

} // namespace tensorwrapper::dsl::executor::detail_