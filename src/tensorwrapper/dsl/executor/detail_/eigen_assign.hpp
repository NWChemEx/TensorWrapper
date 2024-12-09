#pragma once
#include <tensorwrapper/buffer/eigen.hpp>

namespace tensorwrapper::dsl::executor::detail_ {

struct EigenAssign {
    using rank_type = unsigned short;
    template<typename FloatType1, rank_type N1, typename... Args>
    buffer::Eigen<FloatType1, N1>& run(buffer::Eigen<FloatType1, N1>& lhs,
                                       Args&&... args) {
        if constexpr(sizeof...(args) == 1) {
            return run_(lhs, std::forward<Args>(args)...);
        } else {
            throw std::runtime_error("Expected two buffers");
        }
    }

    template<typename FloatType1, rank_type N1, typename FloatType2,
             rank_type N2>
    buffer::Eigen<FloatType1, N1>& run_(
      buffer::Eigen<FloatType1, N1>& lhs,
      const buffer::Eigen<FloatType2, N2>& rhs) {
        if constexpr(std::is_same_v<FloatType1, FloatType2> && N1 == N2) {
            return lhs = rhs;
        } else {
            throw std::runtime_error("Must have same rank and floating type");
        }
    }
};

} // namespace tensorwrapper::dsl::executor::detail_