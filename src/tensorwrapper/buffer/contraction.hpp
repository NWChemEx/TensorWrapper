#pragma once
#include <tensorwrapper/buffer/buffer_base.hpp>

namespace tensorwrapper::buffer {
namespace {
static int max_rank = 6;
}

/// Wraps the contraction once we've worked out all of the template params.
template<typename RVType, typename LHSType, typename RHSType,
         typename ModesType>
BufferBase::dsl_reference contraction(RVType&& rv, LHSType&& lhs, RHSType&& rhs,
                                      ModesType&& sum_modes) {
    rv.value() = lhs.value().contract(rhs.value(), sum_modes);
    return rv;
}

// template<int OutRank, int LHSRank>
// dsl_reference find_lhs_rank(const_labeled_reference lhs,
//                             const_labeled_reference rhs) {
//     if constexpr(LHSRank == max_rank) {
//         throw std::runtime_error("LHS must be less than max_rank");
//     } else {
//         if(lhs.object().rank() == LHSRank) {
//             return find_rhs_rank<OutRank, LHSRank, 0>(lhs, rhs);
//         } else {
//             return find_lhs_rank<OutRank, LHSRank + 1>(lhs, rhs);
//         }
//     }
// }

// template<int OutRank>
// dsl_reference find_out_rank(int out_rank, const_labeled_reference lhs,
//                             const_labeled_reference rhs) {
//     if constexpr(OutRank == max_rank) {
//         throw std::runtime_error("Output must be less than max_rank");
//     } else {
//         if(out_rank == OutRank) {
//             return find_lhs_rank<OutRank, 0>(lhs, rhs);
//         } else {
//             return find_out_rank<OutRank + 1>(lhs, rhs);
//         }
//     }
// }
} // namespace tensorwrapper::buffer