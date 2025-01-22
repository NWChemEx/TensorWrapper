/*
 * Copyright 2025 NWChemEx-Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "eigen_contraction.hpp"
#include <tensorwrapper/allocator/eigen.hpp>
#include <tensorwrapper/buffer/eigen.hpp>

namespace tensorwrapper::buffer {

using rank_type            = unsigned short;
using base_reference       = BufferBase::base_reference;
using const_base_reference = BufferBase::const_base_reference;
using return_type          = BufferBase::dsl_reference;
using pair_type            = std::pair<rank_type, rank_type>;
using vector_type          = std::vector<pair_type>;

// N.b. will create about max_rank**3 instantiations of eigen_contraction
static constexpr unsigned int max_rank = 6;

/// Wraps the contraction once we've worked out all of the template params.
template<typename RVType, typename LHSType, typename RHSType,
         typename ModesType>
return_type eigen_contraction(RVType&& rv, LHSType&& lhs, RHSType&& rhs,
                              ModesType&& sum_modes) {
    rv.value() = lhs.value().contract(rhs.value(), sum_modes);
    return rv;
}

/// This function converts @p sum_modes to a statically sized array
template<std::size_t N = 0ul, typename FloatType, rank_type RVRank,
         rank_type LHSRank, rank_type RHSRank>
return_type n_contraction_modes(buffer::Eigen<FloatType, RVRank>& rv,
                                const buffer::Eigen<FloatType, LHSRank>& lhs,
                                const buffer::Eigen<FloatType, RHSRank>& rhs,
                                const vector_type& sum_modes) {
    // Can't contract more modes than a tensor has (this is recursion end point)
    constexpr auto max_n = std::min({LHSRank, RHSRank});
    if constexpr(N == max_n + 1) {
        throw std::runtime_error("Contracted more modes than a tensor has!!?");
    } else {
        if(N == sum_modes.size()) {
            std::array<pair_type, N> temp;
            for(std::size_t i = 0; i < temp.size(); ++i) temp[i] = sum_modes[i];
            return eigen_contraction(rv, lhs, rhs, std::move(temp));
        } else {
            return n_contraction_modes<N + 1>(rv, lhs, rhs, sum_modes);
        }
    }
}

/// This function works out the rank of RHS
template<rank_type RHSRank = 0ul, typename FloatType, rank_type RVRank,
         rank_type LHSRank>
return_type rhs_rank(buffer::Eigen<FloatType, RVRank>& rv,
                     const buffer::Eigen<FloatType, LHSRank>& lhs,
                     const_base_reference rhs, const vector_type& sum_modes) {
    if constexpr(RHSRank == max_rank + 1) {
        throw std::runtime_error("RHS has rank > max_rank");
    } else {
        if(RHSRank == rhs.rank()) {
            using allocator_type  = allocator::Eigen<FloatType, RHSRank>;
            const auto& rhs_eigen = allocator_type::rebind(rhs);
            return n_contraction_modes(rv, lhs, rhs_eigen, sum_modes);
        } else {
            return rhs_rank<RHSRank + 1>(rv, lhs, rhs, sum_modes);
        }
    }
}

/// This function works out the rank of LHS
template<rank_type LHSRank = 0ul, typename FloatType, rank_type RVRank>
return_type lhs_rank(buffer::Eigen<FloatType, RVRank>& rv,
                     const_base_reference lhs, const_base_reference rhs,
                     const vector_type& sum_modes) {
    if constexpr(LHSRank == max_rank + 1) {
        throw std::runtime_error("LHS has rank > max_rank");
    } else {
        if(LHSRank == lhs.rank()) {
            using allocator_type  = allocator::Eigen<FloatType, LHSRank>;
            const auto& lhs_eigen = allocator_type::rebind(lhs);
            return rhs_rank(rv, lhs_eigen, rhs, sum_modes);
        } else {
            return lhs_rank<LHSRank + 1>(rv, lhs, rhs, sum_modes);
        }
    }
}

/// This function works out the rank of rv
template<typename FloatType, rank_type RVRank>
return_type eigen_contraction_(base_reference rv, const_base_reference lhs,
                               const_base_reference rhs,
                               const vector_type& sum_modes) {
    if constexpr(RVRank == max_rank + 1) {
        throw std::runtime_error("Return has rank > max_rank");
    } else {
        if(RVRank == rv.rank()) {
            using allocator_type = allocator::Eigen<FloatType, RVRank>;
            auto& rv_eigen       = allocator_type::rebind(rv);
            return lhs_rank(rv_eigen, lhs, rhs, sum_modes);
        } else {
            constexpr auto RVp1 = RVRank + 1;
            return eigen_contraction_<FloatType, RVp1>(rv, lhs, rhs, sum_modes);
        }
    }
}

template<typename FloatType>
return_type eigen_contraction(base_reference rv, const_base_reference lhs,
                              const_base_reference rhs,
                              const vector_type& sum_modes) {
    return eigen_contraction_<FloatType, 0>(rv, lhs, rhs, sum_modes);
}

#define EIGEN_CONTRACTION(FLOAT_TYPE)                             \
    template return_type eigen_contraction<FLOAT_TYPE>(           \
      base_reference, const_base_reference, const_base_reference, \
      const vector_type&)

EIGEN_CONTRACTION(float);
EIGEN_CONTRACTION(double);

#ifdef ENABLE_SIGMA
EIGEN_CONTRACTION(sigma::UFloat);
EIGEN_CONTRACTION(sigma::UDouble);
#endif

#undef EIGEN_CONTRACTION

} // namespace tensorwrapper::buffer