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

#include <tensorwrapper/allocator/allocator.hpp>
#include <tensorwrapper/buffer/buffer.hpp>
#include <tensorwrapper/diis/diis.hpp>

namespace tensorwrapper::diis {

using tensor_type = DIIS::tensor_type;

tensor_type DIIS::extrapolate(const tensor_type& X, const tensor_type& E) {
    auto rt = X.buffer().allocator().runtime();
    tensorwrapper::allocator::Eigen<double> allocator(rt);

    // Append new values to stored values
    m_samples_.push_back(X);
    m_errors_.push_back(E);

    // If we're over the max number of stored values, pop the oldest ones
    // Also update m_B_ to overwrite the oldest values
    if(m_errors_.size() > m_max_samples_) {
        m_errors_.pop_front();
        m_samples_.pop_front();

        // Overwrite the top-left block with the bottom right block.
        // No need to zero out the parts that aren't overwritten,
        // they'll be overwritten in the next step
        if(m_max_samples_ > 1) {
            m_B_.block(0, 0, m_max_samples_ - 1, m_max_samples_ - 1) =
              m_B_.block(1, 1, m_max_samples_ - 1, m_max_samples_ - 1);
        }
    }

    // Current number of stored values
    size_type sz = m_errors_.size();

    // Add the new values to m_B_
    size_type i = sz - 1;
    for(size_type j = 0; j <= i; ++j) { // compute upper triangle
        tensor_type& E_i = m_errors_.at(i);
        tensor_type& E_j = m_errors_.at(j);

        tensor_type temp;
        temp("")               = E_i("mu,nu") * E_j("mu,nu");
        const auto& temp_eigen = allocator.rebind(temp.buffer());
        m_B_(i, j)             = temp_eigen.get_elem({});

        // Fill in lower triangle
        if(i != j) m_B_(j, i) = m_B_(i, j);
    }

    // Solve for expansion coefficients
    matrix_type A           = matrix_type::Zero(sz + 1, sz + 1);
    A.topLeftCorner(sz, sz) = m_B_.topLeftCorner(sz, sz);
    A.row(sz).setConstant(-1.0);
    A.col(sz).setConstant(-1.0);
    A(sz, sz) = 0.0;

    vector_type b = vector_type::Zero(sz + 1);
    b(sz)         = -1.0;

    vector_type coefs = A.colPivHouseholderQr().solve(b);

    // Extrapolate the new X from the coefficients.
    tensor_type new_X;
    new_X("mu,nu") = m_samples_.at(0)("mu,nu") * coefs(0);
    for(int i = 1; i < sz; i++) {
        tensor_type x_i;
        x_i("mu,nu")   = m_samples_.at(i)("mu,nu") * coefs(i);
        new_X("mu,nu") = new_X("mu,nu") + x_i("mu,nu");
    }
    return new_X;
}

bool DIIS::operator==(const DIIS& rhs) const noexcept {
    return ((m_max_samples_ == rhs.m_max_samples_) &&
            (m_samples_ == rhs.m_samples_) && (m_errors_ == rhs.m_errors_) &&
            (m_B_ == rhs.m_B_));
}

} // end namespace tensorwrapper::diis
