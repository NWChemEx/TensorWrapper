#include <tensorwrapper/diis/diis.hpp>

namespace tensorwrapper::diis {

using tensor_type = DIIS::tensor_type;

tensor_type DIIS::extrapolate(const tensor_type& X, const tensor_type& E) {
    // Append new values to stored values
    m_x_values_.push_back(X);
    m_errors_.push_back(E);

    // If we're over the max number of stored values, pop the oldest ones
    // Also update m_B_ to overwrite the oldest values
    if(m_errors_.size() > m_max_vec_) {
        m_errors_.pop_front();
        m_x_values_.pop_front();

        // No need to zero out the parts that aren't overwritten,
        // they'll be overwritten in the next step
        if(m_max_vec_ > 1) {
            m_B_.block(0, 0, m_max_vec_ - 1, m_max_vec_ - 1) =
              m_B_.block(1, 1, m_max_vec_ - 1, m_max_vec_ - 1);
        }
    }

    // Current number of stored values
    size_type sz = m_errors_.size();

    // Add the new values to m_B_
    size_type i = sz - 1;
    for(size_type j = 0; j <= i; ++j) // compute upper triangle
    {
        tensor_type& E_i = m_errors_.at(i);
        tensor_type& E_j = m_errors_.at(j);
        m_B_(i, j)       = dot(E_i("mu,nu"), E_j("mu,nu"));

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
    new_X("mu,nu") = coefs(0) * m_x_values_.at(0)("mu,nu");
    for(int i = 1; i < sz; i++) {
        tensor_type x_i;
        x_i("mu,nu")   = coefs(i) * m_x_values_.at(i)("mu,nu");
        new_X("mu,nu") = new_X("mu,nu") + x_i("mu,nu");
    }
    return new_X;
}

bool DIIS::operator==(const DIIS& rhs) const noexcept {
    return ((m_max_vec_ == rhs.m_max_vec_) &&
            (m_x_values_ == rhs.m_x_values_) && (m_errors_ == rhs.m_errors_) &&
            (m_B_ == rhs.m_B_));
}

void DIIS::hash(tensorwrapper::detail_::Hasher& h) const {
    h(m_max_vec_);
    for(auto& x : m_x_values_) h(x);
    for(auto& e : m_errors_) h(e);
}

} // end namespace tensorwrapper::diis
