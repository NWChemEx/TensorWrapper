#pragma once
#include "eigen_pimpl.hpp"
#include <sstream>

namespace tensorwrapper::buffer::detail_ {

template<typename FloatType, unsigned int Rank>
class EigenTensor : public EigenPIMPL<FloatType> {
private:
    using my_type   = EigenTensor<FloatType, Rank>;
    using base_type = EigenPIMPL<FloatType>;

public:
    using typename base_type::const_pimpl_reference;
    using typename base_type::const_pointer;
    using typename base_type::eigen_rank_type;
    using typename base_type::label_type;
    using typename base_type::pointer;
    using typename base_type::string_type;

    using eigen_data_type       = eigen::data_type<FloatType, Rank>;
    using eigen_reference       = eigen_data_type&;
    using const_eigen_reference = const eigen_data_type&;

    eigen_reference value() noexcept { return m_tensor_; }
    const_eigen_reference value() const noexcept { return m_tensor_; }

    bool operator==(const my_type& rhs) const noexcept {
        eigen::data_type<FloatType, 0> r = (m_tensor_ - rhs.m_tensor_).sum();
        return r() == FloatType(0.0);
    }

protected:
    pimpl_pointer clone_() const override {
        return std::make_unique<my_type>(*this);
    }
    eigen_rank_type rank_() const noexcept override { return Rank; }
    pointer data_() noexcept override { return m_tensor_.data(); }
    const_pointer data_() const noexcept override { return m_tensor_.data(); }
    bool are_equal_(const_base_reference rhs) const noexcept override {
        return base_type::template are_equal_impl_<my_type>(rhs);
    }

    string_type to_string_() const override {
        std::stringstream ss;
        ss << m_tensor_;
        return ss.str();
    }
    void addition_assignment_(label_type this_labels, label_type lhs_labels,
                              label_type rhs_labels, const_pimpl_reference lhs,
                              const_pimpl_reference rhs) override;

    void subtraction_assignment_(label_type this_labels, label_type lhs_labels,
                                 label_type rhs_labels,
                                 const_pimpl_reference lhs,
                                 const_pimpl_reference rhs) override;

    void hadamard_assignment_(label_type this_labels, label_type lhs_labels,
                              label_type rhs_labels, const_pimpl_reference lhs,
                              const_pimpl_reference rhs) override;

    void contraction_assignment_(label_type this_labels, label_type lhs_labels,
                                 label_type rhs_labels,
                                 const_pimpl_reference lhs,
                                 const_pimpl_reference rhs) override;

    void permute_assignment_(label_type this_labels, label_type rhs_labels,
                             const_pimpl_reference rhs) override;

    void scalar_multiplication_(label_type this_labels, label_type rhs_labels,
                                FloatType scalar,
                                const_pimpl_reference rhs) override;

private:
    eigen_data_type m_tensor_;
};

} // namespace tensorwrapper::buffer::detail_

#include "eigen_addition_assignment.ipp"
#include "eigen_contraction_assignment.ipp"
#include "eigen_hadamard_assignment.ipp"
#include "eigen_permute_assignment.ipp"
#include "eigen_scalar_multiplication.ipp"
#include "eigen_subtraction_assignment.ipp"