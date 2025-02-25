#pragma once
#include "eigen_pimpl.hpp"
#include <sstream>
#include <tensorwrapper/detail_/integer_utilities.hpp>
#include <tensorwrapper/shape/smooth.hpp>

namespace tensorwrapper::buffer::detail_ {

template<typename FloatType, unsigned int Rank>
class EigenTensor : public EigenPIMPL<FloatType> {
private:
    using my_type   = EigenTensor<FloatType, Rank>;
    using base_type = EigenPIMPL<FloatType>;

public:
    using typename base_type::const_base_reference;
    using typename base_type::const_pimpl_reference;
    using typename base_type::const_pointer;
    using typename base_type::const_reference;
    using typename base_type::const_shape_reference;
    using typename base_type::eigen_rank_type;
    using typename base_type::index_vector;
    using typename base_type::label_type;
    using typename base_type::pimpl_pointer;
    using typename base_type::pointer;
    using typename base_type::reference;
    using typename base_type::size_type;
    using typename base_type::string_type;

    using smooth_view                 = shape::SmoothView<shape::Smooth>;
    using const_smooth_view           = shape::SmoothView<const shape::Smooth>;
    using const_smooth_view_reference = const const_smooth_view&;
    using eigen_data_type             = eigen::data_type<FloatType, Rank>;
    using eigen_reference             = eigen_data_type&;
    using const_eigen_reference       = const eigen_data_type&;

    EigenTensor() = default;

    explicit EigenTensor(const_smooth_view_reference shape) :
      m_tensor_(allocate_from_shape_(shape, std::make_index_sequence<Rank>())) {
    }

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
    size_type extent_(eigen_rank_type i) const override {
        return m_tensor_.dimension(i);
    }
    pointer data_() noexcept override { return m_tensor_.data(); }
    const_pointer data_() const noexcept override { return m_tensor_.data(); }
    reference get_elem_(index_vector index) override {
        return unwrap_vector_(std::move(index),
                              std::make_index_sequence<Rank>());
    }
    const_reference get_elem_(index_vector index) const override {
        return unwrap_vector_(std::move(index),
                              std::make_index_sequence<Rank>());
    }
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
                                 const_shape_reference result_shape,
                                 const_pimpl_reference lhs,
                                 const_pimpl_reference rhs) override;

    void permute_assignment_(label_type this_labels, label_type rhs_labels,
                             const_pimpl_reference rhs) override;

    void scalar_multiplication_(label_type this_labels, label_type rhs_labels,
                                FloatType scalar,
                                const_pimpl_reference rhs) override;

private:
    template<std::size_t... I>
    auto allocate_from_shape_(const_smooth_view_reference shape,
                              std::index_sequence<I...>) {
        return eigen_data_type(shape.extent(I)...);
    }

    template<std::size_t... I>
    reference unwrap_vector_(index_vector index, std::index_sequence<I...>) {
        return m_tensor_(tensorwrapper::detail_::to_long(index.at(I))...);
    }

    template<std::size_t... I>
    const_reference unwrap_vector_(index_vector index,
                                   std::index_sequence<I...>) const {
        return m_tensor_(tensorwrapper::detail_::to_long(index.at(I))...);
    }

    eigen_data_type m_tensor_;
};

} // namespace tensorwrapper::buffer::detail_

#include "eigen_addition_assignment.ipp"
#include "eigen_contraction_assignment.ipp"
#include "eigen_hadamard_assignment.ipp"
#include "eigen_permute_assignment.ipp"
#include "eigen_scalar_multiplication.ipp"
#include "eigen_subtraction_assignment.ipp"