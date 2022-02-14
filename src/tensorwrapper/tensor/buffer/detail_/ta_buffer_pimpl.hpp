#pragma once
#include "buffer_pimpl.hpp"
#include "tensorwrapper/tensor/detail_/backends/tiled_array.hpp"

namespace tensorwrapper::tensor::buffer::detail_ {

template<typename FieldType>
class TABufferPIMPL : public BufferPIMPL<FieldType> {
private:
    using my_type = TABufferPIMPL<FieldType>;

    using base_type = BufferPIMPL<FieldType>;

    using traits_type = tensor::backends::TiledArrayTraits<FieldType>;

    using variant_type = typename traits_type::variant_type;

public:
    using typename base_type::const_annotation_reference;

    using typename base_type::pimpl_pointer;

    using typename base_type::hasher_reference;

    using default_tensor_type =
      typename traits_type::template tensor_type<double>;

    using ta_shape_type = TA::SparseShape<float>;

    using ta_trange_type = TA::TiledRange;

    TABufferPIMPL(default_tensor_type t2wrap = {});

    void retile(ta_trange_type trange);

    void set_shape(ta_shape_type new_shape);

private:
    pimpl_pointer clone_() const override;

    void scale_(const_annotation_reference my_idx,
                const_annotation_reference out_idx, base_type& out,
                double rhs) const override;

    void add_(const_annotation_reference my_idx,
              const_annotation_reference out_idx, base_type& out,
              const_annotation_reference rhs_idx,
              const base_type& rhs) const override;

    void inplace_add_(const_annotation_reference my_idx,
                      const_annotation_reference rhs_idx,
                      const base_type& rhs) override;

    void subtract_(const_annotation_reference my_idx,
                   const_annotation_reference out_idx, base_type& out,
                   const_annotation_reference rhs_idx,
                   const base_type& rhs) const override;

    void inplace_subtract_(const_annotation_reference my_idx,
                           const_annotation_reference rhs_idx,
                           const base_type& rhs) override;

    void times_(const_annotation_reference my_idx,
                const_annotation_reference out_idx, base_type& out,
                const_annotation_reference rhs_idx,
                const base_type& rhs) const override;

    void hash_(hasher_reference h) const override;

    bool are_equal_(const base_type& rhs) const noexcept override;

    std::string to_str_() const override;

    variant_type m_tensor_;
};

extern template class TABufferPIMPL<field::Scalar>;
extern template class TABufferPIMPL<field::Tensor>;

} // namespace tensorwrapper::tensor::buffer::detail_