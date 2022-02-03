#pragma once
#include "buffer_pimpl.hpp"
#include "tensorwrapper/tensor/detail_/backends/tiled_array.hpp"

namespace tensorwrapper::tensor::detail_ {

template<typename FieldType>
class TABufferPIMPL : public BufferPIMPL<FieldType> {
private:
    using my_type = TABufferPIMPL<FieldType>;

    using base_type = BufferPIMPL<FieldType>;

    using traits_type = tensor::backends::TiledArrayTraits<FieldType>;

    using variant_type = typename traits_type::variant_type;

public:
    using typename base_type::const_annotation_reference;

    using default_tensor_type = typename traits_type::tensor_type<double>;

    TABufferPIMPL(default_tensor_type t2wrap = {});

private:
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

    bool are_equal_(const base_type& rhs) const noexcept override;

    std::string to_str_() const override;

    variant_type m_tensor_;
};

extern template class TABufferPIMPL<field::Scalar>;
extern template class TABufferPIMPL<field::Tensor>;

} // namespace tensorwrapper::tensor::detail_
