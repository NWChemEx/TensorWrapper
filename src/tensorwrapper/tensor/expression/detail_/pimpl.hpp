#pragma once
#include <stdexcept>
#include <tensorwrapper/tensor/expression/expression_class.hpp>
#include <tensorwrapper/tensor/tensor_wrapper.hpp>

namespace tensorwrapper::tensor::expression::detail_ {

template<typename FieldType>
class ExpressionPIMPL {
private:
    using pt = Expression<FieldType>;

public:
    using const_label_reference     = typename pt::const_label_reference;
    using const_allocator_reference = typename pt::const_allocator_reference;
    using const_shape_reference     = typename pt::const_shape_reference;
    using tensor_type               = typename pt::tensor_type;
    using pimpl_pointer             = typename pt::pimpl_pointer;

    ExpressionPIMPL() noexcept          = default;
    virtual ~ExpressionPIMPL() noexcept = default;

    pimpl_pointer clone() const { return clone_(); }

    tensor_type tensor(const_label_reference labels,
                       const_shape_reference shape,
                       const_allocator_reference alloc) const {
        return tensor_(labels, shape, alloc);
    }

    bool are_equal(const ExpressionPIMPL& rhs) const noexcept {
        return are_equal_(rhs) && rhs.are_equal_(*this);
    }

protected:
    ExpressionPIMPL(const ExpressionPIMPL& other) = default;
    ExpressionPIMPL(ExpressionPIMPL&& other)      = default;

    virtual pimpl_pointer clone_() const                               = 0;
    virtual tensor_type tensor_(const_label_reference labels,
                                const_shape_reference shape,
                                const_allocator_reference alloc) const = 0;
    virtual bool are_equal_(const ExpressionPIMPL& rhs) const noexcept = 0;

private:
    ExpressionPIMPL& operator=(const ExpressionPIMPL& other) = delete;
};

} // namespace tensorwrapper::tensor::expression::detail_
