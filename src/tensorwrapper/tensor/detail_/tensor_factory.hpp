#pragma once
#include <memory>
#include <tensorwrapper/tensor/tensor_class.hpp>
namespace tensorwrapper::detail_ {

/** @brief Object which helps create tensor objects.
 *
 *  Ultimately there are going to be a lot of possible ways to create a tensor.
 *  In an effort to keep the Tensor's ctors as simple as possible we have
 *  opted to have the actual construction be done by a different class, the
 *  TensorFactory class.
 *
 */
class TensorFactory {
public:
    /// Type this will create PIMPLs for
    using tensor_type = Tensor;

    /// Type used to manage the myriad of possible inputs
    using input_type = TensorInput;

    /// Type powering an object of type tensor_type
    using pimpl_type = typename tensor_type::pimpl_type;

    /// Type of a pointer to an object of type pimpl_type
    using pimpl_pointer = typename tensor_type::pimpl_pointer;

    // Pull types from input_type that we will need for our API
    using const_shape_reference    = input_type::const_shape_reference;
    using shape_pointer            = input_type::shape_pointer;
    using const_symmetry_reference = input_type::const_symmetry_reference;
    using symmetry_pointer         = input_type::symmetry_pointer;
    using const_sparsity_reference = input_type::const_sparsity_reference;
    using sparsity_pointer         = input_type::sparsity_pointer;
    using const_logical_reference  = input_type::const_logical_reference;
    using logical_layout_pointer   = input_type::logical_layout_pointer;
    using const_physical_reference = input_type::const_physical_reference;
    using physical_layout_pointer  = input_type::physical_layout_pointer;
    using allocator_pointer        = input_type::allocator_pointer;
    using runtime_view_type        = input_type::runtime_view_type;

    // -------------------------------------------------------------------------
    // -- Methods for determining reasonable defaults
    // -------------------------------------------------------------------------

    static symmetry_pointer default_logical_symmetry(
      const_shape_reference shape);

    static sparsity_pointer default_logical_sparsity(
      const_shape_reference shape, const_symmetry_reference symmetry);

    static logical_layout_pointer default_logical_layout(
      shape_pointer shape, symmetry_pointer symmetry,
      sparsity_pointer sparsity);

    static physical_layout_pointer default_physical_layout(
      const_logical_reference logical);

    static allocator_pointer default_allocator(
      const_physical_reference physical, runtime_view_type rv);

    static pimpl_pointer construct(input_type input);
};

} // namespace tensorwrapper::detail_
