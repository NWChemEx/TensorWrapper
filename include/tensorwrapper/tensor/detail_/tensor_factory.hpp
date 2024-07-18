#pragma once
#include <memory>
#include <parallelzone/parallelzone.hpp>
#include <tensorwrapper/allocator/allocator_base.hpp>
#include <tensorwrapper/layout/layout_base.hpp>
#include <tensorwrapper/layout/logical.hpp>
#include <tensorwrapper/layout/physical.hpp>

namespace tensorwrapper::detail_ {
class TensorPIMPL;

/** @brief Object which helps create tensor objects.
 *
 *  Ultimately there are going to be a lot of possible ways to create a tensor.
 *  In an effort to keep the Tensor's ctors as simple as possible we have
 *  opted to have the actual construction be done by a different class, the
 *  TensorFactory class.
 *
 */
class TensorFactory {
private:
    /// Type common to all layouts (used to get types of shape, sparsity, etc.)
    using layout_base = layout::LayoutBase;

public:
    /// Type of the object implementing tensor
    using pimpl_type = TensorPIMPL;

    /// Type of a pointer to an object of type pimpl_type
    using pimpl_pointer = std::unique_ptr<pimpl_type>;

    /// Type all shapes inherit from
    using shape_base = typename layout_base::shape_base;

    /// Type of a pointer to an object of type shape_base
    using shape_pointer = typename shape_base::base_pointer;

    /// Type of a symmetry object
    using symmetry_base = typename layout_base::symmetry_type;

    /// Type of a pointer to an object of type symmetry_type
    using symmetry_pointer = std::unique_ptr<symmetry_base>;

    /// Type all sparsity patterns inherit from
    using sparsity_base = typename layout_base::sparsity_type;

    /// Type of a pointer to an object of type sparsity_base
    using sparsity_pointer = std::unique_ptr<sparsity_base>;

    /// Type all logical layouts inherit from
    using logical_layout_type = layout::Logical;

    /// Type of a pointer to an object of type logical_layout_type
    using logical_layout_pointer = std::unique_ptr<logical_layout_type>;

    /// Type all physical layouts inherit from
    using physical_layout_type = layout::Physical;

    /// Type of a pointer to an object of type physical_layout_type
    using physical_layout_pointer = std::unique_ptr<physical_layout_type>;

    /// Type all allocators inherit from
    using allocator_base = allocator::AllocatorBase;

    /// Type all buffer object's inherit from
    using buffer_base = typename allocator_base::buffer_base_type;

    /// Type of a pointer to an object of type buffer_base
    using buffer_pointer = typename buffer_base::base_pointer;

    /// Type of a view of the runtime
    using runtime_view_type = typename allocator_base::runtime_view_type;

    explicit TensorFactory(runtime_view_type rv) : m_rv_(std::move(rv)) {}

    /// Constructs a default initialized tensor
    pimpl_pointer construct() { return construct(shape_pointer{}); }

    pimpl_pointer construct(shape_pointer pshape) {
        return construct(std::move(pshape), nullptr, nullptr);
    }

    pimpl_pointer construct(shape_pointer pshape, symmetry_pointer psymmetry,
                            sparsity_pointer psparsity) {
        if(pshape == nullptr) return construct(logical_layout_pointer{});
        auto symm   = (psymmetry != nullptr) ? *psymmetry : symmetry_base{};
        auto sparse = (psparsity != nullptr) ? *psparsity : sparsity_base{};

        return construct(std::make_unique<logical_layout_type>(
          std::move(pshape), symm, sparse));
    }

    /** @brief User-friendly construct method.
     *
     *  Ideally, given the logical layout, TensorWrapper can decide on the
     *  optimal physical layout. This overload is thus the dispatch
     *  point we want users to target.
     */
    pimpl_pointer construct(logical_layout_pointer plogical);

    /** @brief Expert construct method.
     *
     *  Until the user-friendly construct method works well in all cases,
     * users will likely need to specify not only the logical layout, but the
     *  physical layout too. That's where this overload comes in. This
     * overload ultimately wraps the process of mapping the physical layout to
     * the ideal backend.
     *
     */
    pimpl_pointer construct(logical_layout_pointer plogical,
                            physical_layout_pointer pphysical);

    /** @brief Full-control method.
     *
     *  In addition to the all of the input of the expert construct method,
     *  this method also takes the type-erased result. Meaning the user has now
     *  even selected the backend (and set it up however they want). This
     *  method serves only to wrap the process of creating the pimpl_pointer
     *  object.
     */
    pimpl_pointer construct(logical_layout_pointer plogical,
                            buffer_pointer pbuffer);

private:
    runtime_view_type m_rv_;
};

/** @brief Common templated entry point into the factory.
 *
 *  If we ever decide we want TensorFactory to be polymorphic we can't rely on
 *  a static construct method. To wrap the process of creating a factory and
 *  then calling the construct method we introduce this function. Eventually the
 *  logic for which factory to construct could live here too.
 */
template<typename... Args>
auto construct(Args&&... args) {
    using runtime_view_type = typename TensorFactory::runtime_view_type;
    TensorFactory factory(runtime_view_type{});
    return factory.construct(std::forward<Args>(args)...);
}

} // namespace tensorwrapper::detail_
