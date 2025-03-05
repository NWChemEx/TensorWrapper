#pragma once
#include <tensorwrapper/allocator/eigen.hpp>
#include <tensorwrapper/buffer/buffer_base.hpp>

namespace tensorwrapper::utilities {

/** @brief Wraps the logic needed to work out the floating point type of buffer.
 *
 *  @tparam KernelType Type of a functor. The functor must define a function
 *                     template called `run` that takes one explicit template
 *                     type parameter (will be the floating point type of @p
 *                     buffer) and @p buffer. `run` may take an arbitrary amount
 *                     of additional arguments.
 *  @tparam BufferType The type of @p buffer. Must be derived from BufferBase.
 *                     May contain cv or reference qualifiers.
 *  @tparam Args The types of any additional arguments which will be forwarded
 *               to @p kernel.
 *
 *  @param[in] kernel The functor instance to call `run` on.
 *  @param[in] buffer The type of the elements in @p buffer will be used to
 *                    dispatch.
 *  @param[in] args   Any additional arguments to forward to @p kernel.
 *
 *  @return Returns whatever @p kernel returns.
 *
 *  @throw std::runtime_error if @p buffer is not derived from
 */
template<typename KernelType, typename BufferType, typename... Args>
decltype(auto) floating_point_dispatch(KernelType&& kernel, BufferType&& buffer,
                                       Args&&... args) {
    using buffer_clean       = std::decay_t<BufferType>;
    using buffer_base        = buffer::BufferBase;
    constexpr bool is_buffer = std::is_base_of_v<buffer_base, buffer_clean>;
    static_assert(is_buffer);

    using types::udouble;
    using types::ufloat;

    if(allocator::Eigen<float>::can_rebind(buffer)) {
        return kernel.template run<float>(buffer, std::forward<Args>(args)...);
    } else if(allocator::Eigen<double>::can_rebind(buffer)) {
        return kernel.template run<double>(buffer, std::forward<Args>(args)...);
    } else if(allocator::Eigen<ufloat>::can_rebind(buffer)) {
        return kernel.template run<ufloat>(buffer, std::forward<Args>(args)...);
    } else if(allocator::Eigen<udouble>::can_rebind(buffer)) {
        return kernel.template run<udouble>(buffer,
                                            std::forward<Args>(args)...);
    } else {
        throw std::runtime_error("Can't rebind buffer to Contiguous<>");
    }
}

} // namespace tensorwrapper::utilities