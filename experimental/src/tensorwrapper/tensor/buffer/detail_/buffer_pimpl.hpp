/*
 * Copyright 2022 NWChemEx-Project
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

#pragma once
#include "tensorwrapper/tensor/buffer/buffer.hpp"
namespace tensorwrapper::tensor::buffer::detail_ {

template<typename FieldType>
class BufferPIMPL {
private:
    /// Type of this instance
    using my_type = BufferPIMPL<FieldType>;

    /// Type of the Buffer this PIMPL implements
    using buffer_type = Buffer<FieldType>;

public:
    /// Type of a read-only reference to an annotation
    using const_annotation_reference =
      typename buffer_type::const_annotation_reference;

    /// Type of a pointer to the PIMPL
    using pimpl_pointer = typename buffer_type::pimpl_pointer;

    /// Type of a read-only reference to the shape
    // using const_shape_reference = typename
    // buffer_type::const_shape_reference;

    /// Type of scalar values
    using scalar_value_type = typename buffer_type::scalar_value_type;

    /// Type of extents
    using extents_type = typename buffer_type::extents_type;

    /// Type of extents
    using inner_extents_type = typename buffer_type::inner_extents_type;

    /// Default constructs the derived class
    pimpl_pointer default_clone() const { return default_clone_(); }

    /// Deep polymorphic copy
    pimpl_pointer clone() const { return clone_(); }

    virtual ~BufferPIMPL() noexcept = default;

    void permute(const_annotation_reference my_idx,
                 const_annotation_reference out_idx, my_type& out) const {
        permute_(my_idx, out_idx, out);
    }

    /** @brief Implements operator*(double)
     *
     */
    void scale(const_annotation_reference my_idx,
               const_annotation_reference out_idx, my_type& out,
               double rhs) const {
        scale_(my_idx, out_idx, out, rhs);
    }

    /** @brief Implements operator+
     *
     */
    void add(const_annotation_reference my_idx,
             const_annotation_reference out_idx, my_type& out,
             const_annotation_reference rhs_idx, const my_type& rhs) const {
        add_(my_idx, out_idx, out, rhs_idx, rhs);
    }

    /** @brief Implements operator+=
     *
     */
    void inplace_add(const_annotation_reference my_idx,
                     const_annotation_reference rhs_idx, const my_type& rhs) {
        inplace_add_(my_idx, rhs_idx, rhs);
    }

    /** @brief Implements operator-
     *
     */
    void subtract(const_annotation_reference my_idx,
                  const_annotation_reference out_idx, my_type& out,
                  const_annotation_reference rhs_idx,
                  const my_type& rhs) const {
        subtract_(my_idx, out_idx, out, rhs_idx, rhs);
    }

    /** @brief Implements operator-=
     *
     */
    void inplace_subtract(const_annotation_reference my_idx,
                          const_annotation_reference rhs_idx,
                          const my_type& rhs) {
        inplace_subtract_(my_idx, rhs_idx, rhs);
    }

    /** @brief Implements operator*
     *
     */
    void times(const_annotation_reference my_idx,
               const_annotation_reference out_idx, my_type& out,
               const_annotation_reference rhs_idx, const my_type& rhs) const {
        times_(my_idx, out_idx, out, rhs_idx, rhs);
    }

    scalar_value_type dot(const_annotation_reference my_idx,
                          const_annotation_reference rhs_idx,
                          const my_type& rhs) const {
        return dot_(my_idx, rhs_idx, rhs);
    }

    /// Implements norm operation
    inline scalar_value_type norm() const { return norm_(); }

    /// Implements element sum operation
    inline scalar_value_type sum() const { return sum_(); }

    /// Implements trace operation
    inline scalar_value_type trace() const { return trace_(); }

    /// Implements making extents
    inline extents_type make_extents() const { return make_extents_(); }

    /// Implements making extents
    inline inner_extents_type make_inner_extents() const {
        return make_inner_extents_();
    }

    explicit operator std::string() const { return to_str_(); }

    bool are_equal(const my_type& rhs) const noexcept {
        return are_equal_(rhs) && rhs.are_equal_(*this);
    }

protected:
    /// These are protected to avoid users accidentally slicing the PIMPL, but
    /// still be accesible to derived classes who need them for implementations
    ///@{
    BufferPIMPL() noexcept                     = default;
    BufferPIMPL(const BufferPIMPL&)            = default;
    BufferPIMPL(BufferPIMPL&&)                 = default;
    BufferPIMPL& operator=(const BufferPIMPL&) = default;
    BufferPIMPL& operator=(BufferPIMPL&&)      = default;
    ///@}

private:
    /// To be overriden by derived class to implement default_clone
    virtual pimpl_pointer default_clone_() const = 0;

    /// To be overridden by derived class to implement clone
    virtual pimpl_pointer clone_() const = 0;

    /// To be overridden by derived class to implement permute
    virtual void permute_(const_annotation_reference my_idx,
                          const_annotation_reference out_idx,
                          my_type& out) const = 0;

    /// To be overridden by derived class to implement operator*(double)
    virtual void scale_(const_annotation_reference my_idx,
                        const_annotation_reference out_idx, my_type& out,
                        double rhs) const = 0;

    /// To be overridden by derived class to implement operator+
    virtual void add_(const_annotation_reference my_idx,
                      const_annotation_reference out_idx, my_type& out,
                      const_annotation_reference rhs_idx,
                      const my_type& rhs) const = 0;

    /// To be overridden by derived class to implement operator+=
    virtual void inplace_add_(const_annotation_reference my_idx,
                              const_annotation_reference rhs_idx,
                              const my_type& rhs) = 0;

    /// To be overridden by derived class to implement operator+
    virtual void subtract_(const_annotation_reference my_idx,
                           const_annotation_reference out_idx, my_type& out,
                           const_annotation_reference rhs_idx,
                           const my_type& rhs) const = 0;

    /// To be overridden by derived class to implement operator+=
    virtual void inplace_subtract_(const_annotation_reference my_idx,
                                   const_annotation_reference rhs_idx,
                                   const my_type& rhs) = 0;

    /// To be overridden by derived class to implement operator*
    virtual void times_(const_annotation_reference my_idx,
                        const_annotation_reference out_idx, my_type& out,
                        const_annotation_reference rhs_idx,
                        const my_type& rhs) const = 0;

    virtual scalar_value_type dot_(const_annotation_reference my_idx,
                                   const_annotation_reference rhs_idx,
                                   const my_type& rhs) const = 0;

    /// To be overridden by derived class to implement norm
    virtual scalar_value_type norm_() const = 0;

    /// To be overridden by derived class to implement element sum
    virtual scalar_value_type sum_() const = 0;

    /// To be overridden by derived class to implement trace
    virtual scalar_value_type trace_() const = 0;

    /// To be overridden by derived class to implement make_extents
    virtual extents_type make_extents_() const = 0;

    /// To be overridden by derived class to implement make_inner_extents
    virtual inner_extents_type make_inner_extents_() const = 0;

    /// To be overriden by derived class to implement value equality
    virtual bool are_equal_(const my_type& rhs) const noexcept = 0;

    /// To be overriden by derived class to implement printing
    virtual std::string to_str_() const = 0;
};

template<typename FieldType>
std::ostream& operator<<(std::ostream& os, const BufferPIMPL<FieldType>& b) {
    return os << std::string(b);
}

} // namespace tensorwrapper::tensor::buffer::detail_
