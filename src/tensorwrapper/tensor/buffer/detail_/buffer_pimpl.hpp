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
    //using const_shape_reference = typename buffer_type::const_shape_reference;

    /// Type of a mutable hasher reference
    using hasher_reference = typename buffer_type::hasher_reference;

    /// Default constructs the derived class
    pimpl_pointer default_clone() const { return default_clone_(); }

    /// Deep polymorphic copy
    pimpl_pointer clone() const { return clone_(); }

    virtual ~BufferPIMPL() noexcept = default;

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

    void hash(hasher_reference h) const { hash_(h); }

    explicit operator std::string() const { return to_str_(); }

    bool are_equal(const my_type& rhs) const noexcept {
        return are_equal_(rhs) && rhs.are_equal_(*this);
    }

protected:
    /// These are protected to avoid users accidentally slicing the PIMPL, but
    /// still be accesible to derived classes who need them for implementations
    ///@{
    BufferPIMPL() noexcept          = default;
    BufferPIMPL(const BufferPIMPL&) = default;
    BufferPIMPL(BufferPIMPL&&)      = default;
    BufferPIMPL& operator=(const BufferPIMPL&) = default;
    BufferPIMPL& operator=(BufferPIMPL&&) = default;
    ///@}

private:
    /// To be overriden by derived class to implement default_clone
    virtual pimpl_pointer default_clone_() const = 0;

    /// To be overridden by derived class to implement clone
    virtual pimpl_pointer clone_() const = 0;

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

    /// To be overridden by derived classs to implement hash
    virtual void hash_(hasher_reference h) const = 0;

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
