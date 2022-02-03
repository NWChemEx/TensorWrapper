#pragma once
#include "tensorwrapper/tensor/fields.hpp"
#include <string>

namespace tensorwrapper::tensor::detail_ {

template<typename FieldType>
class BufferPIMPL {
private:
    /// Type of this instance
    using my_type = BufferPIMPL<FieldType>;

public:
    /// Type used for indices in einsum/index-based operations
    using annotation_type = std::string;

    /// Type of a read-only reference to an annotation
    using const_annotation_reference = const std::string&;

    virtual ~BufferPIMPL() noexcept = default;

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

    explicit operator std::string() const { return to_str_(); }

    bool are_equal(const my_type& rhs) const noexcept {
        return are_equal_(rhs) && rhs.are_equal_(*this);
    }

protected:
    BufferPIMPL() noexcept = default;

private:
    /// To be overriden by derived class to implement operator+
    virtual void add_(const_annotation_reference my_idx,
                      const_annotation_reference out_idx, my_type& out,
                      const_annotation_reference rhs_idx,
                      const my_type& rhs) const = 0;

    /// To be overriden by derived class to implement operator+=
    virtual void inplace_add_(const_annotation_reference my_idx,
                              const_annotation_reference rhs_idx,
                              const my_type& rhs) = 0;

    /// To be overriden by derived class to implement operator+
    virtual void subtract_(const_annotation_reference my_idx,
                           const_annotation_reference out_idx, my_type& out,
                           const_annotation_reference rhs_idx,
                           const my_type& rhs) const = 0;

    /// To be overriden by derived class to implement operator+=
    virtual void inplace_subtract_(const_annotation_reference my_idx,
                                   const_annotation_reference rhs_idx,
                                   const my_type& rhs) = 0;

    /// To be overriden by derived class to implement operator*
    virtual void times_(const_annotation_reference my_idx,
                        const_annotation_reference out_idx, my_type& out,
                        const_annotation_reference rhs_idx,
                        const my_type& rhs) const = 0;

    /// To be overriden by derived class to implement value equality
    virtual bool are_equal_(const my_type& rhs) const noexcept = 0;

    /// To be overriden by derived class to implement printing
    virtual std::string to_str_() const = 0;
};

template<typename FieldType>
std::ostream& operator<<(std::ostream& os, const BufferPIMPL<FieldType>& b) {
    return os << std::string(b);
}

} // namespace tensorwrapper::tensor::detail_
