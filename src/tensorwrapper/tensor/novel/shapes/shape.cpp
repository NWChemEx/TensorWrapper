#include "detail_/shape_pimpl.hpp"

namespace tensorwrapper::tensor::novel {
namespace {

template<typename FieldType, typename... Args>
auto make_pimpl(Args&&... args) {
    using return_type = detail_::ShapePIMPL<FieldType>;
    return std::make_unique<return_type>(std::forward<Args>(args)...);
}

} // namespace

#define SHAPE Shape<FieldType>

//------------------------------------------------------------------------------
//                            CTors, DTor, and Assignment
//------------------------------------------------------------------------------

template<typename FieldType>
SHAPE::Shape() noexcept = default;

template<typename FieldType>
SHAPE::Shape(extents_type extents, inner_extents_type inner_extents) :
  Shape(make_pimpl<FieldType>(std::move(extents), std::move(inner_extents))) {}

template<typename FieldType>
SHAPE::~Shape() noexcept = default;

template<typename FieldType>
SHAPE::Shape(const Shape& other) :
  Shape(other.m_pimpl_ ? other.m_pimpl_->clone() : nullptr) {}

template<typename FieldType>
SHAPE::Shape(Shape&& other) noexcept : Shape(std::move(other.m_pimpl_)) {}

template<typename FieldType>
Shape<FieldType>& SHAPE::operator=(const Shape& other) {
    m_pimpl_ = other.m_pimpl_->clone();
    return *this;
}

template<typename FieldType>
Shape<FieldType>& SHAPE::operator=(Shape&& other) noexcept {
    m_pimpl_ = std::move(other.m_pimpl_->clone());
    return *this;
}

//------------------------------------------------------------------------------
//                               Accessors
//------------------------------------------------------------------------------

template<typename FieldType>
typename SHAPE::const_extents_reference SHAPE::extents() const {
    assert_pimpl_();
    return m_pimpl_->extents();
}

template<typename FieldType>
typename SHAPE::const_inner_extents_reference SHAPE::inner_extents() const {
    assert_pimpl_();
    return m_pimpl_->inner_extents();
}

template<typename FieldType>
typename SHAPE::size_type SHAPE::field_rank() const {
    // Short circuit for Scalar
    if constexpr(field::is_scalar_field_v<FieldType>) return 0;

    assert_pimpl_();
    return m_pimpl_->field_rank();
}

//------------------------------------------------------------------------------
//                           Utility Functions
//------------------------------------------------------------------------------

template<typename FieldType>
bool SHAPE::operator==(const Shape& rhs) const noexcept {
    if(m_pimpl_ && rhs.m_pimpl_)
        return pimpl_() == rhs.pimpl_();
    else if(!m_pimpl_ && !rhs.m_pimpl_)
        return true;
    return false; // One has a PIMPL other doesn't
}

template<typename FieldType>
void SHAPE::hash(tensorwrapper::detail_::Hasher& h) const {
    h(m_pimpl_);
}

template<typename FieldType>
bool SHAPE::is_equal(const Shape& rhs) const noexcept {
    // Needs to be symmetric to engage the most derived class of each instance
    if(is_equal_(rhs) && rhs.is_equal_(*this)) return true;
    return false;
}

//------------------------------------------------------------------------------
//                      Protected/Private Functions
//------------------------------------------------------------------------------
template<typename FieldType>
bool SHAPE::is_hard_zero_(const index_type& /*i*/) const {
    return false;
}

template<typename FieldType>
bool SHAPE::is_hard_zero_(const index_type& /*lo*/,
                          const index_type& /*hi*/) const {
    return false;
}

template<typename FieldType>
SHAPE::Shape(pimpl_pointer pimpl) noexcept : m_pimpl_(std::move(pimpl)) {}

template<typename FieldType>
const typename SHAPE::pimpl_type& SHAPE::pimpl_() const {
    assert_pimpl_();
    return *m_pimpl_;
}

template<typename FieldType>
bool SHAPE::has_pimpl_() const noexcept {
    return m_pimpl_.operator bool();
}

template<typename FieldType>
void SHAPE::assert_pimpl_() const {
    if(has_pimpl_()) return;
    throw std::runtime_error("Instance does not contain a PIMPL. Did you "
                             "default construct it or move from it?");
}

template<typename FieldType>
typename SHAPE::pointer_type SHAPE::clone_() const {
    if(has_pimpl_())
        return pointer_type(
          new Shape(pimpl_().extents(), pimpl_().inner_extents()));
    return pointer_type(new Shape());
}

template<typename FieldType>
bool SHAPE::is_equal_(const Shape<FieldType>& rhs) const noexcept {
    return *this == rhs;
}

#undef SHAPE

template class Shape<field::Scalar>;
template class Shape<field::Tensor>;

} // namespace tensorwrapper::tensor::novel
