#pragma once

#include "pimpl.hpp"

namespace tensorwrapper::tensor::expression::detail_ {

/** @brief Code factorization for implementing ExpressionPIMPL derived classes
 *
 *  Implementing a new derived class from ExpressionPIMPL requires overriding
 *  several virtual methods. The implementations for many of these methods are
 *  largely boilerplate. Using the curiously recursive template pattern, the
 *  NNary class implements these methods for the derived class.
 *
 *  The name NNary comes from the generalizaton of unary, binary, trinary, etc.
 *  and refers to the fact that this class can handle implementing expressions
 *  involving an arbitrary number of arguments (although in practice only
 *  unary and binary versions are presently encountered, it was easy enough to
 *  write this class in a fully general manner).
 *
 *  @tparam FieldType A strong type denoting whether the elements of the
 *                    associated tensors are scalars or other tensors. Expected
 *                    to be either field::Scalar or field::Tensor.
 *  @tparam DerivedType This is the type of the class (including any template
 *                      parameters) being implemented by NNary. For example for
 *                      the Add template class this would be Add<FieldType>.
 *  @tparam Args... A variadiac template parameter pack whose members are the
 *                  types of the pieces of the derived expression. For example
 *                  a binary expression which combines two Expression instances
 *                  would set Args to Expression, Expression.
 */
template<typename FieldType, typename DerivedType, typename... Args>
class NNary : public ExpressionPIMPL<FieldType> {
private:
    /// Type of *this
    using my_type = NNary<FieldType, DerivedType, Args...>;

    /// Type of the class *this derives from
    using base_type = ExpressionPIMPL<FieldType>;

public:
    /// Type of a pointer to an ExpressionPIMPL instance. Ultimately a typedef
    /// of Expression::pimpl_pointer.
    using typename base_type::pimpl_pointer;

    /** @brief Creates a new NNary instance with the provided expression pieces
     *
     *  @tparam ArgsIn A template parameter pack whose types are the types of
     *                 the expression pieces. The i-th type in ArgsIn must be
     *                 implicitly convertible to the i-th type in Args.
     *
     *  @param[in] args The pieces of the expression, e.g., for a binary
     *                  expression these would be the expressions on the left
     *                  and right of the operator.
     *
     *  @throws ??? Throws if forwarding the inputs into a std::tuple<Args...>
     *              throws. Under normal circumstances this is most likely to be
     *              caused by a bad allocation occurring when copying one of the
     *              inputs. Moving inputs is no throw guarantee. If an exception
     *              is thrown this method has the same exception safety as the
     *              underlying ctor.
     */
    template<typename... ArgsIn>
    explicit NNary(ArgsIn&&... args) : m_args_(std::forward<ArgsIn>(args)...) {}

    /** @brief Access the @p I -th argument in the expression
     *
     *  A N-nary expression combines N sub-expressions. This method is used to
     *  access the sub-expressions by offset.
     *
     *  @tparam I The offset of the argument. I must be in the range
     *            [0, sizeof...(Args)) or a compiler error will result.
     *
     *
     *  @return A read-only reference to the requested argument.
     *
     *  @throw None No throw guarantee.
     */
    template<std::size_t I>
    const auto& arg() const {
        return std::get<I>(m_args_);
    }

protected:
    /// Implements clone() by dispatching to the derived class's copy ctor
    pimpl_pointer clone_() const override;

    /** @brief  Implements are_equal
     *
     *  This method is implemented by attempting to downcast rhs to my_type. If
     *  the cast succeeds, the m_args_ members are then compared. Of note, this
     *  means that if the derived class contains additional state not exposed to
     *  the NNary base class, the derived class must also override are_equal.
     *
     *  @param[in] rhs The expression we are polymorphically comparing to.
     *
     *  @return True if @p rhs can be downcast to `my_type` and if this->m_args_
     *          compares equal to rhs.m_args_. False otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool are_equal_(const base_type& rhs) const noexcept override;

private:
    /// Downcasts to a read/write instance of the derived class
    DerivedType& downcast_();

    /// Downcasts to a read-only instance of the derived class
    const DerivedType& downcast_() const;

    /// These are the `n_args` arguments used to construct the expression
    std::tuple<Args...> m_args_;
};

/// Specialization of NNary used to implement Labeled
template<typename FieldType, typename DerivedType>
using LabeledBase = NNary<FieldType, DerivedType, LabeledView<FieldType>>;

/// Specialization of NNary used to implement Add, Subtract, and Times
template<typename FieldType, typename DerivedType>
using Binary =
  NNary<FieldType, DerivedType, Expression<FieldType>, Expression<FieldType>>;

/// Specialization of NNary used to implement Scale
template<typename FieldType, typename DerivedType, typename ScalarType>
using ScaleBase =
  NNary<FieldType, DerivedType, Expression<FieldType>, ScalarType>;

} // namespace tensorwrapper::tensor::expression::detail_

#include "nnary.ipp"
