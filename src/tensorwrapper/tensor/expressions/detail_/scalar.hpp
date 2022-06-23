// #pragma once
// #include "unary.hpp"
// #include <stdexcept>

// namespace tensorwrapper::tensor::expressions::detail_ {

// class Scalar : public Unary<Scalar, double> {
// private:
//     using base_type = Unary<Scalar, double>;

// public:
//     using labeled_tensor = typename base_type::labeled_tensor;
//     using labeled_tot    = typename base_type::labeled_tot;

//     using base_type::Unary;

// protected:
//     double eval_() const override { return m_value_; }
//     labeled_tensor& eval_(labeled_tensor& result) const override;
//     labeled_tot& eval_(labeled_tot& result) const override;

// private:
//     double m_value_;
// };

// inline Scalar::labeled_tensor& Scalar::eval_(const labeled_tensor&) const{
//     throw std::runtime_error("Can not evaluate a scalar to a tensor");
// }

// inline Scalar::labeled_tot& Scalar::eval_(const labeled_tot&) const {
//     throw std::runtime_error("Can not evaluate a scalar to a ToT");

// } // namespace tensorwrapper::tensor::expressions::detail_
