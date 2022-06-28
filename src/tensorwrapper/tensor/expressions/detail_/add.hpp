// #pragma once
// #include "binary.hpp"
// #include <tensorwrapper/tensor/tensor_wrapper.hpp>

// namespace tensorwrapper::tensor::expressions::detail_ {

// class Add : public Binary<Add, Expression, Expression> {
// private:
//     using base_type = Binary<Add, Expression, Expression>;

// public:
//     using typename base_type::expression_type;
//     using typename base_type::labeled_tensor;
//     using typename base_type::labeled_tot;
//     using typename base_type::pimpl_pointer;

//     using base_type::Binary;

//     template<typename T>
//     T& eval_common(T& result) const;
// };

// template<typename T>
// T& Add::eval_common(T& result) const {
//     T temp_l(result), temp_r(result);
//     temp_l = m_lhs_.eval(temp_l);
//     temp_r = m_rhs_.eval(temp_r);

//     const auto& result_labels = result.labels();
//     const auto& l_labels      = temp_l.labels();
//     const auto& r_labels      = temp_r.labels();

//     auto& result_buffer = result.tensor().buffer();
//     const auto& lbuffer = temp_l.tensor().buffer();
//     const auto& rbuffer = temp_r.tensor().buffer();

//     lbuffer.add(l_labels, result_labels, result_buffer, r_labels, rbuffer);

//     return result;
// }

// } // namespace tensorwrapper::tensor::expressions::detail_
