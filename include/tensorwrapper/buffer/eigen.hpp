#pragma once
#include <tensorwrapper/backends/eigen.hpp>
#include <tensorwrapper/buffer/replicated.hpp>
#include <variant>

namespace tensorwrapper::buffer {

template<typename FloatType>
class Eigen : public Replicated {
private:
    /// Type of *this
    using my_type = Eigen<FloatType>;

    /// Type *this derives from
    using base_type = Replicated;

public:
    /// Pull in base class's types
    using typename Replicated::const_layout_reference;

    template<unsigned short Rank>
    using tensor_type = eigen::tensor<FloatType, Rank>;

    template<unsigned short Rank>
    using tensor_reference = tensor_type<Rank>&;

    template<unsigned short Rank>
    using const_tensor_reference = const tensor_type<Rank>&;

    template<typename TensorType>
    Eigen(TensorType&& t, const_layout_reference layout) :
      Replicated(layout), m_tensor_(std::forward<TensorType>(t)) {}

    template<unsigned short Rank>
    auto& value() {
        return std::get<Rank>(m_tensor_);
    }

    template<unsigned short Rank>
    const auto& value() const {
        return std::get<Rank>(m_tensor_);
    }

private:
    using t0_t = tensor_type<0>;
    using t1_t = tensor_type<1>;
    using t2_t = tensor_type<2>;
    using t3_t = tensor_type<3>;
    using t4_t = tensor_type<4>;
    using t5_t = tensor_type<5>;
    using t6_t = tensor_type<6>;
    using t7_t = tensor_type<7>;

    std::variant<t0_t, t1_t, t2_t, t3_t, t4_t, t5_t, t6_t, t7_t> m_tensor_;
};

} // namespace tensorwrapper::buffer
