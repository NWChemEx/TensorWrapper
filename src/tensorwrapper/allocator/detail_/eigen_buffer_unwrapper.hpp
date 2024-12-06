#pragma once
#include <tensorwrapper/allocator/eigen.hpp>
#include <utilities/type_traits/variant/cat.hpp>

namespace tensorwrapper::allocator::detail_ {

struct EigenBufferUnwrapper {
    /// The maximum Eigen rank we are considering
    static constexpr std::size_t max_rank = 10;

    /// Type giving us the numbers [0, max_rank) in a parameter pack
    using sequence_type = std::make_index_sequence<max_rank>;

    /// Type of rank @p I buffer::Eigen object with @p FloatType elements
    template<typename FloatType, std::size_t I>
    using buffer_type = buffer::Eigen<FloatType, I>;

    template<typename FloatType, std::size_t... I>
    static auto dummy(std::index_sequence<I...>)
      -> std::variant<buffer_type<FloatType, I>...>;

    template<typename... Args>
    using type_ = utilities::type_traits::variant::cat_t<decltype(dummy<Args>(
      std::declval<sequence_type>()))...>;

    using type = type_<float, double>;

    template<typename BufferType>
    static type unwrap(BufferType&& buffer) {
        return unwrap_<0>(std::forward<BufferType>(buffer));
    }

    template<unsigned short Rank, typename BufferType>
    static type unwrap_(BufferType&& buffer) {
        if constexpr(Rank == max_rank) {
            throw std::runtime_error("Please increase max_rank");
        } else {
            if(buffer.layout().shape().rank() == Rank) {
                using eigen_type = buffer_type<double, Rank>;
                auto pbuffer     = dynamic_cast<eigen_type*>(&buffer());
                if(pbuffer == nullptr)
                    throw std::runtime_error("Not convertible to Eigen");
                return type{*pbuffer};
            }
            return unwrap_<Rank + 1>(std::forward<BufferType>(buffer));
        }
    }
};

} // namespace tensorwrapper::allocator::detail_