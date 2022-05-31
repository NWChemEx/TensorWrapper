#include "../ta_helpers/ta_helpers.hpp"
#include "tensorwrapper/tensor/conversions.hpp"

namespace tensorwrapper::tensor {

void to_contiguous_buffer(const ScalarTensorWrapper& t, double* buffer_begin,
                          const double* buffer_end) {
    const double* cbegin = buffer_begin;
    if(std::distance(cbegin, buffer_end) != t.size()) {
        throw std::runtime_error("to_contiguous_buffer requires buffer size to "
                                 "be exactly the size of the tensor");
    }

    // XXX: In principle we don't have to make things replicated explictly,
    // we can just grab pieces remotely via RPC
    auto t_ta = t.get<TA::TSpArrayD>();
    t_ta.make_replicated();

    // Have to use this range to compute ordinal index, using i_range will give
    // us the offset from the tile start
    auto erange = t_ta.elements_range();

    // XXX: This prohibits vectorization, should be looked into
    for(const auto& tile_i : t_ta) {
        const auto& i_range = tile_i.get().range();
        for(auto idx : i_range)
            buffer_begin[erange.ordinal(idx)] = tile_i.get()[idx];
    }
}

std::vector<double> to_vector(const ScalarTensorWrapper& t) {
    std::vector<double> rv(t.size(), 0.0);
    to_contiguous_buffer(t, rv.data(), rv.data() + rv.size());
    return rv;
}

ScalarTensorWrapper wrap_std_vector(std::vector<double> v) {
    using tensorwrapper::ta_helpers::array_from_vec;
    using tensorwrapper::ta_helpers::make_1D_trange;

    auto& world = TA::get_default_world();
    auto tr1    = make_1D_trange(v.size(), v.size());

    return ScalarTensorWrapper(array_from_vec(v, tr1, world));
}

} // namespace tensorwrapper::tensor
