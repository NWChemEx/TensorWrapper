#pragma once
#include <tensorwrapper/tensor/fields.hpp>

namespace tensorwrapper::tensor::data_evaluator {

template<typename T>
struct TileEvaluator;

template<>
struct TileEvaluator<field::Scalar> {
    /// Type of a tensor index
    using index_t = std::vector<size_t>;

    /// Type of the scalars in the tensor
    using scalar_t = double;

    void operator()(index_t lo, index_t up, scalar_t* d) const;

    template<typename Archive>
    void serialize(Archive& ar);

protected:
    virtual void operator_(index_t lo, index_t up, scalar_t* d) const = 0;

    template<typename Archive>
    void serialize_(Archive& ar);
};

template<>
struct TileEvaluator<field::Tensor> {
    /// Type of a tensor index
    using index_t = std::vector<size_t>;

    /// Type of the scalars in the tensor
    using scalar_t = double;

    void operator()(index_t outer, index_t lo, index_t up, scalar_t* d) const;

    template<typename Archive>
    void serialize(Archive& ar);

protected:
    virtual void operator_(index_t outer, index_t lo, index_t up,
                           scalar_t* d) const = 0;

    template<typename Archive>
    void serialize_(Archive& ar);
};

//------------------------------------------------------------------------------
//                         Inline Implementations
//------------------------------------------------------------------------------

void TileEvaluator<field::Scalar>::operator()(index_t lo, index_t up,
                                              scalar_t* d) const {
    operator_(lo, up, d);
}

template<typename Archive>
void TileEvaluator<field::Scalar>::serialize(Archive& ar) {
    serialize_(ar);
}

void TileEvaluator<field::Tensor>::operator()(index_t outer, index_t lo,
                                              index_t up, scalar_t* d) const {
    operator_(outer, lo, up, d);
}

template<typename Archive>
void TileEvaluator<field::Tensor>::serialize(Archive& ar) {
    serialize_(ar);
}

} // namespace tensorwrapper::tensor::data_evaluator