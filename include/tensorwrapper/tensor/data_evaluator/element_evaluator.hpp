#pragma once
#pragma once
#include <tensorwrapper/tensor/fields.hpp>

namespace tensorwrapper::tensor::data_evaluator {

template<typename T>
struct ElementEvaluator;

template<>
struct ElementEvaluator<field::Scalar> {
    /// Type of a tensor index
    using index_t = std::vector<size_t>;

    /// Type of the scalars in the tensor
    using scalar_t = double;

    void operator()(index_t idx) const;

    template<typename Archive>
    void serialize(Archive& ar);

protected:
    virtual void operator_(index_t idx) const = 0;

    template<typename Archive>
    void serialize_(Archive& ar);
};

template<>
struct ElementEvaluator<field::Tensor> {
    /// Type of a tensor index
    using index_t = std::vector<size_t>;

    /// Type of the scalars in the tensor
    using scalar_t = double;

    void operator()(index_t outer, index_t inner) const;

    template<typename Archive>
    void serialize(Archive& ar);

protected:
    virtual void operator_(index_t outer, index_t inner) const = 0;

    template<typename Archive>
    void serialize_(Archive& ar);
};

//------------------------------------------------------------------------------
//                         Inline Implementations
//------------------------------------------------------------------------------

void ElementEvaluator<field::Scalar>::operator()(index_t idx) const {
    operator_(idx);
}

template<typename Archive>
void ElementEvaluator<field::Scalar>::serialize(Archive& ar) {
    serialize_(ar);
}

void ElementEvaluator<field::Tensor>::operator()(index_t outer,
                                                 index_t inner) const {
    operator_(outer, inner);
}

template<typename Archive>
void ElementEvaluator<field::Tensor>::serialize(Archive& ar) {
    serialize_(ar);
}

} // namespace tensorwrapper::tensor::data_evaluator