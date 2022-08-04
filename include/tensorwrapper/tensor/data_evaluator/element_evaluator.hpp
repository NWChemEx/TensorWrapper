#pragma once
#include <parallelzone/archive_wrapper.hpp>
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

    /// Type of an output serializer
    using serializer_t = parallelzone::Serializer;

    /// Type of an input deserializer
    using deserializer_t = parallelzone::Deserializer;

    void operator()(index_t idx) const;

    void serialize(serializer_t& ar);

    void deserialize(deserializer_t& ar);

protected:
    ElementEvaluator() = delete;
    // ElementEvaluator(const ElementEvaluator&) = delete;
    // ElementEvaluator(ElementEvaluator&&) = delete;
    // ElementEvaluator& operator=(const ElementEvaluator&) = delete;
    // ElementEvaluator& operator=(ElementEvaluator&&) = delete;

    virtual void operator_(index_t idx) const = 0;

    virtual void serialize_(serializer_t& ar) = 0;

    virtual void deserialize_(deserializer_t& ar) = 0;
};

template<>
struct ElementEvaluator<field::Tensor> {
    /// Type of a tensor index
    using index_t = std::vector<size_t>;

    /// Type of the scalars in the tensor
    using scalar_t = double;

    /// Type of an output serializer
    using serializer_t = parallelzone::Serializer;

    /// Type of an input deserializer
    using deserializer_t = parallelzone::Deserializer;

    void operator()(index_t outer, index_t inner) const;

    void serialize(serializer_t& ar);

    void deserialize(deserializer_t& ar);

protected:
    ElementEvaluator() = default;
    // ElementEvaluator(const ElementEvaluator&) = delete;
    // ElementEvaluator(ElementEvaluator&&) = delete;
    // ElementEvaluator& operator=(const ElementEvaluator&) = delete;
    // ElementEvaluator& operator=(ElementEvaluator&&) = delete;

    virtual void operator_(index_t outer, index_t inner) const = 0;

    virtual void serialize_(serializer_t& ar) = 0;

    virtual void deserialize_(deserializer_t& ar) = 0;
};

//------------------------------------------------------------------------------
//                         Inline Implementations
//------------------------------------------------------------------------------

/// Scalar versions

inline void ElementEvaluator<field::Scalar>::operator()(index_t idx) const {
    operator_(idx);
}

inline void ElementEvaluator<field::Scalar>::serialize(serializer_t& ar) {
    serialize_(ar);
}

inline void ElementEvaluator<field::Scalar>::deserialize(deserializer_t& ar) {
    deserialize_(ar);
}

/// Tensor versions

inline void ElementEvaluator<field::Tensor>::operator()(index_t outer,
                                                        index_t inner) const {
    operator_(outer, inner);
}

inline void ElementEvaluator<field::Tensor>::serialize(serializer_t& ar) {
    serialize_(ar);
}

inline void ElementEvaluator<field::Tensor>::deserialize(deserializer_t& ar) {
    deserialize_(ar);
}

using ScalarElementEvaluator = ElementEvaluator<field::Scalar>;
using TensorElementEvaluator = ElementEvaluator<field::Tensor>;

} // namespace tensorwrapper::tensor::data_evaluator