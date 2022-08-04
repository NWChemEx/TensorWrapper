#pragma once
#include <parallelzone/archive_wrapper.hpp>
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

    /// Type of an output serializer
    using serializer_t = parallelzone::Serializer;

    /// Type of an input deserializer
    using deserializer_t = parallelzone::Deserializer;

    void operator()(index_t lo, index_t up, scalar_t* d) const;

    void serialize(serializer_t& ar);

    void deserialize(deserializer_t& ar);

protected:
    // TileEvaluator() = default;
    // TileEvaluator(const TileEvaluator&) = delete;
    // TileEvaluator(TileEvaluator&&) = delete;
    // TileEvaluator& operator=(const TileEvaluator&) = delete;
    // TileEvaluator& operator=(TileEvaluator&&) = delete;

    virtual void operator_(index_t lo, index_t up, scalar_t* d) const = 0;

    virtual void serialize_(serializer_t& ar) = 0;

    virtual void deserialize_(deserializer_t& ar) = 0;
};

template<>
struct TileEvaluator<field::Tensor> {
    /// Type of a tensor index
    using index_t = std::vector<size_t>;

    /// Type of the scalars in the tensor
    using scalar_t = double;

    /// Type of an output serializer
    using serializer_t = parallelzone::Serializer;

    /// Type of an input deserializer
    using deserializer_t = parallelzone::Deserializer;

    void operator()(index_t outer, index_t lo, index_t up, scalar_t* d) const;

    void serialize(serializer_t& ar);

    void deserialize(deserializer_t& ar);

protected:
    TileEvaluator() = delete;
    // TileEvaluator(const TileEvaluator&) = delete;
    // TileEvaluator(TileEvaluator&&) = delete;
    // TileEvaluator& operator=(const TileEvaluator&) = delete;
    // TileEvaluator& operator=(TileEvaluator&&) = delete;

    virtual void operator_(index_t outer, index_t lo, index_t up,
                           scalar_t* d) const = 0;

    virtual void serialize_(serializer_t& ar) = 0;

    virtual void deserialize_(deserializer_t& ar) = 0;
};

//------------------------------------------------------------------------------
//                         Inline Implementations
//------------------------------------------------------------------------------

/// Scalar versions

inline void TileEvaluator<field::Scalar>::operator()(index_t lo, index_t up,
                                                     scalar_t* d) const {
    operator_(lo, up, d);
}

inline void TileEvaluator<field::Scalar>::serialize(serializer_t& ar) {
    serialize_(ar);
}

inline void TileEvaluator<field::Scalar>::deserialize(deserializer_t& ar) {
    deserialize_(ar);
}

/// Tensor versions

inline void TileEvaluator<field::Tensor>::operator()(index_t outer, index_t lo,
                                                     index_t up,
                                                     scalar_t* d) const {
    operator_(outer, lo, up, d);
}

inline void TileEvaluator<field::Tensor>::serialize(serializer_t& ar) {
    serialize_(ar);
}

inline void TileEvaluator<field::Tensor>::deserialize(deserializer_t& ar) {
    deserialize_(ar);
}

using ScalarTileEvaluator = TileEvaluator<field::Scalar>;
using TensorTileEvaluator = TileEvaluator<field::Tensor>;

} // namespace tensorwrapper::tensor::data_evaluator