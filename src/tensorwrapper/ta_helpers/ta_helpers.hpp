#pragma once
#include "get_block_idx.hpp"
#include "reducer.hpp"
#include "ta_hashers.hpp"
#include "ta_headers.hpp"

namespace tensorwrapper::ta_helpers {

//------------------------------------------------------------------------------
// Tensor Creation
//------------------------------------------------------------------------------

/** @brief Creates a new tensor by applying a function elementwise to an
 *         existing tensor.
 *
 *  This function is a convenience function for creating a new tensor whose
 *  initial elements can be determined solely based on the element values of
 *  another tensor.
 *
 * @tparam TileType The type of the tiles in @p input. Expected to satisfy TA's
 *                  concept of "Tile".
 * @tparam PolicyType The type of @p input's policy. Expected to be either
 *                    DensePolicy or SparsePolicy.
 * @tparam Op should be a type commensurate with that of a function having
 *            signature `element_type(element_type)` where `element_type` is the
 *            type of the elements in each tile.
 *
 * @param[in] input The tensor supplying the input values @p op will be called
 *                  with.
 * @param[in] op The function to apply to each element of @p input to generate
 *               the resulting tensor.
 * @return A new tensor having the same shape and distribution as @p input. The
 *         @f$i@f$-th element of the new tensor is the result of `op(input[i])`,
 *         where `input[i]` is the @f$i@f$-th element of @p input.
 */
template<typename TileType, typename PolicyType, typename Op>
auto apply_elementwise(const TA::DistArray<TileType, PolicyType>& input,
                       Op&& op) {
    auto m = [op{std::forward<Op>(op)}](TileType& result_tile,
                                        const TileType& input_tile) {
        result_tile = input_tile.unary(op);
    };

    TA::DistArray<TileType, PolicyType> rv(input, std::move(m));
    input.world().gop.fence();
    return rv;
}

/** @brief Modifies an existing tensor by applying a function elementwise to its
 *         values.
 *
 *  This function is a convenience function for modifying an existing tensor
 *  by an elementwise in-place operation.
 *
 * @tparam TileType The type of the tiles in @p input. Expected to satisfy TA's
 *                  concept of "Tile".
 * @tparam PolicyType The type of @p input's policy. Expected to be either
 *                    DensePolicy or SparsePolicy.
 * @tparam Op should be a type commensurate with that of a function having
 *            signature `void(element_type&)` where `element_type` is the
 *            type of the elements in each tile.
 *
 * @param[in] input The tensor supplying the input values to be changed in-place
 *                  by @p op.
 * @param[in] op The function to apply to each element of @p input to modify
 *               the values in-place.
 */
template<typename TileType, typename PolicyType, typename Op>
void apply_elementwise_inplace(TA::DistArray<TileType, PolicyType>& input,
                               Op&& op) {
    auto m = [op{std::forward<Op>(op)}](TileType& tile) {
        tile.inplace_unary(op);
    };

    foreach_inplace(input, std::move(m));
}

/** @brief Returns the diagonal elements of a square matrix.
 *
 *  The function requires the matrix to have the same TiledRange1
 *  in both its dimensions.
 *
 * @tparam TensorType the type of the input and returned tensor.
 * @param t The tensor to extract the diagonal elements from.
 * @return A new tensor which contains the diagonal elements.
 */
template<typename TensorType>
auto grab_diagonal(TensorType&& t) {
    auto trange1 = t.trange().dim(0);
    if(trange1 != t.trange().dim(1))
        throw std::runtime_error("Expected a square tiling");
    TA::TiledRange trange{{trange1}};
    using tensor_type = std::decay_t<TensorType>;
    using tile_type   = typename tensor_type::value_type;
    t.world().gop.fence();
    auto l = [=](tile_type& tile, const TA::Range& r) {
        auto idx    = get_block_idx(trange, r);
        auto t_tile = t.find({idx[0], idx[0]}).get();
        tile        = tile_type(r);
        for(auto i : r) {
            std::vector ii{i[0], i[0]};
            tile[i] = t_tile[ii];
        }
        return tile.norm();
    };
    auto rv = TA::make_array<tensor_type>(t.world(), trange, l);
    t.world().gop.fence();
    return rv;
}

/** @brief Returns the elements of a std::vector as a 1D TA DistArray.
 *
 *  The function requires the matrix to have the same TiledRange1
 *  in both its dimensions.
 *
 * @tparam T the scalar type of the input vector and returned tensor.
 * @param vec The std::vector to extract the tensor elements from.
 * @param trange The TiledRange of the returned tensor.
 * @param world The TA::World to use in the returned tensor.
 * @return A new 1D tensor which contains the elements of the vector.
 */
template<typename T>
auto array_from_vec(const std::vector<T>& vec, const TA::TiledRange1& trange,
                    TA::World& world) {
    using tensor_type = TA::DistArray<TA::Tensor<T>, TA::SparsePolicy>;
    using tile_type   = typename tensor_type::value_type;

    tensor_type rv(world, TA::TiledRange{trange});

    for(auto itr = rv.begin(); itr != rv.end(); ++itr) {
        auto range = rv.trange().make_tile_range(itr.index());
        tile_type tile(range);
        for(auto idx : range) { tile(idx) = vec.at(idx[0]); }
        *itr = tile;
    }
    return rv;
}

//------------------------------------------------------------------------------
// Element/Tile Retrieval
//------------------------------------------------------------------------------

/** @brief Retrieves the tile of a tensor containing a particular index.
 *
 *  One often needs a particular element of a tensor. This function wraps the
 *  process of retrieving the tile of the tensor which contains that element.
 *
 *  @note If you know the index of the tile you want is `i` just use
 *        `t.find(i)`.
 *
 *  @note This function returns a future to the tile. To get the actual tile
 *        call the `get()` member of the future.
 *
 *  @tparam IndexType Type of the element's index. Assumed to be a random access
 *                    container whose elements are of integral type.
 *  @tparam TensorType Type of the tensor we are accessing. Assumed to be some
 *                     possibly cv-qualified variant of TiledArray::DistArray.
 *  @param[in] elem_idx The index of the element we want. If @p t is a rank
 *                      @f$r@f$ tensor, @p elem_idx should contain (at least)
 *                      @f$r@f$ elements. Indices are interpreted as being
 *                      global element indices.
 *  @param[in] t The tensor we want element @p elem_idx of.
 *
 *  @return A future to the tile which contains @p elem_idx.
 */
template<typename IndexType, typename TensorType>
auto get_tile(IndexType&& elem_idx, TensorType&& t) {
    auto&& trange = t.trange();
    const auto tile_idx =
      trange.element_to_tile(std::forward<IndexType>(elem_idx));
    return t.find(tile_idx);
}

//------------------------------------------------------------------------------
// Reductions
//------------------------------------------------------------------------------

/** @brief Wraps the process of elementwise reducing two tensors.
 *
 *  Every elementwise reduction of two-tensors can be viewed as a generalization
 *  of the inner product (in practice the actual reduction need not actually
 *  satisfy the mathematical concept of an inner product). That is to say there
 *  are two operations: the "add" operation and the "times" operation. Given two
 *  @f$N@f$ element tensors, @p lhs and @p rhs, such that `lhs[i]` and `rhs[i]`
 *  respectively are the @f$i@f$-th elments of @p lhs and @p rhs, then:
 *
 *  - the "times" operation is used to combine @p lhs and @p rhs into a single
 *    @f$N@f$-element tensor, @f$R@f$, whose @f$i@f$-th value is given by
 *    `R[i] = times(lhs[i], rhs[i]);`, and
 *  - the "add" operation is then used to combine the elements of @f$R@f$ into
 *    a single value.
 *
 *  @todo Does this only work with Tensors? Can it be made to work with futures?
 *
 *  @tparam TileType
 *  @tparam PolicyType
 *  @tparam AddOp Should be a type for an object which can be called as if it
 *                were a function with signature `T(T, T)`, where `T` is the
 *                result of calling @p times_op.
 *  @tparam TimesOp Should be a type of an object which can be called as if it
 *                  were a function with signature
 *                  `T(element_type, element_type)`, where `T` is the input and
 *                  result types of @p AddOp
 * @tparam ResultType Will be deduced to be the result of @p AddOp
 *
 * @param[in] lhs The tensor to go into the left side of @p times_op
 * @param[in] rhs The tensor to go into the right side of @p times_op
 * @param[in] add_op The operation to use as the "add" op
 * @param[in] times_op The operation to use as the "times" op
 * @param[in] init The seed value of the "addition"
 * @return The result of reducing @p lhs and @p rhs using @p add_op and
 *         @p times_op.
 */
template<typename TileType, typename PolicyType, typename AddOp,
         typename TimesOp, typename ResultType>
auto reduce_elementwise(const TA::DistArray<TileType, PolicyType>& lhs,
                        const TA::DistArray<TileType, PolicyType>& rhs,
                        AddOp&& add_op, TimesOp&& times_op, ResultType&& init,
                        std::size_t inner_rank = 0) {
    using tensor_type = TA::DistArray<TileType, PolicyType>;
    using add_type    = std::decay_t<AddOp>;
    using times_type  = std::decay_t<TimesOp>;

    detail_::Reducer<tensor_type, add_type, times_type> r(
      std::forward<AddOp>(add_op), std::forward<TimesOp>(times_op), init);

    const auto idx =
      TA::detail::dummy_annotation(lhs.range().rank(), inner_rank);

    return lhs(idx).reduce(rhs(idx), std::move(r));
}

template<typename TileType, typename PolicyType, typename AddOp,
         typename TimesOp, typename ResultType>
auto reduce_tot_elementwise(const TA::DistArray<TileType, PolicyType>& lhs,
                            const TA::DistArray<TileType, PolicyType>& rhs,
                            AddOp&& add_op, TimesOp&& times_op,
                            ResultType&& init) {
    using inner_type  = typename TileType::value_type;
    using scalar_type = typename TileType::scalar_type;
    auto inner_times  = [times_op{std::forward<TimesOp>(times_op)}](
                         ResultType& result, const scalar_type& first,
                         const scalar_type& second) {
        result = times_op(first, second);
    };
    auto outer_times =
      [=, inner_times{std::move(inner_times)}](inner_type lhs, inner_type rhs) {
          return lhs.reduce(rhs, inner_times, add_op, init);
      };
    const auto& tile0 = lhs.begin()->get();
    auto ir           = tile0[0].range().rank();
    return reduce_elementwise(lhs, rhs, std::forward<AddOp>(add_op),
                              outer_times, init, ir)
      .get();
}

//------------------------------------------------------------------------------
// Comparisons
//------------------------------------------------------------------------------

/** @brief Determines if corresponding elements of two tensors are "close"
 *
 *  This function will compare two tensors, @f$A@f$ and @f$B@f$, elementwise and
 *  determine if all their values are close to one another. "Close" is defined
 *  by two parameters. The absolute tolerance, @p atol, defines the effective
 *  zero for all comparisons. If for a given index @f$I@f$,
 *  @f$\mid A[I]-B[I]\mid@f$ two differs by less than @p atol, the
 *  difference is considered indistinguishable from zero and the elements the
 *  same. Particularly for very large elements achieving a difference on the
 *  order of @p atol is not always reasonable; in these cases, one cares more
 *  about the percent error. In this case
 *  @f$\frac{\mid A[I] - B[I]\mid}{\mid B[I]\mid}@f$ must
 *  be less than a relative tolerance, @p rtol.
 *
 *  This function combines these criteria and returns true if @f$A@f$ and
 *  @f$B@f$ satisfy:
 *
 *  @f[
 *     \mid A - B\mid \le atol + rtol\mid B\mid
 *  @f]
 *
 *  Note that this is **NOT** symmetric in @f$A@f$ and @f$B@f$, but is
 *  commensurate with @f$B@f$ being the reference (thereby dictating what is an
 *  acceptable relative tolerance).
 *
 *  @tparam T The type of the first tensor. Assumed to be (possibly
 *            cv-qualified) TiledArray::DistArray
 *  @tparam U The type of the second tensor. Assumed to be the same (up to
 *            cv-qualifiers and value semantics as @p T).
 *  @param[in] actual The tensor you computed and are comparing against a
 *                    reference value.
 *  @param[in] ref The tensor which @p actual is being compared to. Should
 *                 be "the correct value".
 *  @param[in] abs_comp a bool which allows the comparison to run over the
 *                      absolute values of the input tensors.
 *  @param[in] rtol The maximum percent error (as a decimal) allowed for any
 *                  particular value. Assumed to be a positive decimal. Defaults
 *                  to 1.0E-5, *i.e.*, 0.0001%.
 *  @param[in] atol The effective value of zero for comparisons. Assumed to be a
 *                  positive decimal less than 1.0. Defaults to 1.0E-8.
 *  @return True if @p actual is "close" to @p ref and false otherwise.
 */
template<typename T, typename U,
         typename V = typename std::decay_t<T>::scalar_type>
bool allclose(T&& actual, U&& ref, const bool abs_comp = false,
              V&& rtol = 1.0E-5, V&& atol = 1.0E-8,
              std::size_t inner_rank = 0) {
    using tensor_type = std::decay_t<T>;
    using tile_type   = typename tensor_type::value_type;
    using scalar_type = std::decay_t<V>;
    static_assert(std::is_same_v<tensor_type, std::decay_t<U>>,
                  "different tensor types is currently unsupported");

    constexpr bool is_tot = TA::detail::is_tensor_of_tensor_v<tile_type>;

    // Get a dummy string label (something like "i0, i1, i2, ...")
    auto idx = TA::detail::dummy_annotation(actual.range().rank(), inner_rank);

    // Compute A - B, call result AmB
    tensor_type AmB;
    if(abs_comp) {
        // TODO: There probably is some way to do this without forming the
        // intermediate tensors
        auto abs_lambda = [](const scalar_type& val) { return std::fabs(val); };
        auto abs_actual = apply_elementwise(actual, abs_lambda);
        auto abs_ref    = apply_elementwise(ref, abs_lambda);
        AmB(idx)        = abs_actual(idx) - abs_ref(idx);
    } else {
        AmB(idx) = actual(idx) - ref(idx);
    }

    auto times_op = [=](scalar_type lhs, scalar_type rhs) {
        return std::fabs(lhs) <= atol + rtol * std::fabs(rhs);
    };

    std::logical_and<bool> add_op;

    if constexpr(!is_tot) {
        return reduce_elementwise(AmB, ref, add_op, times_op, true).get();
    } else {
        using inner_type = typename tile_type::value_type;
        auto inner_times = [=](bool& result, const scalar_type& first,
                               const scalar_type& second) {
            result = times_op(first, second);
        };
        auto outer_times = [=](inner_type lhs, inner_type rhs) {
            return lhs.reduce(rhs, inner_times, add_op, true);
        };
        return reduce_elementwise(AmB, ref, add_op, outer_times, true,
                                  inner_rank)
          .get();
    }
}

/// Reorders the arguments to be more convenient for a ToT
template<typename T, typename U,
         typename V = typename std::decay_t<T>::scalar_type>
bool allclose_tot(T&& actual, U&& ref, std::size_t inner_rank = 0,
                  const bool abs_comp = false, V&& rtol = 1.0E-5,
                  V&& atol = 1.0E-8) {
    return allclose(std::forward<T>(actual), std::forward<U>(ref), abs_comp,
                    rtol, atol, inner_rank);
}

//------------------------------------------------------------------------------
// TiledRange1 Creation
//------------------------------------------------------------------------------

/** @brief Creates a new TiledRange1.
 *
 *  This function is a convenience function for creating a new 1d tiled range.
 *
 * @param[in] length The extent of the range.
 * @param[in] tilesize The max number of elements in a tile.
 * @param[in] init_offset The value of the first element of the range.
 * @return A new TiledRange1 spanning between @p init_offset and @p length,
 *         where all but the last tile will have a size of @p tilesize.
 */
inline TA::TiledRange1 make_1D_trange(std::size_t length, std::size_t tilesize,
                                      std::size_t init_offset = 0) {
    std::vector<std::size_t> bounds;
    for(std::size_t i = init_offset; i < length; i += tilesize)
        bounds.push_back(i);
    bounds.push_back(length);
    return TA::TiledRange1{bounds.begin(), bounds.end()};
}

} // namespace tensorwrapper::ta_helpers

namespace TiledArray {

/** @brief Enables comparison between TA DistArray objects
 *
 * Free function to enable comparison between TA DistArray objects.
 *
 * @tparam TensorType Type of tensor (TA::DistArray) for @p A.
 * @tparam PolicyType Type of policy for @p A. Either DensePolicy or
 * SparsePolicy.
 * @tparam TensorType Type of tensor (TA::DistArray) for @p B.
 * @tparam PolicyType Type of policy for @p B. Either DensePolicy or
 * SparsePolicy.
 * @param[in] lhs DistArray object
 * @param[in] rhs DistArray object
 */
template<typename LHSTileType, typename LHSPolicyType, typename RHSTileType,
         typename RHSPolicyType>
bool operator==(const TA::DistArray<LHSTileType, LHSPolicyType>& lhs,
                const TA::DistArray<RHSTileType, RHSPolicyType>& rhs) {
    if constexpr(!std::is_same_v<decltype(lhs), decltype(rhs)>)
        return false;
    else {
        if(lhs.is_initialized() != rhs.is_initialized()) return false;
        if(!lhs.is_initialized()) return true;
        if(lhs.trange() != rhs.trange()) return false;

        using scalar_type = typename LHSTileType::scalar_type;
        using namespace tensorwrapper::ta_helpers;
        std::logical_and<bool> add_op;
        std::equal_to<scalar_type> times_op;
        if constexpr(!TA::detail::is_tensor_of_tensor_v<LHSTileType>) {
            return reduce_elementwise(lhs, rhs, add_op, times_op, true);
        } else {
            return reduce_tot_elementwise(lhs, rhs, add_op, times_op, true);
        }
    }
}

template<typename LHSTileType, typename LHSPolicyType, typename RHSTileType,
         typename RHSPolicyType>
bool operator!=(const TA::DistArray<LHSTileType, LHSPolicyType>& lhs,
                const TA::DistArray<RHSTileType, RHSPolicyType>& rhs) {
    return !(lhs == rhs);
}

} // namespace TiledArray
