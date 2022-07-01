#pragma once
#include "ta_headers.hpp"
#include "tensorwrapper/sparse_map/sparse_map.hpp"

namespace tensorwrapper::ta_helpers {

/** @brief Creates a new tensor that from selected tiles of another tensor.
 *
 * @tparam T The type of the tile elements in @p full_matrix.
 *
 * @param[in] full_matrix The tensor from which the submatrix originates.
 * @param[in] mask A mask whose values are used to determine which tiles are
 *                 copied.
 * @return A new tensor that is a submatrix of the original, potentially with
 *         some tiles set to zero based on the masking.
 */
template<typename T>
inline TA::DistArray<TA::Tensor<T>, TA::SparsePolicy> submatrix(
  const TA::DistArray<TA::Tensor<T>, TA::SparsePolicy>& full_matrix,
  const TA::Tensor<float>& mask) {
    // Some typedefs
    using size        = std::size_t;
    using block_index = std::initializer_list<size>;
    using bounds_type = std::vector<size>;
    using bool_vec    = std::vector<bool>;

    // Get dimensions and tile extents
    const auto& dim0 = full_matrix.trange().dim(0);
    const auto& dim1 = full_matrix.trange().dim(1);
    const auto n0    = dim0.tile_extent();
    const auto n1    = dim1.tile_extent();

    // Find dimension tiles to include
    bool_vec cdim0_has(n0, false);
    bool_vec cdim1_has(n1, false);
    for(size i = 0; i < n0; ++i) {
        for(size j = 0; j < n1; ++j) {
            if(mask(i, j) != 0) {
                if(!cdim0_has[i]) cdim0_has[i] = true;
                if(!cdim1_has[j]) cdim1_has[j] = true;
            }
        }
    }

    // Make bounds for compressed dimension 0
    bounds_type cdim0_bounds = {0};
    for(size i = 0; i < n0; ++i) {
        if(cdim0_has[i]) {
            auto [one, two] = dim0.tile(i);
            auto length     = two - one;
            cdim0_bounds.push_back(cdim0_bounds.back() + length);
        }
    }
    auto cdim0 = TA::TiledRange1(cdim0_bounds.begin(), cdim0_bounds.end());

    // Make bounds for compressed dimension 1
    bounds_type cdim1_bounds = {0};
    for(size j = 0; j < n1; ++j) {
        if(cdim1_has[j]) {
            auto [one, two] = dim1.tile(j);
            auto length     = two - one;
            cdim1_bounds.push_back(cdim1_bounds.back() + length);
        }
    }
    auto cdim1 = TA::TiledRange1(cdim1_bounds.begin(), cdim1_bounds.end());

    // Make compressed TiledRange
    TA::TiledRange compressed_trange{cdim0, cdim1};

    // Make and fill compressed submatrix
    TA::DistArray<TA::Tensor<T>, TA::SparsePolicy> submatrix{
      full_matrix.world(), compressed_trange};
    size a = 0, b = 0; // Track submatrix blocks
    for(size i = 0; i < n0; ++i) {
        if(!cdim0_has[i]) continue;
        for(size j = 0; j < n1; ++j) {
            if(!cdim1_has[j]) continue;

            // Block IDs
            block_index cmat_lo{a, b};
            block_index cmat_hi{a + 1, b + 1};
            block_index fmat_lo{i, j};
            block_index fmat_hi{i + 1, j + 1};

            // Fill in blocks
            if(full_matrix.is_zero(fmat_lo) || (mask(i, j) == 0)) {
                submatrix.set(cmat_lo, 0);
            } else {
                submatrix("i, j").block(cmat_lo, cmat_hi) =
                  full_matrix("i, j").block(fmat_lo, fmat_hi);
            }

            // Increment submatrix block
            b++;
            if(b >= cdim1.tile_extent()) {
                a++;
                b = 0;
            }
        }
    }
    submatrix.truncate();

    return submatrix;
}

/** @brief Expands the non-zero tiles of a tensor into a new tensor with a
 *         different TiledRange.
 *
 * @tparam T The type of the tile elements in @p full_matrix.
 *
 * @param[in] submatrix The tensor that is expanded.
 * @param[in] full_trange The TiledRange of the new tensor
 * @param[in] mask A mask whose values are used to determine where the
 *                 submatrix tiles are placed in the new tensor.
 * @return A new tensor that is a with the given TiledRange and whose tiles
 *         are either zero or copied from the submatrix.
 */
template<typename T>
inline TA::DistArray<TA::Tensor<T>, TA::SparsePolicy> expand_submatrix(
  const TA::DistArray<TA::Tensor<T>, TA::SparsePolicy>& submatrix,
  const TA::TiledRange& full_trange, const TA::Tensor<float>& mask) {
    // Some typedefs
    using size        = std::size_t;
    using block_index = std::initializer_list<size>;

    // Get dimensions and tile extents
    const auto& dim0  = full_trange.dim(0);
    const auto& dim1  = full_trange.dim(1);
    const auto n0     = dim0.tile_extent();
    const auto n1     = dim1.tile_extent();
    const auto& cdim0 = submatrix.trange().dim(0);
    const auto& cdim1 = submatrix.trange().dim(1);

    // Make and fill full sized matrix
    TA::DistArray<TA::Tensor<T>, TA::SparsePolicy> full_matrix{
      submatrix.world(), full_trange};

    size a = 0, b = 0; // Track submatrix blocks
    for(size i = 0; i < n0; ++i) {
        for(size j = 0; j < n1; ++j) {
            if(mask(i, j) != 0) {
                // Block IDs
                block_index cmat_lo{a, b};
                block_index cmat_hi{a + 1, b + 1};
                block_index fmat_lo{i, j};
                block_index fmat_hi{i + 1, j + 1};

                // Fill in blocks
                if(submatrix.is_zero(cmat_lo)) {
                    full_matrix.set(fmat_lo, 0);
                } else {
                    full_matrix("i, j").block(fmat_lo, fmat_hi) =
                      submatrix("i, j").block(cmat_lo, cmat_hi);
                }

                // Increment submatrix block
                b++;
                if(b >= cdim1.tile_extent()) {
                    a++;
                    b = 0;
                }
            } else {
                // Set everything else to zero
                block_index fmat_lo{i, j};
                full_matrix.set(fmat_lo, 0);
            }
        }
    }

    // Apply sparse shape
    TA::SparseShape<float> sparse_shape(mask, full_trange);
    full_matrix("i, j") = full_matrix("i, j").set_shape(sparse_shape);
    full_matrix.truncate();

    return full_matrix;
}

} // namespace tensorwrapper::ta_helpers
