#include "../ta_helpers/ta_helpers.hpp"
#include "buffer/detail_/ta_buffer_pimpl.hpp"
#include "conversion/conversion.hpp"
#include "detail_/ta_to_tw.hpp"
#include <TiledArray/conversions/retile.h>
#include <tensorwrapper/tensor/creation.hpp>

namespace tensorwrapper::tensor {

ScalarTensorWrapper concatenate(const ScalarTensorWrapper& lhs,
                                const ScalarTensorWrapper& rhs,
                                std::size_t dim) {
    to_ta_distarrayd_t converter;
    const auto& C_lhs = converter.convert(lhs.buffer());
    const auto& C_rhs = converter.convert(rhs.buffer());
    const auto rank   = C_lhs.trange().rank();

    if(rank != C_rhs.trange().rank())
        throw std::runtime_error("Must have the same rank");

    if(dim >= rank) throw std::runtime_error("dim must be in [0, rank)");

    std::vector<int> lhs_lo, lhs_hi, rhs_lo, rhs_hi;
    std::vector<TA::TiledRange1> out_tr1;
    for(std::size_t i = 0; i < rank; ++i) {
        const auto& lhs_tr1 = C_lhs.trange().dim(i);
        const auto& rhs_tr1 = C_rhs.trange().dim(i);

        const auto nlhs = lhs_tr1.tile_extent();
        const auto nrhs = rhs_tr1.tile_extent();
        lhs_lo.push_back(0);
        if(i == dim) {
            lhs_hi.push_back(nlhs);
            rhs_lo.push_back(nlhs);
            rhs_hi.push_back(nlhs + nrhs);
            out_tr1.push_back(TA::concat(lhs_tr1, rhs_tr1));
            continue;
        } else if(lhs_tr1 != rhs_tr1)
            throw std::runtime_error("Must have the same shape");

        lhs_hi.push_back(nlhs);
        rhs_lo.push_back(0);
        rhs_hi.push_back(nlhs);
        out_tr1.push_back(lhs_tr1);
    }

    TA::TiledRange out_tr(out_tr1.begin(), out_tr1.end());
    TA::TSpArrayD C_out(C_lhs.world(), out_tr);
    auto idx                         = lhs.make_annotation();
    C_out(idx).block(lhs_lo, lhs_hi) = C_lhs(idx);
    C_out(idx).block(rhs_lo, rhs_hi) = C_rhs(idx);

    const auto total = out_tr1[dim].extent();
    TA::TiledRange1 new_tr1(0, total);
    out_tr1[dim] = new_tr1;
    TA::TiledRange new_tr(out_tr1.begin(), out_tr1.end());
    C_out = TA::retile(C_out, new_tr);
    return detail_::ta_to_tw(C_out);
}

TensorOfTensorsWrapper concatenate(const TensorOfTensorsWrapper& lhs,
                                   const TensorOfTensorsWrapper& rhs,
                                   std::size_t dim) {
    throw std::runtime_error("NYI");
    // Guts of the ToT matrix concatenate copy/pasted from MP2
    //     const auto& Cocc = occ.C();
    //     const auto& Cpno = pno.C();

    //     using tot_type = std::decay_t<decltype(Cocc)>;
    //     using namespace tensorwrapper::ta_helpers;

    //     if(Cocc.trange() != Cpno.trange()) {
    //         throw std::runtime_error("TiledRanges are assumed the same");
    //     }

    //     auto l = [=](auto& t, const TA::Range& r) {
    //         using tile_type   = std::decay_t<decltype(t)>;
    //         using tensor_type = typename tile_type::value_type;

    //         auto tile_idx        = get_block_idx(Cocc.trange(), r);
    //         auto occ_block       = Cocc.find(tile_idx).get();
    //         auto pno_block       = Cpno.find(tile_idx).get();
    //         auto& occ_tile_range = occ_block.range();

    //         if(occ_tile_range != pno_block.range())
    //             throw std::runtime_error("Huh? Thought we already checked
    //             this...");

    //         t = tile_type(r);
    //         for(auto& ij : occ_tile_range) {
    //             auto& occ_ij = occ_block(ij);
    //             auto& pno_ij = pno_block(ij);
    //             auto naos    = occ_ij.range().extent(0);
    //             if(naos != pno_ij.range().extent(0))
    //                 throw std::runtime_error("Expected the same number of
    //                 AOs");

    //             auto npno = pno_ij.range().extent(1);
    //             TA::Range new_range(naos, 1 + npno);
    //             tensor_type inner(new_range, 0.0);

    //             using size_type = decltype(naos);
    //             for(size_type mu = 0; mu < naos; ++mu) {
    //                 inner({mu, size_type{0}}) = occ_ij({mu});

    //                 for(decltype(npno) a = 0; a < npno; ++a)
    //                     inner({mu, 1 + a}) = pno_ij({mu, a});
    //             }
    //             t(ij) = inner;
    //         }
    //         return t.norm();
    //     };

    //     auto Corb = TA::make_array<tot_type>(Cocc.world(), Cocc.trange(),
    //     l); return mp2::type::sparse_derived<double>(Corb,
    //     occ.from_space());
}

ScalarTensorWrapper grab_diagonal(const ScalarTensorWrapper& t) {
    to_ta_distarrayd_t converter;
    const auto& t_ta = converter.convert(t.buffer());
    return detail_::ta_to_tw(ta_helpers::grab_diagonal(t_ta));
}

ScalarTensorWrapper diagonal_tensor_wrapper(
  double val, const allocator::Allocator<field::Scalar>& allocator,
  const Shape<field::Scalar>& shape) {
    auto& world = TA::get_default_world();

    std::vector<TA::TiledRange1> dims{};
    for(auto i : shape.extents()) {
        dims.push_back(ta_helpers::make_1D_trange(i, i));
    }
    TA::TiledRange trange(dims);

    auto ta_diag = TA::diagonal_array<TA::TSpArrayD>(world, trange, val);

    return detail_::ta_to_tw(ta_diag); // shape.clone(), allocator.clone());
};

ScalarTensorWrapper diagonal_tensor_wrapper(
  const std::vector<double>& vals,
  const allocator::Allocator<field::Scalar>& allocator,
  const Shape<field::Scalar>& shape) {
    auto& world = TA::get_default_world();

    std::vector<TA::TiledRange1> dims{};
    for(auto i : shape.extents()) {
        dims.push_back(ta_helpers::make_1D_trange(i, i));
    }
    TA::TiledRange trange(dims);

    auto ta_diag = TA::diagonal_array<TA::TSpArrayD>(world, trange,
                                                     vals.begin(), vals.end());

    return detail_::ta_to_tw(ta_diag); //, shape.clone(), allocator.clone());
};

ScalarTensorWrapper stack_tensors(std::vector<ScalarTensorWrapper> tensors) {
    using ta_type   = TA::TSpArrayD;
    using tile_type = typename ta_type::value_type;

    to_ta_distarrayd_t converter;
    auto leading_ta   = converter.convert(tensors[0].buffer());
    auto slice_trange = leading_ta.trange();
    auto& world       = leading_ta.world();

    // Prepend the new dimension to the existing ones
    auto new_dim = ta_helpers::make_1D_trange(tensors.size(), 1);
    std::vector<TA::TiledRange1> intermediate_dims{new_dim};
    for(auto i = 0; i < slice_trange.rank(); ++i) {
        intermediate_dims.push_back(slice_trange.dim(i));
    }
    TA::TiledRange intermediate_trange(intermediate_dims);

    // Build up stacked tensor
    ta_type new_tensor(world, intermediate_trange);
    for(auto dim = 0; dim < tensors.size(); ++dim) {
        auto current_ta     = converter.convert(tensors[dim].buffer());
        auto current_trange = current_ta.trange();

        // Check that the array has the correct layout
        if(current_trange != slice_trange)
            throw std::runtime_error(
              "Stacking tensors must have the same tiled range");

        // Place the tiles from the current array into their correct location
        // inside the new array.
        for(std::size_t i = 0; i < current_ta.size(); ++i) {
            auto i_range = current_trange.make_tile_range(i);
            auto index   = ta_helpers::get_block_idx(current_trange, i_range);

            // Extend tile index into new dimension
            std::vector<long> output_index{static_cast<long>(dim)};
            for(auto i = 0; i < i_range.rank(); ++i) {
                output_index.push_back(index[i]);
            }

            // Get tiledrange of new location
            auto output_range =
              intermediate_trange.make_tile_range(output_index);

            // Make and set new tile
            if(current_ta.is_zero(index)) {
                tile_type rv_tile(output_range, 0.0);
                new_tensor.set(output_index, rv_tile);
            } else {
                tile_type rv_tile(output_range,
                                  current_ta.find(index).get().begin());
                new_tensor.set(output_index, rv_tile);
            }
        }
    }

    /// Retile into default OneBigTile arrangement
    /// Probably shouldn't have to do this.
    /// TODO: Remove at some point?
    auto dim1 = ta_helpers::make_1D_trange(tensors.size(), tensors.size());
    std::vector<TA::TiledRange1> final_dims{dim1};
    for(auto i : slice_trange.elements_range().extent()) {
        final_dims.push_back(ta_helpers::make_1D_trange(i, i));
    }
    TA::TiledRange final_trange(final_dims);
    new_tensor = TA::retile(new_tensor, final_trange);

    return detail_::ta_to_tw(new_tensor);
}

Eigen::MatrixXd tensor_wrapper_to_eigen(const ScalarTensorWrapper& tensor) {
    to_ta_distarrayd_t converter;
    const auto& ta_tensor = converter.convert(tensor.buffer());
    return TA::array_to_eigen(ta_tensor);
};

ScalarTensorWrapper eigen_to_tensor_wrapper(const Eigen::MatrixXd& matrix) {
    auto& world = TA::get_default_world();

    auto cols_tr = ta_helpers::make_1D_trange(matrix.cols(), matrix.cols());
    auto rows_tr = ta_helpers::make_1D_trange(matrix.rows(), matrix.cols());
    TA::TiledRange trange({cols_tr, rows_tr});

    auto tensor = TA::eigen_to_array<TA::TSpArrayD>(world, trange, matrix);

    return detail_::ta_to_tw(tensor);
};

} // namespace tensorwrapper::tensor
