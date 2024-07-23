#include "../../helpers.hpp"
#include "../../inputs.hpp"
#include <tensorwrapper/buffer/eigen.hpp>
#include <tensorwrapper/layout/logical.hpp>
#include <tensorwrapper/shape/smooth.hpp>
#include <tensorwrapper/tensor/detail_/tensor_pimpl.hpp>

using namespace tensorwrapper;
using buffer_type = buffer::Eigen<double, 2>;
using tensor_type = typename buffer_type::tensor_type;

TEST_CASE("TensorPIMPL") {
    auto input = testing::smooth_vector();
    symmetry::Group g;
    sparsity::Pattern sparsity;

    layout::Logical logical_corr(*input.m_pshape, g, sparsity);
    auto pbuffer_corr    = input.m_pbuffer->clone();
    auto plogical        = logical_corr.clone_as<layout::Logical>();
    auto logical_address = plogical.get();

    auto buffer_address = input.m_pbuffer.get();

    detail_::TensorPIMPL value(std::move(plogical), std::move(input.m_pbuffer));

    SECTION("CTor") {
        SECTION("Value") {
            REQUIRE(value.logical_layout().are_equal(logical_corr));
            REQUIRE(&value.logical_layout() == logical_address);

            REQUIRE(value.buffer().are_equal(*pbuffer_corr));
            REQUIRE(&value.buffer() == buffer_address);

            using except_t = std::runtime_error;
            REQUIRE_THROWS_AS(
              detail_::TensorPIMPL(nullptr, std::move(pbuffer_corr)), except_t);

            REQUIRE_THROWS_AS(
              detail_::TensorPIMPL(logical_corr.clone_as<layout::Logical>(),
                                   nullptr),
              except_t);
        }
    }

    SECTION("clone") {
        auto pvalue_copy = value.clone();
        REQUIRE(*pvalue_copy == value);
    }

    SECTION("logical_layout()") {
        REQUIRE(value.logical_layout().are_equal(logical_corr));
    }

    SECTION("logical_layout() const") {
        const auto& const_value = value;
        REQUIRE(const_value.logical_layout().are_equal(logical_corr));
    }

    SECTION("buffer()") { REQUIRE(value.buffer().are_equal(*pbuffer_corr)); }

    SECTION("buffer() const") {
        const auto& const_value = value;
        REQUIRE(const_value.buffer().are_equal(*pbuffer_corr));
    }

    SECTION("operator==") {
        auto plogical2 = logical_corr.clone_as<layout::Logical>();
        auto pbuffer2  = pbuffer_corr->clone();

        SECTION("Same state") {
            detail_::TensorPIMPL same(std::move(plogical2),
                                      std::move(pbuffer2));
            REQUIRE(value == same);
        }

        SECTION("Different logical layout") {
            shape::Smooth scalar{};
            auto pl2 = std::make_unique<layout::Logical>(scalar, g, sparsity);
            detail_::TensorPIMPL diff(std::move(pl2), std::move(pbuffer2));
            REQUIRE_FALSE(value == diff);
        }

        SECTION("Different buffer") {
            auto other_vector = testing::smooth_vector_alt();
            detail_::TensorPIMPL diff(std::move(plogical2),
                                      std::move(other_vector.m_pbuffer));
            buffer_type buffer2;
        }
    }
}
