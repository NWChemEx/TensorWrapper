#include "../../helpers.hpp"
#include "../../inputs.hpp"

using namespace tensorwrapper;

/* There's a plethora of possible combinations. We can't test them all */

TEST_CASE("TensorInput") {
    parallelzone::runtime::RuntimeView rv;

    auto defaulted   = testing::default_input();
    auto scalar      = testing::smooth_scalar();
    auto symm_matrix = testing::smooth_symmetric_matrix();

    // Until we have more features we test the remainder of the input class with
    // the following "stub" objects
    shape::Smooth shape{3, 3};
    symmetry::Group g{symmetry::Permutation{0, 1}};
    sparsity::Pattern sparsity;
    layout::Logical logical(shape, g, sparsity);
    layout::Physical physical(shape, g, sparsity);

    SECTION("Ctors") {
        SECTION("Default") {
            REQUIRE(defaulted.m_pshape == nullptr);
            REQUIRE(defaulted.m_psymmetry == nullptr);
            REQUIRE(defaulted.m_psparsity == nullptr);
            REQUIRE(defaulted.m_plogical == nullptr);
            REQUIRE(defaulted.m_pphysical == nullptr);
            REQUIRE(defaulted.m_palloc == nullptr);
            REQUIRE(defaulted.m_pbuffer == nullptr);
            REQUIRE(defaulted.m_rv == rv);
        }

        SECTION("Shape (by value)") {
            REQUIRE(scalar.m_pshape->are_equal(shape::Smooth{}));
            REQUIRE(scalar.m_psymmetry == nullptr);
            REQUIRE(scalar.m_psparsity == nullptr);
            REQUIRE(scalar.m_plogical == nullptr);
            REQUIRE(scalar.m_pphysical == nullptr);
            REQUIRE(scalar.m_palloc == nullptr);
            REQUIRE(scalar.m_pbuffer == nullptr);
            REQUIRE(scalar.m_rv == rv);
        }

        SECTION("Shape (by pointer)") {
            shape::Smooth scalar_shape{};
            auto pscalar_shape        = scalar_shape.clone();
            auto scalar_shape_address = pscalar_shape.get();
            detail_::TensorInput i(std::move(pscalar_shape));
            REQUIRE(scalar.m_pshape->are_equal(*i.m_pshape));
            REQUIRE(i.m_pshape.get() == scalar_shape_address);
            REQUIRE(i.m_psymmetry == nullptr);
            REQUIRE(i.m_psparsity == nullptr);
            REQUIRE(i.m_plogical == nullptr);
            REQUIRE(i.m_pphysical == nullptr);
            REQUIRE(i.m_palloc == nullptr);
            REQUIRE(i.m_pbuffer == nullptr);
            REQUIRE(i.m_rv == rv);
        }

        SECTION("Group (by value)") {
            REQUIRE(symm_matrix.m_pshape->are_equal(shape));
            REQUIRE(*symm_matrix.m_psymmetry == g);
            REQUIRE(symm_matrix.m_psparsity == nullptr);
            REQUIRE(symm_matrix.m_plogical == nullptr);
            REQUIRE(symm_matrix.m_pphysical == nullptr);
            REQUIRE(symm_matrix.m_palloc == nullptr);
            REQUIRE(symm_matrix.m_pbuffer == nullptr);
            REQUIRE(symm_matrix.m_rv == rv);
        }

        SECTION("Group (by pointer)") {
            auto pg         = std::make_unique<symmetry::Group>(g);
            auto pg_address = pg.get();
            detail_::TensorInput i(std::move(pg), shape);
            REQUIRE(i.m_pshape->are_equal(shape));
            REQUIRE(*i.m_psymmetry == g);
            REQUIRE(i.m_psymmetry.get() == pg_address);
            REQUIRE(i.m_psparsity == nullptr);
            REQUIRE(i.m_plogical == nullptr);
            REQUIRE(i.m_pphysical == nullptr);
            REQUIRE(i.m_palloc == nullptr);
            REQUIRE(i.m_pbuffer == nullptr);
            REQUIRE(i.m_rv == rv);
        }

        SECTION("Sparsity (by value)") {
            detail_::TensorInput i(g, shape, sparsity);
            REQUIRE(i.m_pshape->are_equal(shape));
            REQUIRE(*i.m_psymmetry == g);
            REQUIRE(*i.m_psparsity == sparsity);
            REQUIRE(i.m_plogical == nullptr);
            REQUIRE(i.m_pphysical == nullptr);
            REQUIRE(i.m_palloc == nullptr);
            REQUIRE(i.m_pbuffer == nullptr);
            REQUIRE(i.m_rv == rv);
        }

        SECTION("Sparsity (by pointer)") {
            auto psparsity = std::make_unique<sparsity::Pattern>(sparsity);
            auto psparsity_address = psparsity.get();
            detail_::TensorInput i(g, shape, std::move(psparsity));
            REQUIRE(i.m_pshape->are_equal(shape));
            REQUIRE(*i.m_psymmetry == g);
            REQUIRE(*i.m_psparsity == sparsity);
            REQUIRE(i.m_psparsity.get() == psparsity_address);
            REQUIRE(i.m_plogical == nullptr);
            REQUIRE(i.m_pphysical == nullptr);
            REQUIRE(i.m_palloc == nullptr);
            REQUIRE(i.m_pbuffer == nullptr);
            REQUIRE(i.m_rv == rv);
        }

        SECTION("Logical (by value)") {
            detail_::TensorInput i(logical);
            REQUIRE(i.m_pshape == nullptr);
            REQUIRE(i.m_psymmetry == nullptr);
            REQUIRE(i.m_psparsity == nullptr);
            REQUIRE(i.m_plogical->are_equal(logical));
            REQUIRE(i.m_pphysical == nullptr);
            REQUIRE(i.m_palloc == nullptr);
            REQUIRE(i.m_pbuffer == nullptr);
            REQUIRE(i.m_rv == rv);
        }

        SECTION("Logical (by pointer)") {
            auto plogical         = std::make_unique<layout::Logical>(logical);
            auto plogical_address = plogical.get();
            detail_::TensorInput i(std::move(plogical));
            REQUIRE(i.m_pshape == nullptr);
            REQUIRE(i.m_psymmetry == nullptr);
            REQUIRE(i.m_psparsity == nullptr);
            REQUIRE(i.m_plogical->are_equal(logical));
            REQUIRE(i.m_plogical.get() == plogical_address);
            REQUIRE(i.m_pphysical == nullptr);
            REQUIRE(i.m_palloc == nullptr);
            REQUIRE(i.m_pbuffer == nullptr);
            REQUIRE(i.m_rv == rv);
        }

        SECTION("Physical (by value)") {
            detail_::TensorInput i(physical, logical);
            REQUIRE(i.m_pshape == nullptr);
            REQUIRE(i.m_psymmetry == nullptr);
            REQUIRE(i.m_psparsity == nullptr);
            REQUIRE(i.m_plogical->are_equal(logical));
            REQUIRE(i.m_pphysical->are_equal(physical));
            REQUIRE(i.m_palloc == nullptr);
            REQUIRE(i.m_pbuffer == nullptr);
            REQUIRE(i.m_rv == rv);
        }

        SECTION("Physical (by pointer)") {
            auto pphysical = std::make_unique<layout::Physical>(physical);
            auto pphysical_address = pphysical.get();
            detail_::TensorInput i(std::move(pphysical), logical);
            REQUIRE(i.m_pshape == nullptr);
            REQUIRE(i.m_psymmetry == nullptr);
            REQUIRE(i.m_psparsity == nullptr);
            REQUIRE(i.m_plogical->are_equal(logical));
            REQUIRE(i.m_pphysical->are_equal(physical));
            REQUIRE(i.m_pphysical.get() == pphysical_address);
            REQUIRE(i.m_palloc == nullptr);
            REQUIRE(i.m_pbuffer == nullptr);
            REQUIRE(i.m_rv == rv);
        }

        SECTION("Allocator (by value)") {}

        SECTION("Allocator (by pointer)") {}

        SECTION("Buffer (by value)") {}

        SECTION("Buffer (by pointer)") {}
    }

    SECTION("has_shape") {
        REQUIRE_FALSE(defaulted.has_shape());
        REQUIRE(scalar.has_shape());
    }

    SECTION("has_symmetry") {
        REQUIRE_FALSE(defaulted.has_symmetry());
        REQUIRE(symm_matrix.has_symmetry());
    }

    SECTION("has_sparsity") { REQUIRE_FALSE(defaulted.has_sparsity()); }

    SECTION("has_logical_layout") {
        REQUIRE_FALSE(defaulted.has_logical_layout());
    }

    SECTION("has_physical_layout") {
        REQUIRE_FALSE(defaulted.has_physical_layout());
    }

    SECTION("has_allocator") { REQUIRE_FALSE(defaulted.has_allocator()); }

    SECTION("has_buffer") { REQUIRE_FALSE(defaulted.has_buffer()); }
}
