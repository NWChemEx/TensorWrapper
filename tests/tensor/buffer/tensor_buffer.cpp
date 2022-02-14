#include "tensorwrapper/tensor/buffer/buffer.hpp"

#include "make_pimpl.hpp"
#include <catch2/catch.hpp>

using namespace tensorwrapper::tensor;

/*
 * Unit testing notes:
 *
 * The Buffer class only operates with a PIMPL in it. We have already unit
 * tested the TABufferPIMPL so we use that one. The point of these unit tests
 * are more to ensure that the PIMPL is hooked in correctly than to exahustively
 * test its functionality.
 */

TEST_CASE("Buffer<Tensor>") {
    using field_type  = field::Tensor;
    using buffer_type = buffer::Buffer<field_type>;
    using pimpl_type  = buffer::detail_::TABufferPIMPL<field_type>;

    buffer_type defaulted;
    auto&& [pvov, pvom, pmov] = testing::make_pimpl<field_type>();
    buffer_type vov(pvov->clone());
    buffer_type vom(pvom->clone());
    buffer_type mov(pmov->clone());

    SECTION("CTors") {
        SECTION("Default") { REQUIRE_FALSE(defaulted.is_initialized()); }
        SECTION("PIMPL") { REQUIRE(vov.is_initialized()); }
        SECTION("Copy") {
            buffer_type v2(vov);
            REQUIRE(v2.is_initialized());
            REQUIRE(v2 == vov);
        }
        SECTION("Move") {
            buffer_type corr(vov);
            buffer_type v2(std::move(vov));
            REQUIRE(v2.is_initialized());
            REQUIRE_FALSE(vov.is_initialized());
            REQUIRE(v2 == corr);
        }
        SECTION("Copy assignment") {
            buffer_type v2;
            auto pv2 = &(v2 = vov);
            REQUIRE(pv2 == &v2);
            REQUIRE(v2.is_initialized());
            REQUIRE(v2 == vov);
        }
        SECTION("Move assignment") {
            buffer_type v2;
            buffer_type corr(vov);
            auto pv2 = &(v2 = std::move(vov));
            REQUIRE(pv2 == &v2);
            REQUIRE(v2.is_initialized());
            REQUIRE_FALSE(vov.is_initialized());
            REQUIRE(v2 == corr);
        }
    }

    SECTION("Scale") {
        buffer_type out;
        auto out_pimpl = std::make_unique<pimpl_type>();
        SECTION("vector-of-vectors") {
            vov.scale("i;j", "i;j", out, 2.0);
            pvov->scale("i;j", "i;j", *out_pimpl, 2.0);
            buffer_type corr(std::move(out_pimpl));
            REQUIRE(out == corr);
        }
        SECTION("vector-of-matrices") {
            vom.scale("i;j,k", "i;j,k", out, 2.0);
            pvom->scale("i;j,k", "i;j,k", *out_pimpl, 2.0);
            buffer_type corr(std::move(out_pimpl));
            REQUIRE(out == corr);
        }
        SECTION("matrix-of-vectors") {
            mov.scale("i,j;k", "i,j;k", out, 2.0);
            pmov->scale("i,j;k", "i,j;k", *out_pimpl, 2.0);
            buffer_type corr(std::move(out_pimpl));
            REQUIRE(out == corr);
        }

        SECTION("throws if not initialized") {
            using error_t = std::runtime_error;
            REQUIRE_THROWS_AS(defaulted.scale("i", "i", out, 2.0), error_t);
        }
    }

    SECTION("add") {
        buffer_type out;
        auto out_pimpl = std::make_unique<pimpl_type>();
        auto rhs_pimpl = std::make_unique<pimpl_type>();

        SECTION("vector-of-vectors") {
            pvov->scale("i;j", "i;j", *rhs_pimpl, 2.0);
            buffer_type rhs(rhs_pimpl->clone());
            vov.add("i;j", "i;j", out, "i;j", rhs);
            pvov->add("i;j", "i;j", *out_pimpl, "i;j", *rhs_pimpl);
            buffer_type corr(std::move(out_pimpl));
            REQUIRE(out == corr);
        }

        SECTION("vector-of-matrices") {
            pvom->scale("i;j,k", "i;j,k", *rhs_pimpl, 2.0);
            buffer_type rhs(rhs_pimpl->clone());
            vom.add("i;j,k", "i;j,k", out, "i;j,k", rhs);
            pvom->add("i;j,k", "i;j,k", *out_pimpl, "i;j,k", *rhs_pimpl);
            buffer_type corr(std::move(out_pimpl));
            REQUIRE(out == corr);
        }

        SECTION("matrix-of-vectors") {
            pmov->scale("i,j;k", "i,j;k", *rhs_pimpl, 2.0);
            buffer_type rhs(rhs_pimpl->clone());
            mov.add("i,j;k", "i,j;k", out, "i,j;k", rhs);
            pmov->add("i,j;k", "i,j;k", *out_pimpl, "i,j;k", *rhs_pimpl);
            buffer_type corr(std::move(out_pimpl));
            REQUIRE(out == corr);
        }

        SECTION("throws if this is not initialized") {
            using error_t = std::runtime_error;
            REQUIRE_THROWS_AS(defaulted.add("i;j", "i;j", out, "i;j", vov),
                              error_t);
        }

        SECTION("throws if rhs is not initialized") {
            using error_t = std::runtime_error;
            REQUIRE_THROWS_AS(vov.add("i;j", "i;j", out, "i;j", defaulted),
                              error_t);
        }
    }

    SECTION("inplace_add") {
        auto rhs_pimpl = std::make_unique<pimpl_type>();

        SECTION("vector-of-vectors") {
            pvov->scale("i;j", "i;j", *rhs_pimpl, 2.0);
            buffer_type rhs(rhs_pimpl->clone());
            vov.inplace_add("i;j", "i;j", rhs);
            pvov->inplace_add("i;j", "i;j", *rhs_pimpl);
            buffer_type corr(std::move(pvov));
            REQUIRE(vov == corr);
        }

        SECTION("vector-of-matrices") {
            pvom->scale("i;j,k", "i;j,k", *rhs_pimpl, 2.0);
            buffer_type rhs(rhs_pimpl->clone());
            vom.inplace_add("i;j,k", "i;j,k", rhs);
            pvom->inplace_add("i;j,k", "i;j,k", *rhs_pimpl);
            buffer_type corr(std::move(pvom));
            REQUIRE(vom == corr);
        }

        SECTION("matrix-of-vectors") {
            pmov->scale("i,j;k", "i,j;k", *rhs_pimpl, 2.0);
            buffer_type rhs(rhs_pimpl->clone());
            mov.inplace_add("i,j;k", "i,j;k", rhs);
            pmov->inplace_add("i,j;k", "i,j;k", *rhs_pimpl);
            buffer_type corr(std::move(pmov));
            REQUIRE(mov == corr);
        }

        SECTION("throws if this is not initialized") {
            using error_t = std::runtime_error;
            REQUIRE_THROWS_AS(defaulted.inplace_add("i;j", "i;j", vov),
                              error_t);
        }

        SECTION("throws if rhs is not initialized") {
            using error_t = std::runtime_error;
            REQUIRE_THROWS_AS(vov.inplace_add("i;j", "i;j", defaulted),
                              error_t);
        }
    }

    SECTION("subtract") {
        buffer_type out;
        auto out_pimpl = std::make_unique<pimpl_type>();
        auto rhs_pimpl = std::make_unique<pimpl_type>();

        SECTION("vector-of-vectors") {
            pvov->scale("i;j", "i;j", *rhs_pimpl, 2.0);
            buffer_type rhs(rhs_pimpl->clone());
            vov.subtract("i;j", "i;j", out, "i;j", rhs);
            pvov->subtract("i;j", "i;j", *out_pimpl, "i;j", *rhs_pimpl);
            buffer_type corr(std::move(out_pimpl));
            REQUIRE(out == corr);
        }

        SECTION("vector-of-matrices") {
            pvom->scale("i;j,k", "i;j,k", *rhs_pimpl, 2.0);
            buffer_type rhs(rhs_pimpl->clone());
            vom.subtract("i;j,k", "i;j,k", out, "i;j,k", rhs);
            pvom->subtract("i;j,k", "i;j,k", *out_pimpl, "i;j,k", *rhs_pimpl);
            buffer_type corr(std::move(out_pimpl));
            REQUIRE(out == corr);
        }

        SECTION("matrix-of-vectors") {
            pmov->scale("i,j;k", "i,j;k", *rhs_pimpl, 2.0);
            buffer_type rhs(rhs_pimpl->clone());
            mov.subtract("i,j;k", "i,j;k", out, "i,j;k", rhs);
            pmov->subtract("i,j;k", "i,j;k", *out_pimpl, "i,j;k", *rhs_pimpl);
            buffer_type corr(std::move(out_pimpl));
            REQUIRE(out == corr);
        }

        SECTION("throws if this is not initialized") {
            using error_t = std::runtime_error;
            REQUIRE_THROWS_AS(defaulted.subtract("i;j", "i;j", out, "i;j", vov),
                              error_t);
        }

        SECTION("throws if rhs is not initialized") {
            using error_t = std::runtime_error;
            REQUIRE_THROWS_AS(vov.subtract("i;j", "i;j", out, "i;j", defaulted),
                              error_t);
        }
    }

    SECTION("inplace_subtract") {
        auto rhs_pimpl = std::make_unique<pimpl_type>();

        SECTION("vector-of-vectors") {
            pvov->scale("i;j", "i;j", *rhs_pimpl, 2.0);
            buffer_type rhs(rhs_pimpl->clone());
            vov.inplace_subtract("i;j", "i;j", rhs);
            pvov->inplace_subtract("i;j", "i;j", *rhs_pimpl);
            buffer_type corr(std::move(pvov));
            REQUIRE(vov == corr);
        }

        SECTION("vector-of-matrices") {
            pvom->scale("i;j,k", "i;j,k", *rhs_pimpl, 2.0);
            buffer_type rhs(rhs_pimpl->clone());
            vom.inplace_subtract("i;j,k", "i;j,k", rhs);
            pvom->inplace_subtract("i;j,k", "i;j,k", *rhs_pimpl);
            buffer_type corr(std::move(pvom));
            REQUIRE(vom == corr);
        }

        SECTION("matrix-of-vectors") {
            pmov->scale("i,j;k", "i,j;k", *rhs_pimpl, 2.0);
            buffer_type rhs(rhs_pimpl->clone());
            mov.inplace_subtract("i,j;k", "i,j;k", rhs);
            pmov->inplace_subtract("i,j;k", "i,j;k", *rhs_pimpl);
            buffer_type corr(std::move(pmov));
            REQUIRE(mov == corr);
        }

        SECTION("throws if this is not initialized") {
            using error_t = std::runtime_error;
            REQUIRE_THROWS_AS(defaulted.inplace_subtract("i;j", "i;j", vov),
                              error_t);
        }

        SECTION("throws if rhs is not initialized") {
            using error_t = std::runtime_error;
            REQUIRE_THROWS_AS(vov.inplace_subtract("i;j", "i;j", defaulted),
                              error_t);
        }
    }

    SECTION("print") {
        std::stringstream ss;
        auto pss = &(vov.print(ss));
        SECTION("Returns ss for chaining") { REQUIRE(pss == &ss); }
        SECTION("Value") {
            std::string corr = "0: [ [0], [3] ) {\n"
                               "  [0]:[ [0], [3] ) { 1 2 3 }\n"
                               "  [1]:[ [0], [3] ) { 1 2 3 }\n"
                               "  [2]:[ [0], [3] ) { 1 2 3 }\n"
                               "}\n";
            REQUIRE(corr == ss.str());
        }
    }

    SECTION("hash") {
        using parallelzone::hash_objects;
        REQUIRE(hash_objects(defaulted) == hash_objects(buffer_type{}));

        // TODO: Reenable when hashing includes types
        using other_buffer = buffer::Buffer<field::Scalar>;
        // REQUIRE_FALSE(hash_objects(defaulted) ==
        // hash_objects(other_buffer{}));

        REQUIRE_FALSE(hash_objects(defaulted) == hash_objects(vov));

        REQUIRE_FALSE(hash_objects(vov) == hash_objects(vom));
    }

    SECTION("Comparisons") {
        REQUIRE(defaulted == buffer_type{});
        REQUIRE_FALSE(defaulted != buffer_type{});

        REQUIRE_FALSE(defaulted == buffer::Buffer<field::Scalar>{});
        REQUIRE(defaulted != buffer::Buffer<field::Scalar>{});

        REQUIRE_FALSE(defaulted == vov);
        REQUIRE(defaulted != vov);

        REQUIRE_FALSE(vov == mov);
        REQUIRE(vov != mov);
    }
}

TEST_CASE("operator<<(std::ostream, Buffer<Tensor>)") {
    using field_type  = field::Tensor;
    using buffer_type = buffer::Buffer<field_type>;

    auto&& [pvov, pvom, pmov] = testing::make_pimpl<field_type>();
    buffer_type vov(pvov->clone());

    std::stringstream ss;
    auto pss = &(ss << vov);
    SECTION("Returns ss for chaining") { REQUIRE(pss == &ss); }
    SECTION("Value") {
        std::string corr = "0: [ [0], [3] ) {\n"
                           "  [0]:[ [0], [3] ) { 1 2 3 }\n"
                           "  [1]:[ [0], [3] ) { 1 2 3 }\n"
                           "  [2]:[ [0], [3] ) { 1 2 3 }\n"
                           "}\n";
        REQUIRE(corr == ss.str());
    }
}
