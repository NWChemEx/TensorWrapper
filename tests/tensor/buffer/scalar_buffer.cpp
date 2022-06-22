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

TEST_CASE("Buffer<Scalar>") {
    using field_type  = field::Scalar;
    using buffer_type = buffer::Buffer<field_type>;
    using pimpl_type  = buffer::detail_::TABufferPIMPL<field_type>;
    using tensor_type = typename pimpl_type::default_tensor_type;

    buffer_type defaulted;
    auto&& [pvec, pmat, pt3d] = testing::make_pimpl<field_type>();
    buffer_type vec(pvec->clone());
    buffer_type mat(pmat->clone());
    buffer_type t3d(pt3d->clone());

    SECTION("CTors") {
        SECTION("Default") { REQUIRE_FALSE(defaulted.is_initialized()); }
        SECTION("Copy") {
            buffer_type v2(vec);
            REQUIRE(v2.is_initialized());
            REQUIRE(v2 == vec);
        }
        SECTION("Move") {
            buffer_type corr(vec);
            buffer_type v2(std::move(vec));
            REQUIRE(v2.is_initialized());
            REQUIRE_FALSE(vec.is_initialized());
            REQUIRE(v2 == corr);
        }
        SECTION("Copy assignment") {
            buffer_type v2;
            auto pv2 = &(v2 = vec);
            REQUIRE(pv2 == &v2);
            REQUIRE(v2.is_initialized());
            REQUIRE(v2 == vec);
        }
        SECTION("Move assignment") {
            buffer_type v2;
            buffer_type corr(vec);
            auto pv2 = &(v2 = std::move(vec));
            REQUIRE(pv2 == &v2);
            REQUIRE_FALSE(vec.is_initialized());
            REQUIRE(v2.is_initialized());
            REQUIRE(v2 == corr);
        }
    }

    SECTION("Scale") {
        buffer_type out;
        auto out_pimpl = std::make_unique<pimpl_type>();
        SECTION("vector") {
            vec.scale("i", "i", out, 2.0);
            pvec->scale("i", "i", *out_pimpl, 2.0);
            buffer_type corr(std::move(out_pimpl));
            REQUIRE(out == corr);
        }
        SECTION("matrix") {
            mat.scale("i,j", "i,j", out, 2.0);
            pmat->scale("i,j", "i,j", *out_pimpl, 2.0);
            buffer_type corr(std::move(out_pimpl));
            REQUIRE(out == corr);
        }
        SECTION("tensor") {
            t3d.scale("i,j,k", "i,j,k", out, 2.0);
            pt3d->scale("i,j,k", "i,j,k", *out_pimpl, 2.0);
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
        SECTION("vector") {
            pvec->scale("i", "i", *rhs_pimpl, 2.0);
            buffer_type rhs(rhs_pimpl->clone());
            vec.add("i", "i", out, "i", rhs);
            pvec->add("i", "i", *out_pimpl, "i", *rhs_pimpl);
            buffer_type corr(std::move(out_pimpl));
            REQUIRE(out == corr);
        }
        SECTION("matrix") {
            pmat->scale("i,j", "i,j", *rhs_pimpl, 2.0);
            buffer_type rhs(rhs_pimpl->clone());
            mat.add("i,j", "i,j", out, "i,j", rhs);
            pmat->add("i,j", "i,j", *out_pimpl, "i,j", *rhs_pimpl);
            buffer_type corr(std::move(out_pimpl));
            REQUIRE(out == corr);
        }
        SECTION("tensor") {
            pt3d->scale("i,j,k", "i,j,k", *rhs_pimpl, 2.0);
            buffer_type rhs(rhs_pimpl->clone());
            t3d.add("i,j,k", "i,j,k", out, "i,j,k", rhs);
            pt3d->add("i,j,k", "i,j,k", *out_pimpl, "i,j,k", *rhs_pimpl);
            buffer_type corr(std::move(out_pimpl));
            REQUIRE(out == corr);
        }

        SECTION("throws if this is not initialized") {
            using error_t = std::runtime_error;
            REQUIRE_THROWS_AS(defaulted.add("i", "i", out, "i", vec), error_t);
        }

        SECTION("throws if rhs is not initialized") {
            using error_t = std::runtime_error;
            REQUIRE_THROWS_AS(vec.add("i", "i", out, "i", defaulted), error_t);
        }
    }

    SECTION("inplace_add") {
        auto rhs_pimpl = std::make_unique<pimpl_type>();
        SECTION("vector") {
            pvec->scale("i", "i", *rhs_pimpl, 2.0);
            buffer_type rhs(rhs_pimpl->clone());
            vec.inplace_add("i", "i", rhs);
            pvec->inplace_add("i", "i", *rhs_pimpl);
            buffer_type corr(std::move(pvec));
            REQUIRE(vec == corr);
        }
        SECTION("matrix") {
            pmat->scale("i,j", "i,j", *rhs_pimpl, 2.0);
            buffer_type rhs(rhs_pimpl->clone());
            mat.inplace_add("i,j", "i,j", rhs);
            pmat->inplace_add("i,j", "i,j", *rhs_pimpl);
            buffer_type corr(std::move(pmat));
            REQUIRE(mat == corr);
        }
        SECTION("tensor") {
            pt3d->scale("i,j,k", "i,j,k", *rhs_pimpl, 2.0);
            buffer_type rhs(rhs_pimpl->clone());
            t3d.inplace_add("i,j,k", "i,j,k", rhs);
            pt3d->inplace_add("i,j,k", "i,j,k", *rhs_pimpl);
            buffer_type corr(std::move(pt3d));
            REQUIRE(t3d == corr);
        }

        SECTION("throws if this is not initialized") {
            using error_t = std::runtime_error;
            REQUIRE_THROWS_AS(defaulted.inplace_add("i", "i", vec), error_t);
        }

        SECTION("throws if rhs is not initialized") {
            using error_t = std::runtime_error;
            REQUIRE_THROWS_AS(vec.inplace_add("i", "i", defaulted), error_t);
        }
    }

    SECTION("subtract") {
        buffer_type out;
        auto out_pimpl = std::make_unique<pimpl_type>();
        auto rhs_pimpl = std::make_unique<pimpl_type>();
        SECTION("vector") {
            pvec->scale("i", "i", *rhs_pimpl, 2.0);
            buffer_type rhs(rhs_pimpl->clone());
            vec.subtract("i", "i", out, "i", rhs);
            pvec->subtract("i", "i", *out_pimpl, "i", *rhs_pimpl);
            buffer_type corr(std::move(out_pimpl));
            REQUIRE(out == corr);
        }
        SECTION("matrix") {
            pmat->scale("i,j", "i,j", *rhs_pimpl, 2.0);
            buffer_type rhs(rhs_pimpl->clone());
            mat.subtract("i,j", "i,j", out, "i,j", rhs);
            pmat->subtract("i,j", "i,j", *out_pimpl, "i,j", *rhs_pimpl);
            buffer_type corr(std::move(out_pimpl));
            REQUIRE(out == corr);
        }
        SECTION("tensor") {
            pt3d->scale("i,j,k", "i,j,k", *rhs_pimpl, 2.0);
            buffer_type rhs(rhs_pimpl->clone());
            t3d.subtract("i,j,k", "i,j,k", out, "i,j,k", rhs);
            pt3d->subtract("i,j,k", "i,j,k", *out_pimpl, "i,j,k", *rhs_pimpl);
            buffer_type corr(std::move(out_pimpl));
            REQUIRE(out == corr);
        }

        SECTION("throws if this is not initialized") {
            using error_t = std::runtime_error;
            REQUIRE_THROWS_AS(defaulted.subtract("i", "i", out, "i", vec),
                              error_t);
        }

        SECTION("throws if rhs is not initialized") {
            using error_t = std::runtime_error;
            REQUIRE_THROWS_AS(vec.subtract("i", "i", out, "i", defaulted),
                              error_t);
        }
    }

    SECTION("inplace_subtract") {
        auto rhs_pimpl = std::make_unique<pimpl_type>();
        SECTION("vector") {
            pvec->scale("i", "i", *rhs_pimpl, 2.0);
            buffer_type rhs(rhs_pimpl->clone());
            vec.inplace_subtract("i", "i", rhs);
            pvec->inplace_subtract("i", "i", *rhs_pimpl);
            buffer_type corr(std::move(pvec));
            REQUIRE(vec == corr);
        }
        SECTION("matrix") {
            pmat->scale("i,j", "i,j", *rhs_pimpl, 2.0);
            buffer_type rhs(rhs_pimpl->clone());
            mat.inplace_subtract("i,j", "i,j", rhs);
            pmat->inplace_subtract("i,j", "i,j", *rhs_pimpl);
            buffer_type corr(std::move(pmat));
            REQUIRE(mat == corr);
        }
        SECTION("tensor") {
            pt3d->scale("i,j,k", "i,j,k", *rhs_pimpl, 2.0);
            buffer_type rhs(rhs_pimpl->clone());
            t3d.inplace_subtract("i,j,k", "i,j,k", rhs);
            pt3d->inplace_subtract("i,j,k", "i,j,k", *rhs_pimpl);
            buffer_type corr(std::move(pt3d));
            REQUIRE(t3d == corr);
        }

        SECTION("throws if this is not initialized") {
            using error_t = std::runtime_error;
            REQUIRE_THROWS_AS(defaulted.inplace_subtract("i", "i", vec),
                              error_t);
        }

        SECTION("throws if rhs is not initialized") {
            using error_t = std::runtime_error;
            REQUIRE_THROWS_AS(vec.inplace_subtract("i", "i", defaulted),
                              error_t);
        }
    }

    SECTION("times") {
        buffer_type out;
        auto out_pimpl = std::make_unique<pimpl_type>();
        auto rhs_pimpl = std::make_unique<pimpl_type>();
        SECTION("vector") {
            pvec->scale("i", "i", *rhs_pimpl, 2.0);
            buffer_type rhs(rhs_pimpl->clone());
            vec.times("i", "i", out, "i", rhs);
            pvec->times("i", "i", *out_pimpl, "i", *rhs_pimpl);
            buffer_type corr(std::move(out_pimpl));
            REQUIRE(out == corr);
        }
        SECTION("matrix") {
            pmat->scale("i,j", "i,j", *rhs_pimpl, 2.0);
            buffer_type rhs(rhs_pimpl->clone());
            mat.times("i,j", "i,j", out, "i,j", rhs);
            pmat->times("i,j", "i,j", *out_pimpl, "i,j", *rhs_pimpl);
            buffer_type corr(std::move(out_pimpl));
            REQUIRE(out == corr);
        }
        SECTION("tensor") {
            pt3d->scale("i,j,k", "i,j,k", *rhs_pimpl, 2.0);
            buffer_type rhs(rhs_pimpl->clone());
            t3d.times("i,j,k", "i,j,k", out, "i,j,k", rhs);
            pt3d->times("i,j,k", "i,j,k", *out_pimpl, "i,j,k", *rhs_pimpl);
            buffer_type corr(std::move(out_pimpl));
            REQUIRE(out == corr);
        }

        SECTION("throws if this is not initialized") {
            using error_t = std::runtime_error;
            REQUIRE_THROWS_AS(defaulted.times("i", "i", out, "i", vec),
                              error_t);
        }

        SECTION("throws if rhs is not initialized") {
            using error_t = std::runtime_error;
            REQUIRE_THROWS_AS(vec.times("i", "i", out, "i", defaulted),
                              error_t);
        }
    }

    SECTION("norm") {
        SECTION("vector") {
            auto ref_norm = pvec->norm();
            auto norm     = vec.norm();
            REQUIRE(ref_norm == norm);
        }
        SECTION("matrix") {
            auto ref_norm = pmat->norm();
            auto norm     = mat.norm();
            REQUIRE(ref_norm == norm);
        }
        SECTION("tensor") {
            auto ref_norm = pt3d->norm();
            auto norm     = t3d.norm();
            REQUIRE(ref_norm == norm);
        }
        SECTION("throws if this is not initialized") {
            using error_t = std::runtime_error;
            REQUIRE_THROWS_AS(defaulted.norm(), error_t);
        }
    }

    SECTION("sum") {
        SECTION("vector") {
            auto ref_sum = pvec->sum();
            auto sum     = vec.sum();
            REQUIRE(ref_sum == sum);
        }
        SECTION("matrix") {
            auto ref_sum = pmat->sum();
            auto sum     = mat.sum();
            REQUIRE(ref_sum == sum);
        }
        SECTION("tensor") {
            auto ref_sum = pt3d->sum();
            auto sum     = t3d.sum();
            REQUIRE(ref_sum == sum);
        }
        SECTION("throws if this is not initialized") {
            using error_t = std::runtime_error;
            REQUIRE_THROWS_AS(defaulted.sum(), error_t);
        }
    }

    SECTION("trace") {
        SECTION("invalid") {
            REQUIRE_THROWS_AS(vec.trace(), std::runtime_error);
            REQUIRE_THROWS_AS(t3d.trace(), std::runtime_error);
            REQUIRE_THROWS_AS(defaulted.trace(), std::runtime_error);
        }
        SECTION("matrix") {
            auto ref_trace = pmat->trace();
            auto trace     = mat.trace();
            REQUIRE(trace == ref_trace);
        }
    }

    SECTION("make_extents") {
        SECTION("defaulted") {
            REQUIRE_THROWS_AS(defaulted.make_extents(), std::runtime_error);
        }
        SECTION("with value") {
            REQUIRE(vec.make_extents() == std::vector<std::size_t>{3});
            REQUIRE(mat.make_extents() == std::vector<std::size_t>{2, 2});
            REQUIRE(t3d.make_extents() == std::vector<std::size_t>{2, 2, 2});
        }
    }

    SECTION("print") {
        std::stringstream ss;
        auto pss = &(vec.print(ss));
        SECTION("Returns ss for chaining") { REQUIRE(pss == &ss); }
        SECTION("Value") {
            std::string corr = "0: [ [0], [3] ) { 1 2 3 }\n";
            REQUIRE(corr == ss.str());
        }
    }

    SECTION("hash") {
        using parallelzone::hash_objects;
        REQUIRE(hash_objects(defaulted) == hash_objects(buffer_type{}));

        // TODO: Reenable when hashing includes types
        using other_buffer = buffer::Buffer<field::Tensor>;
        // REQUIRE_FALSE(hash_objects(defaulted) ==
        // hash_objects(other_buffer{}));

        REQUIRE_FALSE(hash_objects(defaulted) == hash_objects(vec));

        REQUIRE_FALSE(hash_objects(vec) == hash_objects(mat));
    }

    SECTION("Comparisons") {
        REQUIRE(defaulted == buffer_type{});
        REQUIRE_FALSE(defaulted != buffer_type{});

        REQUIRE_FALSE(defaulted == buffer::Buffer<field::Tensor>{});
        REQUIRE(defaulted != buffer::Buffer<field::Tensor>{});

        REQUIRE_FALSE(defaulted == vec);
        REQUIRE(defaulted != vec);

        REQUIRE_FALSE(vec == mat);
        REQUIRE(vec != mat);
    }
}

TEST_CASE("operator<<(std::ostream, Buffer<Scalar>") {
    using field_type  = field::Scalar;
    using buffer_type = buffer::Buffer<field_type>;

    auto&& [pvec, pmat, pt3d] = testing::make_pimpl<field_type>();
    buffer_type vec(pvec->clone());

    std::stringstream ss;
    auto pss = &(ss << vec);
    SECTION("Returns ss for chaining") { REQUIRE(pss == &ss); }
    SECTION("Value") {
        std::string corr = "0: [ [0], [3] ) { 1 2 3 }\n";
        REQUIRE(corr == ss.str());
    }
}
