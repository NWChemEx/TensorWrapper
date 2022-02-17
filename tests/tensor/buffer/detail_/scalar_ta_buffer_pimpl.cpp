#include "tensorwrapper/tensor/buffer/detail_/ta_buffer_pimpl.hpp"

#include "../../test_tensor.hpp"

using namespace tensorwrapper::tensor;

/*
 * Unit testing notes:
 *
 * We assume TA works. What this means is we don't necessarily need to try all
 * sorts of say adds (i.e., with permutations, without permutations, with
 * scaling, etc.). Rather we're testing that the information gets forwarded
 * correctly.
 */

TEST_CASE("TABufferPIMPL<Scalar>") {
    using field_type    = field::Scalar;
    using buffer_type   = buffer::detail_::TABufferPIMPL<field_type>;
    using tensor_type   = typename buffer_type::default_tensor_type;
    using trange_type   = typename buffer_type::ta_trange_type;
    using ta_shape_type = typename buffer_type::ta_shape_type;

    auto& world = TA::get_default_world();
    tensor_type vec_ta(world, {1.0, 2.0, 3.0});
    tensor_type mat_ta(world, {{1.0, 2.0}, {3.0, 4.0}});
    tensor_type t3d_ta(world,
                       {{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}});

    buffer_type vec(vec_ta);
    buffer_type mat(mat_ta);
    buffer_type t3d(t3d_ta);

    SECTION("default_clone()") {
        buffer_type defaulted;
        REQUIRE(vec.default_clone()->are_equal(defaulted));
    }

    SECTION("clone()") {
        auto vec2 = vec.clone();
        REQUIRE(vec2->are_equal(vec));

        auto mat2 = mat.clone();
        REQUIRE(mat2->are_equal(mat));

        auto t3d2 = t3d.clone();
        REQUIRE(t3d2->are_equal(t3d));
    }

    SECTION("retile") {
        // These may need to use allclose XXX: these should be fine, no ambiguity on copies
        SECTION("vector") {
            trange_type tr{{0, 1, 2, 3}};
            vec.retile(tr);
            buffer_type corr(tensor_type(world, tr, {1.0, 2.0, 3.0}));
            REQUIRE(vec.are_equal(corr));
        }
        SECTION("matrix") {
            trange_type tr{{0, 1, 2}, {0, 1, 2}};
            mat.retile(tr);
            buffer_type corr(tensor_type(world, tr, {{1.0, 2.0}, {3.0, 4.0}}));
            REQUIRE(mat.are_equal(corr));
        }
        SECTION("tensor") {
            trange_type tr{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}};
            t3d.retile(tr);
            buffer_type corr(tensor_type(
              world, tr, {{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}));
            REQUIRE(t3d.are_equal(corr));
        }
    }

    SECTION("set_shape") {
        auto max = std::numeric_limits<float>::max();
        SECTION("vector") {
            trange_type tr{{0, 1, 2, 3}};
            TA::Tensor<float> tile_norms(TA::Range{{0, 3}}, {max, 0.0, max});
            ta_shape_type ss(tile_norms, tr);
            vec.retile(tr);
            vec.set_shape(ss);
            buffer_type corr(tensor_type(world, tr, {1.0, 0.0, 3.0}));
            REQUIRE(vec.are_equal(corr));
        }
        SECTION("matrix") {
            trange_type tr{{0, 1, 2}, {0, 1, 2}};
            TA::Range r{{0, 2}, {0, 2}};
            TA::Tensor<float> tile_norms(r, {max, 0.0, max, 0.0});
            ta_shape_type ss(tile_norms, tr);
            mat.retile(tr);
            mat.set_shape(ss);
            buffer_type corr(tensor_type(world, tr, {{1.0, 0.0}, {3.0, 0.0}}));
            REQUIRE(mat.are_equal(corr));
        }
        SECTION("tensor") {
            trange_type tr{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}};
            TA::Range r{{0, 2}, {0, 2}, {0, 2}};
            TA::Tensor<float> tile_norms(
              r, {max, 0.0, max, 0.0, max, 0.0, max, 0.0});
            ta_shape_type ss(tile_norms, tr);
            t3d.retile(tr);
            t3d.set_shape(ss);
            buffer_type corr(tensor_type(
              world, tr, {{{1.0, 0.0}, {3.0, 0.0}}, {{5.0, 0.0}, {7.0, 0.0}}}));
            REQUIRE(t3d.are_equal(corr));
        }
    }

    // For these tests we do exactly the same operations under the hood so
    // we should be able to achieve value equality
    SECTION("scale") {
        tensor_type out_ta;
        buffer_type out;
        SECTION("vector") {
            vec.scale("i", "i", out, 2.0);
            out_ta("i") = 2.0 * vec_ta("i");
            REQUIRE(out.are_equal(buffer_type(out_ta)));
        }

        SECTION("matrix") {
            mat.scale("i,j", "i,j", out, 2.0);
            out_ta("i,j") = 2.0 * mat_ta("i,j");
            REQUIRE(out.are_equal(buffer_type(out_ta)));
        }

        SECTION("tensor") {
            t3d.scale("i,j,k", "i,j,k", out, 2.0);
            out_ta("i,j,k") = 2.0 * t3d_ta("i,j,k");
            REQUIRE(out.are_equal(buffer_type(out_ta)));
        }
    }

    SECTION("add") {
        tensor_type out_ta, rhs_ta;

        SECTION("vector") {
            rhs_ta("i") = 2.0 * vec_ta("i");
            buffer_type out, rhs(rhs_ta);

            vec.add("i", "i", out, "i", rhs);
            out_ta("i") = vec_ta("i") + rhs_ta("i");
            REQUIRE(out.are_equal(buffer_type(out_ta)));
        }

        SECTION("matrix") {
            rhs_ta("i,j") = 2.0 * mat_ta("i,j");
            buffer_type out, rhs(rhs_ta);

            mat.add("i,j", "i,j", out, "i,j", rhs);
            out_ta("i,j") = mat_ta("i,j") + rhs_ta("i,j");
            REQUIRE(out.are_equal(buffer_type(out_ta)));
        }

        SECTION("tensor") {
            rhs_ta("i,j,k") = 2.0 * t3d_ta("i,j,k");
            buffer_type out, rhs(rhs_ta);

            t3d.add("i,j,k", "i,j,k", out, "i,j,k", rhs);
            out_ta("i,j,k") = t3d_ta("i,j,k") + rhs_ta("i,j,k");
            REQUIRE(out.are_equal(buffer_type(out_ta)));
        }
    }

    SECTION("inplace_add") {
        tensor_type rhs_ta;

        SECTION("vector") {
            rhs_ta("i") = 2.0 * vec_ta("i");
            buffer_type rhs(rhs_ta);

            vec.inplace_add("i", "i", rhs);
            vec_ta("i") += rhs_ta("i");
            REQUIRE(vec.are_equal(buffer_type(vec_ta)));
        }

        SECTION("matrix") {
            rhs_ta("i,j") = 2.0 * mat_ta("i,j");
            buffer_type rhs(rhs_ta);

            mat.inplace_add("i,j", "i,j", rhs);
            mat_ta("i,j") += rhs_ta("i,j");
            REQUIRE(mat.are_equal(buffer_type(mat_ta)));
        }

        SECTION("tensor") {
            rhs_ta("i,j,k") = 2.0 * t3d_ta("i,j,k");
            buffer_type rhs(rhs_ta);

            t3d.inplace_add("i,j,k", "i,j,k", rhs);
            t3d_ta("i,j,k") += rhs_ta("i,j,k");
            REQUIRE(t3d.are_equal(buffer_type(t3d_ta)));
        }
    }

    SECTION("subtract") {
        tensor_type out_ta, rhs_ta;

        SECTION("vector") {
            rhs_ta("i") = 2.0 * vec_ta("i");
            buffer_type out, rhs(rhs_ta);

            vec.subtract("i", "i", out, "i", rhs);
            out_ta("i") = vec_ta("i") - rhs_ta("i");
            REQUIRE(out.are_equal(buffer_type(out_ta)));
        }

        SECTION("matrix") {
            rhs_ta("i,j") = 2.0 * mat_ta("i,j");
            buffer_type out, rhs(rhs_ta);

            mat.subtract("i,j", "i,j", out, "i,j", rhs);
            out_ta("i,j") = mat_ta("i,j") - rhs_ta("i,j");
            REQUIRE(out.are_equal(buffer_type(out_ta)));
        }

        SECTION("tensor") {
            rhs_ta("i,j,k") = 2.0 * t3d_ta("i,j,k");
            buffer_type out, rhs(rhs_ta);

            t3d.subtract("i,j,k", "i,j,k", out, "i,j,k", rhs);
            out_ta("i,j,k") = t3d_ta("i,j,k") - rhs_ta("i,j,k");
            REQUIRE(out.are_equal(buffer_type(out_ta)));
        }
    }

    SECTION("inplace_subtract") {
        tensor_type rhs_ta;

        SECTION("vector") {
            rhs_ta("i") = 2.0 * vec_ta("i");
            buffer_type rhs(rhs_ta);

            vec.inplace_subtract("i", "i", rhs);
            vec_ta("i") -= rhs_ta("i");
            REQUIRE(vec.are_equal(buffer_type(vec_ta)));
        }

        SECTION("matrix") {
            rhs_ta("i,j") = 2.0 * mat_ta("i,j");
            buffer_type rhs(rhs_ta);

            mat.inplace_subtract("i,j", "i,j", rhs);
            mat_ta("i,j") -= rhs_ta("i,j");
            REQUIRE(mat.are_equal(buffer_type(mat_ta)));
        }

        SECTION("tensor") {
            rhs_ta("i,j,k") = 2.0 * t3d_ta("i,j,k");
            buffer_type rhs(rhs_ta);

            t3d.inplace_subtract("i,j,k", "i,j,k", rhs);
            t3d_ta("i,j,k") -= rhs_ta("i,j,k");
            REQUIRE(t3d.are_equal(buffer_type(t3d_ta)));
        }
    }

    SECTION("times") {
        tensor_type out_ta, rhs_ta;

        SECTION("vector") {
            rhs_ta("i") = 2.0 * vec_ta("i");
            buffer_type out, rhs(rhs_ta);

            vec.times("i", "i", out, "i", rhs);
            out_ta("i") = vec_ta("i") * rhs_ta("i");
            REQUIRE(out.are_equal(buffer_type(out_ta)));
        }

        SECTION("matrix") {
            rhs_ta("i,j") = 2.0 * mat_ta("i,j");
            buffer_type out, rhs(rhs_ta);

            mat.times("i,j", "i,j", out, "i,j", rhs);
            out_ta("i,j") = mat_ta("i,j") * rhs_ta("i,j");
            REQUIRE(out.are_equal(buffer_type(out_ta)));
        }

        SECTION("tensor") {
            rhs_ta("i,j,k") = 2.0 * t3d_ta("i,j,k");
            buffer_type out, rhs(rhs_ta);

            t3d.times("i,j,k", "i,j,k", out, "i,j,k", rhs);
            out_ta("i,j,k") = t3d_ta("i,j,k") * rhs_ta("i,j,k");
            REQUIRE(out.are_equal(buffer_type(out_ta)));
        }
    }

    SECTION("operator std::string") {
        std::string corr = "0: [ [0], [3] ) { 1 2 3 }\n";
        REQUIRE(corr == std::string(vec));
    }

    SECTION("operator<<") {
        std::stringstream ss;
        auto pss = &(ss << vec);
        SECTION("Returns ss for chaining") { REQUIRE(pss == &ss); }
        SECTION("Value") {
            std::string corr = "0: [ [0], [3] ) { 1 2 3 }\n";
            REQUIRE(corr == ss.str());
        }
    }

    SECTION("hash") {
        using tensorwrapper::detail_::hash_objects;
        auto lhs = hash_objects(vec);

        SECTION("Are same") {
            REQUIRE(lhs == hash_objects(buffer_type(vec_ta)));
        }
        SECTION("Different") { REQUIRE(lhs != hash_objects(mat)); }
    }

    SECTION("are_equal") {
        SECTION("Are same") {
            buffer_type other_vec(vec_ta);
            REQUIRE(vec.are_equal(other_vec));
        }
        SECTION("Different") { REQUIRE_FALSE(vec.are_equal(mat)); }
    }
}
