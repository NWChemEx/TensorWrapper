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

TEST_CASE("TABufferPIMPL<Tensor>") {
    using field_type    = field::Tensor;
    using buffer_type   = buffer::detail_::TABufferPIMPL<field_type>;
    using tensor_type   = typename buffer_type::default_tensor_type;
    using tile_type     = typename tensor_type::value_type;
    using inner_tile    = typename tile_type::value_type;
    using trange_type   = typename buffer_type::ta_trange_type;
    using ta_shape_type = typename buffer_type::ta_shape_type;

    auto& world = TA::get_default_world();
    inner_tile v0(TA::Range{3}, {1.0, 2.0, 3.0});
    inner_tile m0(TA::Range{2, 2}, {1.0, 2.0, 3.0, 4.0});

    tensor_type vov_ta(world, {v0, v0, v0});
    tensor_type vom_ta(world, {m0, m0, m0});
    tensor_type mov_ta(world, {{v0, v0}, {v0, v0}});

    buffer_type vov(vov_ta);
    buffer_type vom(vom_ta);
    buffer_type mov(mov_ta);

    SECTION("clone()") {
        auto vov2 = vov.clone();
        REQUIRE(vov2->are_equal(vov));

        auto vom2 = vom.clone();
        REQUIRE(vom2->are_equal(vom));

        auto mov2 = mov.clone();
        REQUIRE(mov2->are_equal(mov));
    }

    SECTION("retile") {
        TA::TiledRange tr{{0, 1, 2, 3}};
        REQUIRE_THROWS_AS(vov.retile(tr), std::runtime_error);
    }

    SECTION("set_shape") {
        auto max = std::numeric_limits<float>::max();
        SECTION("vector-of-vectors") {
            trange_type tr{{0, 1, 2, 3}};
            TA::Tensor<float> tile_norms(TA::Range{{0, 3}}, {max, 0.0, max});
            ta_shape_type ss(tile_norms, tr);
            buffer_type vov2(tensor_type(world, tr, {v0, v0, v0}));
            vov2.set_shape(ss);
            buffer_type corr(tensor_type(world, tr, {v0, inner_tile{}, v0}));
            REQUIRE(vov2.are_equal(corr));
        }
        SECTION("vector-of-matrices") {
            trange_type tr{{0, 1, 2, 3}};
            TA::Tensor<float> tile_norms(TA::Range{{0, 3}}, {max, 0.0, max});
            ta_shape_type ss(tile_norms, tr);
            buffer_type vom2(tensor_type(world, tr, {m0, m0, m0}));
            vom2.set_shape(ss);
            buffer_type corr(tensor_type(world, tr, {m0, inner_tile{}, m0}));
            REQUIRE(vom2.are_equal(corr));
        }
        SECTION("matrix-of-vectors") {
            trange_type tr{{0, 1, 2}, {0, 1, 2}};
            TA::Tensor<float> tile_norms(TA::Range{{0, 2}, {0, 2}},
                                         {max, 0.0, max, 0.0});
            ta_shape_type ss(tile_norms, tr);
            buffer_type mov2(tensor_type(world, tr, {{v0, v0}, {v0, v0}}));
            mov2.set_shape(ss);
            buffer_type corr(
              tensor_type(world, tr, {{v0, inner_tile{}}, {v0, inner_tile{}}}));
            REQUIRE(mov2.are_equal(corr));
        }
    }

    // For these tests we do exactly the same operations under the hood so
    // we should be able to achieve value equality
    SECTION("scale") {
        tensor_type out_ta;
        buffer_type out;
        SECTION("vector-of-vectors") {
            vov.scale("i;j", "i;j", out, 2.0);
            out_ta("i;j") = 2.0 * vov_ta("i;j");
            REQUIRE(out.are_equal(buffer_type(out_ta)));
        }

        SECTION("vector-of-matrices") {
            vom.scale("i;j,k", "i;j,k", out, 2.0);
            out_ta("i;j,k") = 2.0 * vom_ta("i;j,k");
            REQUIRE(out.are_equal(buffer_type(out_ta)));
        }

        SECTION("matrix-of-vectors") {
            mov.scale("i,j;k", "i,j;k", out, 2.0);
            out_ta("i,j;k") = 2.0 * mov_ta("i,j;k");
            REQUIRE(out.are_equal(buffer_type(out_ta)));
        }
    }

    SECTION("add") {
        tensor_type out_ta, rhs_ta;

        SECTION("vector-of-vectors") {
            rhs_ta("i;j") = 2.0 * vov_ta("i;j");
            buffer_type out, rhs(rhs_ta);

            vov.add("i;j", "i;j", out, "i;j", rhs);
            out_ta("i;j") = vov_ta("i;j") + rhs_ta("i;j");
            REQUIRE(out.are_equal(buffer_type(out_ta)));
        }

        SECTION("vector-of-matrices") {
            rhs_ta("i;j,k") = 2.0 * vom_ta("i;j,k");
            buffer_type out, rhs(rhs_ta);

            vom.add("i;j,k", "i;j,k", out, "i;j,k", rhs);
            out_ta("i;j,k") = vom_ta("i;j,k") + rhs_ta("i;j,k");
            REQUIRE(out.are_equal(buffer_type(out_ta)));
        }

        SECTION("matrix-of-vectors") {
            rhs_ta("i,j;k") = 2.0 * mov_ta("i,j;k");
            buffer_type out, rhs(rhs_ta);

            mov.add("i,j;k", "i,j;k", out, "i,j;k", rhs);
            out_ta("i,j;k") = mov_ta("i,j;k") + rhs_ta("i,j;k");
            REQUIRE(out.are_equal(buffer_type(out_ta)));
        }
    }

    SECTION("inplace_add") {
        tensor_type rhs_ta;

        SECTION("vector-of-vectors") {
            rhs_ta("i;j") = 2.0 * vov_ta("i;j");
            buffer_type rhs(rhs_ta);

            vov.inplace_add("i;j", "i;j", rhs);
            vov_ta("i;j") += rhs_ta("i;j");
            REQUIRE(vov.are_equal(buffer_type(vov_ta)));
        }

        SECTION("vector-of-matrices") {
            rhs_ta("i;j,k") = 2.0 * vom_ta("i;j,k");
            buffer_type rhs(rhs_ta);

            vom.inplace_add("i;j,k", "i;j,k", rhs);
            vom_ta("i;j,k") += rhs_ta("i;j,k");
            REQUIRE(vom.are_equal(buffer_type(vom_ta)));
        }

        SECTION("matrix-of-vectors") {
            rhs_ta("i,j;k") = 2.0 * mov_ta("i,j;k");
            buffer_type rhs(rhs_ta);

            mov.inplace_add("i,j;k", "i,j;k", rhs);
            mov_ta("i,j;k") += rhs_ta("i,j;k");
            REQUIRE(mov.are_equal(buffer_type(mov_ta)));
        }
    }

    SECTION("subtract") {
        tensor_type out_ta, rhs_ta;

        SECTION("vector-of-vectors") {
            rhs_ta("i;j") = 2.0 * vov_ta("i;j");
            buffer_type out, rhs(rhs_ta);

            vov.subtract("i;j", "i;j", out, "i;j", rhs);
            out_ta("i;j") = vov_ta("i;j") - rhs_ta("i;j");
            REQUIRE(out.are_equal(buffer_type(out_ta)));
        }

        SECTION("vector-of-matrices") {
            rhs_ta("i;j,k") = 2.0 * vom_ta("i;j,k");
            buffer_type out, rhs(rhs_ta);

            vom.subtract("i;j,k", "i;j,k", out, "i;j,k", rhs);
            out_ta("i;j,k") = vom_ta("i;j,k") - rhs_ta("i;j,k");
            REQUIRE(out.are_equal(buffer_type(out_ta)));
        }

        SECTION("matrix-of-vectors") {
            rhs_ta("i,j;k") = 2.0 * mov_ta("i,j;k");
            buffer_type out, rhs(rhs_ta);

            mov.subtract("i,j;k", "i,j;k", out, "i,j;k", rhs);
            out_ta("i,j;k") = mov_ta("i,j;k") - rhs_ta("i,j;k");
            REQUIRE(out.are_equal(buffer_type(out_ta)));
        }
    }

    SECTION("inplace_subtract") {
        tensor_type rhs_ta;

        SECTION("vector-of-vectors") {
            rhs_ta("i;j") = 2.0 * vov_ta("i;j");
            buffer_type rhs(rhs_ta);

            vov.inplace_subtract("i;j", "i;j", rhs);
            vov_ta("i;j") -= rhs_ta("i;j");
            REQUIRE(vov.are_equal(buffer_type(vov_ta)));
        }

        SECTION("vector-of-matrices") {
            rhs_ta("i;j,k") = 2.0 * vom_ta("i;j,k");
            buffer_type rhs(rhs_ta);

            vom.inplace_subtract("i;j,k", "i;j,k", rhs);
            vom_ta("i;j,k") -= rhs_ta("i;j,k");
            REQUIRE(vom.are_equal(buffer_type(vom_ta)));
        }

        SECTION("matrix-of-vectors") {
            rhs_ta("i,j;k") = 2.0 * mov_ta("i,j;k");
            buffer_type rhs(rhs_ta);

            mov.inplace_subtract("i,j;k", "i,j;k", rhs);
            mov_ta("i,j;k") -= rhs_ta("i,j;k");
            REQUIRE(mov.are_equal(buffer_type(mov_ta)));
        }
    }

    SECTION("times") {
        tensor_type out_ta, rhs_ta;

        SECTION("vector-of-vectors") {
            rhs_ta("i;j") = 2.0 * vov_ta("i;j");
            buffer_type out, rhs(rhs_ta);

            vov.times("i;j", "i;j", out, "i;j", rhs);
            out_ta("i;j") = vov_ta("i;j") * rhs_ta("i;j");
            REQUIRE(out.are_equal(buffer_type(out_ta)));
        }

        SECTION("vector-of-matrices") {
            rhs_ta("i;j,k") = 2.0 * vom_ta("i;j,k");
            buffer_type out, rhs(rhs_ta);

            vom.times("i;j,k", "i;j,k", out, "i;j,k", rhs);
            out_ta("i;j,k") = vom_ta("i;j,k") * rhs_ta("i;j,k");
            REQUIRE(out.are_equal(buffer_type(out_ta)));
        }

        SECTION("matrix-of-vectors") {
            rhs_ta("i,j;k") = 2.0 * mov_ta("i,j;k");
            buffer_type out, rhs(rhs_ta);

            mov.times("i,j;k", "i,j;k", out, "i,j;k", rhs);
            out_ta("i,j;k") = mov_ta("i,j;k") * rhs_ta("i,j;k");
            REQUIRE(out.are_equal(buffer_type(out_ta)));
        }
    }

    SECTION("operator std::string") {
        std::string corr = "0: [ [0], [3] ) {\n"
                           "  [0]:[ [0], [3] ) { 1 2 3 }\n"
                           "  [1]:[ [0], [3] ) { 1 2 3 }\n"
                           "  [2]:[ [0], [3] ) { 1 2 3 }\n"
                           "}\n";
        REQUIRE(corr == std::string(vov));
    }

    SECTION("operator<<") {
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

    SECTION("hash") {
        using tensorwrapper::detail_::hash_objects;
        auto lhs = hash_objects(vov);

        SECTION("Are same") {
            REQUIRE(lhs == hash_objects(buffer_type(vov_ta)));
        }
        SECTION("Different") { REQUIRE(lhs != hash_objects(mov)); }
    }

    SECTION("are_equal") {
        SECTION("Are same") {
            buffer_type other_vov(vov_ta);
            REQUIRE(vov.are_equal(other_vov));
        }
        SECTION("Different") { REQUIRE_FALSE(vov.are_equal(mov)); }
    }
}
