/*
 * Copyright 2022 NWChemEx-Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
    using lazy_type     = typename buffer_type::lazy_tensor_type;
    using trange_type   = typename buffer_type::ta_trange_type;
    using ta_shape_type = typename buffer_type::ta_shape_type;

    /// Direct array utilities
    using lazy_tile_type = tensorwrapper::ta_helpers::lazy_scalar_type;
    using range_type     = typename lazy_tile_type::range_type;
    using tile_type      = typename lazy_tile_type::eval_type;

    auto scalar_lambda = [](range_type range) -> tile_type {
        auto t = tile_type(range, 0.0);
        for(const auto& idx : range) {
            auto n_dims  = idx.size();
            double value = 1.0;
            for(auto i = 0; i < n_dims; ++i) {
                value += std::pow(2.0, i) * idx[n_dims - 1 - i];
            }
            t[idx] = value;
        }
        return t;
    };
    lazy_tile_type::add_evaluator(scalar_lambda, "ta_scalar_test");

    auto tile_lambda = [](lazy_tile_type& t, const range_type& r) -> float {
        t = lazy_tile_type(r, "ta_scalar_test");
        return 1.0;
    };

    auto& world = TA::get_default_world();
    tensor_type vec_ta(world, {1.0, 2.0, 3.0});
    tensor_type mat_ta(world, {{1.0, 2.0}, {3.0, 4.0}});
    tensor_type t3d_ta(world,
                       {{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}});

    trange_type trange1{{0, 3}}, trange2{{0, 2}, {0, 2}};
    auto dvec_ta =
      TiledArray::make_array<lazy_type>(world, trange1, tile_lambda);
    auto dmat_ta =
      TiledArray::make_array<lazy_type>(world, trange2, tile_lambda);

    buffer_type defaulted;
    buffer_type vec(vec_ta);
    buffer_type mat(mat_ta);
    buffer_type t3d(t3d_ta);
    buffer_type dvec(dvec_ta);
    buffer_type dmat(dmat_ta);
    buffer_type dout(lazy_type{});

    SECTION("default_clone()") {
        REQUIRE(vec.default_clone()->are_equal(defaulted));
    }

    SECTION("clone()") {
        auto vec2 = vec.clone();
        REQUIRE(vec2->are_equal(vec));

        auto mat2 = mat.clone();
        REQUIRE(mat2->are_equal(mat));

        auto t3d2 = t3d.clone();
        REQUIRE(t3d2->are_equal(t3d));

        auto dvec2 = dvec.clone();
        REQUIRE(dvec2->are_equal(dvec));

        auto dmat2 = dmat.clone();
        REQUIRE(dmat2->are_equal(dmat));
    }

    SECTION("retile") {
        // These may need to use allclose
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
        SECTION("direct") {
            using error_t = std::runtime_error;
            REQUIRE_THROWS_AS(dvec.retile(trange_type{}), error_t);
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
        SECTION("direct") {
            using error_t = std::runtime_error;
            REQUIRE_THROWS_AS(dvec.set_shape(ta_shape_type{}), error_t);
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

        SECTION("direct") {
            dvec.scale("i", "i", out, 2.0);
            out_ta("i") = 2.0 * vec_ta("i");
            REQUIRE(out.are_equal(buffer_type(out_ta)));
        }

        SECTION("throws if trying to assign to direct") {
            using error_t = std::runtime_error;
            REQUIRE_THROWS_AS(dvec.scale("i", "i", dout, 2.0), error_t);
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

        SECTION("direct") {
            rhs_ta("i") = 2.0 * vec_ta("i");
            buffer_type out, rhs(rhs_ta);

            dvec.add("i", "i", out, "i", rhs);
            out_ta("i") = vec_ta("i") + rhs_ta("i");
            REQUIRE(out.are_equal(buffer_type(out_ta)));
        }

        SECTION("throws if trying to assign to direct") {
            using error_t = std::runtime_error;
            REQUIRE_THROWS_AS(dvec.add("i", "i", dout, "i", vec), error_t);
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

        SECTION("throws if trying to assign to direct") {
            using error_t = std::runtime_error;
            REQUIRE_THROWS_AS(dvec.inplace_add("i", "i", vec), error_t);
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

        SECTION("direct") {
            rhs_ta("i") = 2.0 * vec_ta("i");
            buffer_type out, rhs(rhs_ta);

            dvec.subtract("i", "i", out, "i", rhs);
            out_ta("i") = vec_ta("i") - rhs_ta("i");
            REQUIRE(out.are_equal(buffer_type(out_ta)));
        }

        SECTION("throws if trying to assign to direct") {
            using error_t = std::runtime_error;
            REQUIRE_THROWS_AS(dvec.subtract("i", "i", dout, "i", vec), error_t);
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

        SECTION("throws if trying to assign to direct") {
            using error_t = std::runtime_error;
            REQUIRE_THROWS_AS(dvec.inplace_subtract("i", "i", vec), error_t);
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

        SECTION("direct") {
            rhs_ta("i,j") = 2.0 * mat_ta("i,j");
            buffer_type out, rhs(rhs_ta);

            dmat.times("i,j", "i,k", out, "j,k", rhs);
            out_ta("i,k") = mat_ta("i,j") * rhs_ta("j,k");
            REQUIRE(out.are_equal(buffer_type(out_ta)));
        }

        SECTION("throws if trying to assign to direct") {
            using error_t = std::runtime_error;
            REQUIRE_THROWS_AS(dvec.times("i", "i", dout, "i", vec), error_t);
        }
    }

    SECTION("norm") {
        SECTION("vector") {
            auto ref_norm = vec_ta("i").norm().get();
            auto norm     = vec.norm();
            REQUIRE(norm == ref_norm);
        }

        SECTION("matrix") {
            auto ref_norm = mat_ta("i,j").norm().get();
            auto norm     = mat.norm();
            REQUIRE(norm == ref_norm);
        }

        SECTION("tensor") {
            auto ref_norm = t3d_ta("i,j,k").norm().get();
            auto norm     = t3d.norm();
            REQUIRE(norm == ref_norm);
        }

        SECTION("direct") {
            auto ref_norm = vec_ta("i").norm().get();
            auto norm     = dvec.norm();
            REQUIRE(norm == ref_norm);
        }
    }

    SECTION("sum") {
        SECTION("vector") {
            auto ref_sum = vec_ta("i").sum().get();
            auto sum     = vec.sum();
            REQUIRE(sum == ref_sum);
        }

        SECTION("matrix") {
            auto ref_sum = mat_ta("i,j").sum().get();
            auto sum     = mat.sum();
            REQUIRE(sum == ref_sum);
        }

        SECTION("tensor") {
            auto ref_sum = t3d_ta("i,j,k").sum().get();
            auto sum     = t3d.sum();
            REQUIRE(sum == ref_sum);
        }

        SECTION("direct") {
            auto ref_sum = vec_ta("i").sum().get();
            auto sum     = dvec.sum();
            REQUIRE(sum == ref_sum);
        }
    }

    SECTION("trace") {
        SECTION("invalid") {
            REQUIRE_THROWS_AS(vec.trace(), std::runtime_error);
            REQUIRE_THROWS_AS(t3d.trace(), std::runtime_error);
        }

        SECTION("matrix") {
            auto ref_trace = mat_ta("i,j").trace().get();
            auto trace     = mat.trace();
            REQUIRE(trace == ref_trace);
        }

        SECTION("direct") {
            auto ref_trace = mat_ta("i,j").trace().get();
            auto trace     = dmat.trace();
            REQUIRE(trace == ref_trace);
        }
    }

    SECTION("make_extents") {
        REQUIRE(defaulted.make_extents() == std::vector<std::size_t>{});
        REQUIRE(vec.make_extents() == std::vector<std::size_t>{3});
        REQUIRE(mat.make_extents() == std::vector<std::size_t>{2, 2});
        REQUIRE(t3d.make_extents() == std::vector<std::size_t>{2, 2, 2});
        REQUIRE(dvec.make_extents() == std::vector<std::size_t>{3});
        REQUIRE(dmat.make_extents() == std::vector<std::size_t>{2, 2});
    }

    SECTION("make_inner_extents") {
        REQUIRE(defaulted.make_inner_extents() == 1);
        REQUIRE(vec.make_inner_extents() == 1);
        REQUIRE(mat.make_inner_extents() == 1);
        REQUIRE(t3d.make_inner_extents() == 1);
        REQUIRE(dvec.make_inner_extents() == 1);
        REQUIRE(dmat.make_inner_extents() == 1);
    }

    SECTION("operator std::string") {
        SECTION("data") {
            std::string corr = "0: [ [0], [3] ) { 1 2 3 }\n";
            REQUIRE(corr == std::string(vec));
        }
        SECTION("direct") {
            std::string corr = "0: [ [0], [3] )\n";
            REQUIRE(corr == std::string(dvec));
        }
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

    SECTION("are_equal") {
        SECTION("Are same") {
            buffer_type other_vec(vec_ta);
            buffer_type other_dvec(dvec_ta);
            REQUIRE(vec.are_equal(other_vec));
            REQUIRE(dvec.are_equal(other_dvec));
        }
        SECTION("Different") {
            REQUIRE_FALSE(vec.are_equal(mat));
            REQUIRE_FALSE(vec.are_equal(dvec));
        }
    }
}
