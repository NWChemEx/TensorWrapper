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

#include "test_tensor.hpp"

using namespace tensorwrapper::tensor;

/* Testing Strategy:
 *
 * The actual TensorWrapper class is a pass through to the PIMPL in many
 * circumstances. For these unit tests we assume that the PIMPLs work and are
 * thoroughly tested. Thus for functions which are just pass throughs, we simply
 * need to ensure that arguments and returns are forwarded correctly.
 */

TEST_CASE("TensorWrapper<Scalar>") {
    using field_type   = field::Scalar;
    using TWrapper     = TensorWrapper<field_type>;
    using shape_type   = typename TWrapper::shape_type;
    using extents_type = typename TWrapper::extents_type;

    using allocator::ta::Distribution;
    using allocator::ta::Storage;

    auto default_alloc = default_allocator<field_type>();
    auto other_alloc   = allocator::ta_allocator<field_type>(
      Storage::Core, Distribution::Distributed);

    auto ref_tensors = testing::get_tensors<field_type>();
    auto& vec        = ref_tensors["vector"];
    auto& mat        = ref_tensors["matrix"];
    auto& t3d        = ref_tensors["tensor"];
    TWrapper defaulted;

    auto vec_shape = std::make_unique<shape_type>(extents_type{3});

    SECTION("CTors") {
        SECTION("Default") {
            REQUIRE(defaulted.rank() == 0);
            REQUIRE(defaulted.extents() == extents_type{});
            REQUIRE(defaulted.size() == 0);
        }
        SECTION("From Tile Lambda") {
            auto l = [](const auto& lo, const auto& up, auto* data) {
                for(auto i = lo[0]; i < up[0]; ++i) data[i] = i + 1;
            };
            TWrapper tw(l, vec_shape->clone(), default_alloc->clone());
            REQUIRE(tw == vec);
        }
        SECTION("From Element Lambda") {
            auto l = [](const auto& idx) { return idx[0] + 1; };
            TWrapper tw(l, vec_shape->clone(), default_alloc->clone());
            REQUIRE(tw == vec);
        }
        SECTION("Copy") {
            TWrapper copied(vec);
            REQUIRE(copied.rank() == 1);
            REQUIRE(copied.extents() == extents_type{3});
            REQUIRE(copied.allocator().is_equal(vec.allocator()));
        }
        SECTION("Move") {
            const auto* pa = &(vec.allocator());
            TWrapper moved(std::move(vec));
            REQUIRE(moved.rank() == 1);
            REQUIRE(moved.extents() == extents_type{3});
            REQUIRE(&moved.allocator() == pa);
        }
        SECTION("Copy assignment") {
            TWrapper copied;
            auto pcopied = &(copied = vec);
            REQUIRE(pcopied == &copied);
            REQUIRE(copied.rank() == 1);
            REQUIRE(copied.extents() == extents_type{3});
            REQUIRE(copied.allocator().is_equal(vec.allocator()));
        }

        SECTION("Move assignment") {
            TWrapper moved;
            const auto* pa = &(vec.allocator());
            auto pmoved    = &(moved = std::move(vec));
            REQUIRE(pmoved == &moved);
            REQUIRE(moved.rank() == 1);
            REQUIRE(moved.extents() == extents_type{3});
            REQUIRE(&moved.allocator() == pa);
        }

        SECTION("Initializer Lists") {
            TWrapper vec_from_il({1.0, 2.0, 3.0});
            TWrapper mat_from_il({{1.0, 2.0}, {3.0, 4.0}});
            TWrapper t3d_from_il(
              {{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}});

            REQUIRE(vec_from_il == vec);
            REQUIRE(mat_from_il == mat);
            REQUIRE(t3d_from_il == t3d);
        }
    }

    SECTION("reallocate") {
        auto new_p     = other_alloc->clone();
        const auto* pa = new_p.get();

        SECTION("Non-default") {
            auto v_copy = vec.pimpl().clone();
            v_copy->reallocate(new_p->clone());
            TWrapper corr(std::move(v_copy));
            vec.reallocate(std::move(new_p));
            REQUIRE(vec == corr);
            REQUIRE(&vec.allocator() == pa);
        }
    }

    SECTION("slice()") {
        auto v_slice = vec.pimpl().slice({0ul}, {2ul}, default_alloc->clone());
        TWrapper corr(std::move(v_slice));
        auto tw_slice = vec.slice({0ul}, {2ul}, default_alloc->clone());
        REQUIRE(tw_slice == corr);
    }

    SECTION("reshape()") {
        SECTION("Incorrect shape") {
            auto p = std::make_unique<shape_type>(extents_type{2, 3});
            REQUIRE_THROWS_AS(vec.reshape(std::move(p)), std::runtime_error);
        }
        SECTION("Vector to matrix") {
            auto p     = std::make_unique<shape_type>(extents_type{1, 3});
            auto v_cpy = vec.pimpl().clone();
            v_cpy->reshape(p->clone());
            TWrapper corr(std::move(v_cpy));
            auto new_v = vec.reshape(p->clone());
            REQUIRE(new_v == corr);
        }
    }

    SECTION("allocator") {
        REQUIRE_THROWS_AS(defaulted.allocator(), std::runtime_error);
        REQUIRE(vec.allocator().is_equal(*default_alloc));
        REQUIRE(mat.allocator().is_equal(*default_alloc));
        REQUIRE(t3d.allocator().is_equal(*default_alloc));
    }

    SECTION("make_annotation") {
        REQUIRE(defaulted.make_annotation() == "");
        REQUIRE(vec.make_annotation() == "i0");
        REQUIRE(mat.make_annotation("j") == "j0,j1");
        REQUIRE(t3d.make_annotation() == "i0,i1,i2");
    }

    SECTION("rank()") {
        REQUIRE(defaulted.rank() == 0);
        REQUIRE(vec.rank() == 1);
        REQUIRE(mat.rank() == 2);
        REQUIRE(t3d.rank() == 3);
    }

    SECTION("extents()") {
        REQUIRE(defaulted.extents() == extents_type{});
        REQUIRE(vec.extents() == extents_type{3});
        REQUIRE(mat.extents() == extents_type{2, 2});
        REQUIRE(t3d.extents() == extents_type{2, 2, 2});
    }

    SECTION("shape()") {
        REQUIRE(vec.shape() == shape_type(vec.extents()));
        REQUIRE(mat.shape() == shape_type(mat.extents()));
        REQUIRE(t3d.shape() == shape_type(t3d.extents()));
    }

    SECTION("norm()") {
        REQUIRE_THROWS_AS(defaulted.norm(), std::runtime_error);
        REQUIRE(vec.norm() == Approx(3.74165738).margin(1E-8));
        REQUIRE(mat.norm() == Approx(5.47722557).margin(1E-8));
        REQUIRE(t3d.norm() == Approx(14.2828568).margin(1E-8));
    }

    SECTION("sum()") {
        REQUIRE_THROWS_AS(defaulted.sum(), std::runtime_error);
        REQUIRE(vec.sum() == 6);
        REQUIRE(mat.sum() == 10);
        REQUIRE(t3d.sum() == 36);
    }

    SECTION("trace()") {
        REQUIRE_THROWS_AS(defaulted.trace(), std::runtime_error);
        REQUIRE_THROWS_AS(vec.trace(), std::runtime_error);
        REQUIRE_THROWS_AS(t3d.trace(), std::runtime_error);
        REQUIRE(mat.trace() == 5);
    }

    SECTION("operator()") {
        // Basically just testing that it compiles, real test happens in
        // labeled_tensor_wrapper.cpp
        REQUIRE_NOTHROW(vec("i"));
        REQUIRE_NOTHROW(mat("i,j"));
        REQUIRE_NOTHROW(t3d("i,j,k"));
    }

    SECTION("operator() const") {
        // Basically just testing that it compiles, real test happens in
        // labeled_tensor_wrapper.cpp
        REQUIRE_NOTHROW(std::as_const(vec)("i"));
        REQUIRE_NOTHROW(std::as_const(mat)("i,j"));
        REQUIRE_NOTHROW(std::as_const(t3d)("i,j,k"));
    }
}
