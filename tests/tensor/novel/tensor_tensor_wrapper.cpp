#include "test_tensor.hpp"

using namespace tensorwrapper::tensor;
using namespace tensorwrapper::tensor::novel;

/* Testing Strategy:
 *
 * The actual TensorWrapper class is a pass through to the PIMPL in many
 * circumstances. For these unit tests we assume that the PIMPLs work and are
 * thoroughly tested. Thus for functions which are just pass throughs, we simply
 * need to ensure that arguments and returns are forwarded correctly.
 */

TEST_CASE("TensorWrapper<Tensor>") {
    using field_type   = field::Tensor;
    using TWrapper     = novel::TensorWrapper<field_type>;
    using shape_type   = typename TWrapper::shape_type;
    using extents_type = typename TWrapper::extents_type;

    using allocator::ta::Distribution;
    using allocator::ta::Storage;
    using allocator::ta::Tiling;

    auto default_alloc = default_allocator<field_type>();
    auto other_alloc   = novel::allocator::ta_allocator<field_type>(
      Storage::Core, Tiling::SingleElementTile, Distribution::Distributed);

    auto ref_tensors = testing::get_tensors<field_type>();
    auto& vov        = ref_tensors["vector-of-vectors"];
    auto& vom        = ref_tensors["vector-of-matrices"];
    auto& mov        = ref_tensors["matrix-of-vectors"];
    TWrapper defaulted;

    extents_type vector_extents{3}, matrix_extents{2,2};
    auto vov_shape =
      testing::make_uniform_tot_shape(vector_extents, vector_extents);
    auto vom_shape =
      testing::make_uniform_tot_shape(vector_extents, matrix_extents);
    auto mov_shape =
      testing::make_uniform_tot_shape(matrix_extents, vector_extents);


    SECTION("CTors") {
        SECTION("Default") {
            REQUIRE(defaulted.rank() == 0);
            REQUIRE(defaulted.extents() == extents_type{});
            REQUIRE(defaulted.size() == 0);
        }
        SECTION("From Tile Lambda") {
            auto l = [](auto outer, auto inner) -> double {
              return inner[0] + 1;
            };
            TWrapper tw(l, vov_shape.clone(), default_alloc->clone());
            REQUIRE(tw == vov);
        }
        SECTION("Copy") {
            TWrapper copied(vom);
            REQUIRE(copied.rank() == 3);
            REQUIRE(copied.extents() == vector_extents);
            REQUIRE(copied.shape().inner_extents() == vom_shape.inner_extents());
            REQUIRE(copied.allocator().is_equal(vom.allocator()));
        }
        SECTION("Move") {
            const auto* pa = &(vov.allocator());
            TWrapper moved(std::move(vov));
            REQUIRE(moved.rank() == 2);
            REQUIRE(moved.extents() == extents_type{3});
            REQUIRE(moved.shape().inner_extents() == vov_shape.inner_extents());
            REQUIRE(&moved.allocator() == pa);
        }
        SECTION("Copy assignment") {
            TWrapper copied;
            auto pcopied = &(copied = vov);
            REQUIRE(pcopied == &copied);
            REQUIRE(copied.rank() == 2);
            REQUIRE(copied.extents() == extents_type{3});
            REQUIRE(copied.shape().inner_extents() == vov_shape.inner_extents());
            REQUIRE(copied.allocator().is_equal(vov.allocator()));
        }
        SECTION("Move assignment") {
            TWrapper moved;
            const auto* pa = &(vov.allocator());
            auto pmoved    = &(moved = std::move(vov));
            REQUIRE(pmoved == &moved);
            REQUIRE(moved.rank() == 2);
            REQUIRE(moved.extents() == extents_type{3});
            REQUIRE(moved.shape().inner_extents() == vov_shape.inner_extents());
            REQUIRE(&moved.allocator() == pa);
        }
    }

#if 0
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
#endif

    SECTION("allocator") {
        REQUIRE_THROWS_AS(defaulted.allocator(), std::runtime_error);
        REQUIRE(vov.allocator().is_equal(*default_alloc));
        REQUIRE(vom.allocator().is_equal(*default_alloc));
        REQUIRE(mov.allocator().is_equal(*default_alloc));
    }

    SECTION("make_annotation") {
        REQUIRE(defaulted.make_annotation() == "");
        REQUIRE(vov.make_annotation("i") == "i0;i1");
        REQUIRE(mov.make_annotation("j") == "j0,j1;j2");
        REQUIRE(vom.make_annotation("jk") == "jk0;jk1,jk2");
    }

    SECTION("rank()") {
        REQUIRE(defaulted.rank() == 0);
        REQUIRE(vov.rank() == 2);
        REQUIRE(vom.rank() == 3);
        REQUIRE(mov.rank() == 3);
    }

    SECTION("extents()") {
        REQUIRE(defaulted.extents() == extents_type{});
        REQUIRE(vov.extents() == extents_type{3});
        REQUIRE(vom.extents() == extents_type{3});
        REQUIRE(mov.extents() == extents_type{2, 2});
    }

    SECTION("norm()") {
        REQUIRE(vov.norm() == Approx(6.4807406984).margin(1E-8));
        REQUIRE(mov.norm() == Approx(7.4833147735).margin(1E-8));
        REQUIRE(vom.norm() == Approx(9.4868329805).margin(1E-8));
    }

    SECTION("sum()") {
        REQUIRE(vov.sum() == 18);
        REQUIRE(mov.sum() == 24);
        REQUIRE(vom.sum() == 30);
    }

    SECTION("trace()") {
        REQUIRE_THROWS_AS(vov.trace(), std::runtime_error);
        REQUIRE_THROWS_AS(mov.trace(), std::runtime_error);
        REQUIRE_THROWS_AS(vom.trace(), std::runtime_error);
    }

    SECTION("print") {
        std::stringstream ss;

        SECTION("vector-of-vectors") {
            auto pss = &(vov.print(ss));

            // Returns ss for chaining
            REQUIRE(pss == &ss);

            std::string corr = "0: [ [0], [3] ) {\n"
                               "  [0]:[ [0], [3] ) { 1 2 3 }\n"
                               "  [1]:[ [0], [3] ) { 1 2 3 }\n"
                               "  [2]:[ [0], [3] ) { 1 2 3 }\n"
                               "}\n";
            REQUIRE(corr == ss.str());
        }

        SECTION("matrix-of-vectors") {
            auto pss = &(mov.print(ss));

            // Returns ss for chaining
            REQUIRE(pss == &ss);

            std::string corr = "0: [ [0,0], [2,2] ) {\n"
                               "  [0,0]:[ [0], [3] ) { 1 2 3 }\n"
                               "  [0,1]:[ [0], [3] ) { 1 2 3 }\n"
                               "  [1,0]:[ [0], [3] ) { 1 2 3 }\n"
                               "  [1,1]:[ [0], [3] ) { 1 2 3 }\n"
                               "}\n";
            REQUIRE(corr == ss.str());
        }

        SECTION("vector-of-matrices") {
            auto pss = &(vom.print(ss));

            // Returns ss for chaining
            REQUIRE(pss == &ss);

            std::string corr = "0: [ [0], [3] ) {\n"
                               "  [0]:[ [0,0], [2,2] ) { 1 2 3 4 }\n"
                               "  [1]:[ [0,0], [2,2] ) { 1 2 3 4 }\n"
                               "  [2]:[ [0,0], [2,2] ) { 1 2 3 4 }\n"
                               "}\n";
            REQUIRE(corr == ss.str());
        }
    }

    SECTION("reallocate") {
        using except_t = std::runtime_error;

        SECTION("vector-of-vectors") {
            REQUIRE_THROWS_AS(vov.reallocate(other_alloc->clone()), except_t);
        }

        SECTION("matrix-of-vectors") {
            REQUIRE_THROWS_AS(mov.reallocate(other_alloc->clone()), except_t);
        }

        SECTION("vector-of-matrices") {
            REQUIRE_THROWS_AS(vom.reallocate(other_alloc->clone()), except_t);
        }
    }

    SECTION("operator()") {
        // Basically just testing that it compiles, real test happens in
        // labeled_tensor_wrapper.cpp
        REQUIRE_NOTHROW(vov("i;j"));
        REQUIRE_NOTHROW(mov("i,j;k"));
        REQUIRE_NOTHROW(vom("i;j,k"));
    }

    SECTION("operator() const") {
        // Basically just testing that it compiles, real test happens in
        // labeled_tensor_wrapper.cpp
        REQUIRE_NOTHROW(std::as_const(vov)("i;j"));
        REQUIRE_NOTHROW(std::as_const(mov)("i,j;k"));
        REQUIRE_NOTHROW(std::as_const(vom)("i;j,k"));
    }


#if 0
    SECTION("shape()") {
        REQUIRE(vec.shape() == shape_type(vec.extents()));
        REQUIRE(mat.shape() == shape_type(mat.extents()));
        REQUIRE(t3d.shape() == shape_type(t3d.extents()));
    }
    SECTION("hash") {
        using tensorwrapper::detail_::hash_objects;
        TWrapper other_vec(vec);
        REQUIRE(hash_objects(other_vec) == hash_objects(vec));
        REQUIRE(hash_objects(vec) != hash_objects(mat));
    }
#endif
}
