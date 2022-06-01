#include <catch2/catch.hpp>
//#include "../../test_tensor.hpp"
#include "../../buffer/make_pimpl.hpp"
#include "tensorwrapper/ta_helpers/slice.hpp"
#include "tensorwrapper/ta_helpers/ta_helpers.hpp"
#include "tensorwrapper/tensor/novel/detail_/pimpl.hpp"

namespace ta_helpers = tensorwrapper::ta_helpers;
using namespace tensorwrapper::tensor;
using namespace tensorwrapper::tensor::novel;

/* Testing Strategy:
 *
 * We assume that all allocators and shapes work correctly. this means that
 * functions which depend on the shape and allocator state should work correctly
 * as long as those functions properly call and process the results of
 * interacting with allocators/shapes. In practice there are a lot of
 */

TEST_CASE("novel::TensorWrapperPIMPL<Scalar>") {
    using field_type     = field::Scalar;
    using pimpl_type     = novel::detail_::TensorWrapperPIMPL<field_type>;
    using buffer_type    = typename pimpl_type::buffer_type;
    using buffer_pointer = typename pimpl_type::buffer_pointer;
    using variant_type   = typename pimpl_type::variant_type;
    using ta_tensor_type = std::variant_alternative_t<0, variant_type>;
    using shape_type     = typename pimpl_type::shape_type;
    using extents_type   = typename pimpl_type::extents_type;
    using ta_trange_type = TA::TiledRange;

    using allocator::ta::Distribution;
    using allocator::ta::Storage;
    using allocator::ta::Tiling;

    auto palloc = novel::default_allocator<field_type>();
    auto oalloc = novel::allocator::ta_allocator<field_type>(
      Storage::Core, Tiling::SingleElementTile, Distribution::Distributed);

    buffer_pointer vec_buffer_obt, mat_buffer_obt, t3d_buffer_obt;
    buffer_pointer vec_buffer_set, mat_buffer_set, t3d_buffer_set;
    {
        auto [pv, pm, pt] = testing::make_pimpl<field_type>();
        vec_buffer_obt    = std::make_unique<buffer_type>(pv->clone());
        mat_buffer_obt    = std::make_unique<buffer_type>(pm->clone());
        t3d_buffer_obt    = std::make_unique<buffer_type>(pt->clone());

        ta_trange_type se_tr_vec{{0, 1, 2, 3}};
        ta_trange_type se_tr_mat{{0, 1, 2}, {0, 1, 2}};
        ta_trange_type se_tr_ten{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}};
        pv->retile(se_tr_vec);
        pm->retile(se_tr_mat);
        pt->retile(se_tr_ten);
        vec_buffer_set = std::make_unique<buffer_type>(std::move(pv));
        mat_buffer_set = std::make_unique<buffer_type>(std::move(pm));
        t3d_buffer_set = std::make_unique<buffer_type>(std::move(pt));
    }

    auto v_shape = std::make_unique<shape_type>(extents_type{3});
    auto m_shape = std::make_unique<shape_type>(extents_type{2, 2});
    auto t_shape = std::make_unique<shape_type>(extents_type{2, 2, 2});

    auto from_buffer = [](auto&& b) {
        return std::make_unique<buffer_type>(b->pimpl()->clone());
    };

    pimpl_type v(from_buffer(vec_buffer_obt), v_shape->clone(),
                 palloc->clone());
    pimpl_type m(from_buffer(mat_buffer_obt), m_shape->clone(),
                 palloc->clone());
    pimpl_type t(from_buffer(t3d_buffer_obt), t_shape->clone(),
                 palloc->clone());

    pimpl_type v2(from_buffer(vec_buffer_set), v_shape->clone(),
                  oalloc->clone());
    pimpl_type m2(from_buffer(mat_buffer_set), m_shape->clone(),
                  oalloc->clone());
    pimpl_type t2(from_buffer(t3d_buffer_set), t_shape->clone(),
                  oalloc->clone());

    SECTION("CTors") {
        SECTION("From Components") {
            REQUIRE(v.allocator() == *palloc);
            REQUIRE(v.shape() == *v_shape);
            REQUIRE(v.buffer() == *vec_buffer_obt);
            REQUIRE(v.size() == 3);

            REQUIRE(m.allocator() == *palloc);
            REQUIRE(m.shape() == *m_shape);
            REQUIRE(m.buffer() == *mat_buffer_obt);
            REQUIRE(m.size() == 4);

            REQUIRE(t.allocator() == *palloc);
            REQUIRE(t.shape() == *t_shape);
            REQUIRE(t.buffer() == *t3d_buffer_obt);
            REQUIRE(t.size() == 8);

            REQUIRE(v2.allocator() == *oalloc);
            REQUIRE(v2.shape() == *v_shape);
            REQUIRE(v2.buffer() == *vec_buffer_set);
            REQUIRE(v2.size() == 3);

            REQUIRE(m2.allocator() == *oalloc);
            REQUIRE(m2.shape() == *m_shape);
            REQUIRE(m2.buffer() == *mat_buffer_set);
            REQUIRE(m2.size() == 4);

            REQUIRE(t2.allocator() == *oalloc);
            REQUIRE(t2.shape() == *t_shape);
            REQUIRE(t2.buffer() == *t3d_buffer_set);
            REQUIRE(t2.size() == 8);
        }

        SECTION("clone") {
            auto v_copy = v.clone();
            REQUIRE(*v_copy == v);
            // Make sure we didn't just alias
            REQUIRE(&v_copy->allocator() != &v.allocator());
            REQUIRE(&v_copy->shape() != &v.shape());

            REQUIRE(*(m.clone()) == m);
            REQUIRE(*(t.clone()) == t);
        }
    }

    SECTION("make_annotation") {
        REQUIRE(v.make_annotation("i") == "i0");
        REQUIRE(m.make_annotation("j") == "j0,j1");
        REQUIRE(t.make_annotation("jk") == "jk0,jk1,jk2");
    }

    SECTION("rank") {
        REQUIRE(v.rank() == 1);
        REQUIRE(m.rank() == 2);
        REQUIRE(t.rank() == 3);
    }

    SECTION("norm()") {
        REQUIRE(v.norm() == Approx(3.74165738).margin(1E-8));
        REQUIRE(m.norm() == Approx(5.47722557).margin(1E-8));
        REQUIRE(t.norm() == Approx(14.2828568).margin(1E-8));
    }

    SECTION("sum()") {
        REQUIRE(v.sum() == 6);
        REQUIRE(m.sum() == 10);
        REQUIRE(t.sum() == 36);
    }

    SECTION("trace()") {
        REQUIRE_THROWS_AS(v.trace(), std::runtime_error);
        REQUIRE_THROWS_AS(t.trace(), std::runtime_error);
        REQUIRE(m.trace() == 5);
    }

    SECTION("print") {
        std::stringstream ss;

        SECTION("vector") {
            auto pss = &(v.print(ss));

            // Returns ss for chaining
            REQUIRE(pss == &ss);

            std::string corr = "0: [ [0], [3] ) { 1 2 3 }\n";
            REQUIRE(corr == ss.str());
        }

        SECTION("matrix") {
            auto pss = &(m.print(ss));

            // Returns ss for chaining
            REQUIRE(pss == &ss);

            std::string corr = "0: [ [0,0], [2,2] ) { 1 2 3 4 }\n";
            REQUIRE(corr == ss.str());
        }

        SECTION("tensor") {
            auto pss = &(t.print(ss));

            // Returns ss for chaining
            REQUIRE(pss == &ss);

            std::string corr = "0: [ [0,0,0], [2,2,2] ) { 1 2 3 4 5 6 7 8 }\n";
            REQUIRE(corr == ss.str());
        }
    }

    SECTION("reallocate") {
        SECTION("vector") {
            auto v_copy = v.clone();
            v_copy->reallocate(oalloc->clone());

            REQUIRE(v_copy->allocator() == *oalloc);
            REQUIRE(v_copy->buffer() == v2.buffer());
            REQUIRE(v_copy->buffer() != v.buffer());
        }

        SECTION("matrix") {
            auto m_copy = m.clone();
            m_copy->reallocate(oalloc->clone());

            REQUIRE(m_copy->allocator() == *oalloc);
            REQUIRE(m_copy->buffer() == m2.buffer());
            REQUIRE(m_copy->buffer() != m.buffer());
        }

        SECTION("tensor") {
            auto t_copy = t.clone();
            t_copy->reallocate(oalloc->clone());

            REQUIRE(t_copy->allocator() == *oalloc);
            REQUIRE(t_copy->buffer() == t2.buffer());
            REQUIRE(t_copy->buffer() != t.buffer());
        }
    }

    SECTION("hash") {
        using tensorwrapper::detail_::hash_objects;

        auto lhs = hash_objects(m);

        SECTION("Same") {
            pimpl_type rhs(from_buffer(mat_buffer_obt), m_shape->clone(),
                           palloc->clone());
            REQUIRE(lhs == hash_objects(rhs));
        }

        SECTION("Different values") {
            auto rhs_buffer = from_buffer(mat_buffer_obt);
            mat_buffer_obt->scale("i,j", "i,j", *rhs_buffer, 4.2);

            pimpl_type rhs(from_buffer(rhs_buffer), m_shape->clone(),
                           palloc->clone());
            REQUIRE(lhs != hash_objects(rhs));
        }

        SECTION("Different shape") {
            using sparse_shape    = SparseShape<field_type>;
            using sparse_map_type = typename sparse_shape::sparse_map_type;
            using index_type      = typename sparse_map_type::key_type;

            index_type i0{0}, i1{1};
            sparse_map_type sm{{i0, {i0, i1}}, {i1, {i0, i1}}};
            auto new_shape =
              std::make_unique<sparse_shape>(extents_type{2, 2}, sm);

            pimpl_type rhs(from_buffer(mat_buffer_obt), new_shape->clone(),
                           palloc->clone());
            REQUIRE(lhs != hash_objects(rhs));
        }
    }

    SECTION("operator==") {
        SECTION("Same") {
            pimpl_type rhs(from_buffer(mat_buffer_obt), m_shape->clone(),
                           palloc->clone());
            REQUIRE(m == rhs);
        }

        SECTION("Different values") {
            auto rhs_buffer = from_buffer(mat_buffer_obt);
            mat_buffer_obt->scale("i,j", "i,j", *rhs_buffer, 4.2);

            pimpl_type rhs(from_buffer(rhs_buffer), m_shape->clone(),
                           palloc->clone());
            REQUIRE_FALSE(m == rhs);
        }

        SECTION("Different Allocator") { REQUIRE_FALSE(m == m2); }

        SECTION("Different shape") {
            using sparse_shape    = SparseShape<field_type>;
            using sparse_map_type = typename sparse_shape::sparse_map_type;
            using index_type      = typename sparse_map_type::key_type;

            index_type i0{0}, i1{1};
            sparse_map_type sm{{i0, {i0, i1}}, {i1, {i0, i1}}};
            auto new_shape =
              std::make_unique<sparse_shape>(extents_type{2, 2}, sm);

            pimpl_type rhs(from_buffer(mat_buffer_obt), new_shape->clone(),
                           palloc->clone());
            REQUIRE(m.buffer() == rhs.buffer()); // Sanity check
            REQUIRE_FALSE(m == rhs);
        }
    }

    SECTION("slice()") {
        SECTION("Vector") {
            auto slice_corr = std::visit(
              [](auto&& arg) { return ta_helpers::slice(arg, {0}, {2}); },
              v.variant());
            auto slice = v.slice({0ul}, {2ul}, palloc->clone());
            REQUIRE(std::get<0>(slice->variant()) == slice_corr);
            REQUIRE(slice->shape() == *v_shape->slice({0}, {2}));
            REQUIRE(slice->allocator() == *palloc);
        }

        SECTION("Matrix") {
            auto slice_corr = std::visit(
              [](auto&& arg) {
                  return ta_helpers::slice(arg, {0, 1}, {1, 2});
              },
              m.variant());
            auto slice = m.slice({0ul, 1ul}, {1ul, 2ul}, palloc->clone());
            REQUIRE(std::get<0>(slice->variant()) == slice_corr);
            REQUIRE(slice->shape() == *m_shape->slice({0, 1}, {1, 2}));
            REQUIRE(slice->allocator() == *palloc);
        }

        SECTION("Tensor") {
            auto slice_corr = std::visit(
              [](auto&& arg) {
                  return ta_helpers::slice(arg, {0, 0, 1}, {2, 2, 2});
              },
              t.variant());
            auto slice =
              t.slice({0ul, 0ul, 1ul}, {2ul, 2ul, 2ul}, palloc->clone());
            REQUIRE(std::get<0>(slice->variant()) == slice_corr);
            REQUIRE(slice->shape() == *t_shape->slice({0, 0, 1}, {2, 2, 2}));
            REQUIRE(slice->allocator() == *palloc);
        }
        // SECTION("Different allocator") {
        // TODO
        //}
    }

    SECTION("reshape") {
#if 0
        SECTION("Literal reshape") {
            SECTION("vector") {
                extents_type four{3, 1};
                auto new_shape = std::make_unique<shape_type>(four);
                auto tr        = alloc->make_tiled_range(four);
                ta_tensor_type corr(alloc->runtime(), tr, {{1}, {2}, {3}});

                pimpl_type m3(vector, new_shape->clone(), alloc->clone());
                REQUIRE(m3.allocator() == *alloc);
                REQUIRE(m3.shape() == *new_shape);
                REQUIRE(std::get<0>(m3.variant()) == corr);
                REQUIRE(m3.size() == 3);
            }

            SECTION("matrix") {
                extents_type four{4};
                auto new_shape = std::make_unique<shape_type>(four);
                auto tr        = alloc->make_tiled_range(four);
                ta_tensor_type corr(alloc->runtime(), tr, {1, 2, 3, 4});

                pimpl_type m3(matrix, new_shape->clone(), alloc->clone());
                REQUIRE(m3.allocator() == *alloc);
                REQUIRE(m3.shape() == *new_shape);
                REQUIRE(std::get<0>(m3.variant()) == corr);
                REQUIRE(m3.size() == 4);
            }

            SECTION("tensor") {
                extents_type four{8};
                auto new_shape = std::make_unique<shape_type>(four);
                auto tr        = alloc->make_tiled_range(four);
                ta_tensor_type corr(alloc->runtime(), tr,
                                    {1, 2, 3, 4, 5, 6, 7, 8});

                pimpl_type m3(tensor, new_shape->clone(), alloc->clone());
                REQUIRE(m3.allocator() == *alloc);
                REQUIRE(m3.shape() == *new_shape);
                REQUIRE(std::get<0>(m3.variant()) == corr);
                REQUIRE(m3.size() == 8);
            }
        }
#endif

        SECTION("Apply sparsity") {
            using sparse_shape    = SparseShape<field_type>;
            using sparse_map_type = typename sparse_shape::sparse_map_type;
            using index_type      = typename sparse_map_type::key_type;

            auto new_alloc = oalloc->clone();
            index_type i0{0}, i00{0, 0}, i1{1}, i11{1, 1}, i10{1, 0};

            // Can't apply to vector (need an independent and a dependent index)

            SECTION("matrix") {
		// [x 0]
		// [x 0]
                sparse_map_type sm{{i0, {i0}}, {i1, {i0}}};
                auto new_shape = std::make_unique<sparse_shape>(extents_type{2,2}, sm);

                auto m3 = m2.clone();
		m3->reshape(new_shape->clone());

                REQUIRE(m3->allocator() == *new_alloc);
                REQUIRE(m3->shape() == *new_shape);
                REQUIRE(m3->sum() == Approx(4.0));
                REQUIRE(m3->size() == 4);
            }

	    SECTION("tensor") {
                SECTION("Rank 1 ind, rank 2 dependent") {
                    sparse_map_type sm{{i0, {i00}}, {i1, {i00}}};
                    auto new_shape = std::make_unique<sparse_shape>(extents_type{2,2,2}, sm);

                    auto t3 = t2.clone();
		    t3->reshape(new_shape->clone());

                    REQUIRE(t3->allocator() == *new_alloc);
                    REQUIRE(t3->shape() == *new_shape);
                    REQUIRE(t3->sum() == Approx(6.0));
                    REQUIRE(t3->size() == 8);
		}
                SECTION("Rank 2 ind, rank 1 dependent") {
                    sparse_map_type sm{{i00, {i0}}, {i10, {i0}}};
                    auto new_shape = std::make_unique<sparse_shape>(extents_type{2,2,2}, sm);

                    auto t3 = t2.clone();
		    t3->reshape(new_shape->clone());

                    REQUIRE(t3->allocator() == *new_alloc);
                    REQUIRE(t3->shape() == *new_shape);
                    REQUIRE(t3->sum() == Approx(6.0));
                    REQUIRE(t3->size() == 8);
		}
	    }

        }
    }
#if 0
    SECTION("CTors") {
        SECTION("No Shape") {
            // Just checking that it's triggered, reallocate looks at more
            // edge-cases
            SECTION("Reallocates if necessary") {
                auto new_alloc = std::make_unique<other_alloc>();
                auto tr        = new_alloc->make_tiled_range(extents_type{3});
                ta_tensor_type corr(new_alloc->runtime(), tr, {1, 2, 3});

                pimpl_type v3(vector, new_alloc->clone());
                REQUIRE(v3.allocator() == *alloc);
                REQUIRE(std::get<0>(v3.variant()) == corr);
                REQUIRE(v3.size() == 3);
            }
        }

        SECTION("With shape") {
            // Just want to check that the following are triggered,
            // reshape/reallocate will check more in depth
            SECTION("Reshapes if necessary") {
                extents_type four{4};
                auto new_shape = std::make_unique<shape_type>(four);
                auto tr        = alloc->make_tiled_range(four);
                ta_tensor_type corr(alloc->runtime(), tr, {1, 2, 3, 4});

                pimpl_type m3(matrix, new_shape->clone(), alloc->clone());
                REQUIRE(m3.allocator() == *alloc);
                REQUIRE(m3.shape() == *new_shape);
                REQUIRE(std::get<0>(m3.variant()) == corr);
                REQUIRE(m3.size() == 4);
            }

            SECTION("Applies sparsity if needed") {
                using single_tiles    = SingleElementTiles<field_type>;
                using sparse_shape    = SparseShape<field_type>;
                using sparse_map_type = typename sparse_shape::sparse_map_type;
                using index_type      = typename sparse_map_type::key_type;

                auto new_alloc = std::make_unique<single_tiles>();
                extents_type two{2, 2};
                auto tr     = new_alloc->make_tiled_range(two);
                auto& world = new_alloc->runtime();
                variant_type input{ta_tensor_type(world, tr, {{1, 2}, {3, 4}})};
                ta_tensor_type corr(world, tr, {{1, 0}, {3, 0}});

                index_type i0{0}, i1{1};
                sparse_map_type sm{{i0, {i0}}, {i1, {i0}}};
                auto new_shape = std::make_unique<sparse_shape>(two, sm);

                pimpl_type m3(input, new_shape->clone(), new_alloc->clone());
                REQUIRE(m3.allocator() == *new_alloc);
                REQUIRE(m3.shape() == *new_shape);
                REQUIRE(std::get<0>(m3.variant()) == corr);
                REQUIRE(m3.size() == 4);
            }

            SECTION("Reallocates if necessary") {
                auto new_alloc = std::make_unique<other_alloc>();
                auto tr        = new_alloc->make_tiled_range(extents_type{3});
                ta_tensor_type corr(new_alloc->runtime(), tr, {1, 2, 3});

                pimpl_type v3(vector, v_shape->clone(), new_alloc->clone());
                REQUIRE(v3.allocator() == *alloc);
                REQUIRE(v3.shape() == *v_shape);
                REQUIRE(std::get<0>(v3.variant()) == corr);
                REQUIRE(v3.size() == 3);
            }
        }
    }

    SECTION("reshape") {
        SECTION("Literal reshape") {
            SECTION("vector") {
                extents_type four{3, 1};
                auto new_shape = std::make_unique<shape_type>(four);
                auto tr        = alloc->make_tiled_range(four);
                ta_tensor_type corr(alloc->runtime(), tr, {{1}, {2}, {3}});

                pimpl_type m3(vector, new_shape->clone(), alloc->clone());
                REQUIRE(m3.allocator() == *alloc);
                REQUIRE(m3.shape() == *new_shape);
                REQUIRE(std::get<0>(m3.variant()) == corr);
                REQUIRE(m3.size() == 3);
            }

            SECTION("matrix") {
                extents_type four{4};
                auto new_shape = std::make_unique<shape_type>(four);
                auto tr        = alloc->make_tiled_range(four);
                ta_tensor_type corr(alloc->runtime(), tr, {1, 2, 3, 4});

                pimpl_type m3(matrix, new_shape->clone(), alloc->clone());
                REQUIRE(m3.allocator() == *alloc);
                REQUIRE(m3.shape() == *new_shape);
                REQUIRE(std::get<0>(m3.variant()) == corr);
                REQUIRE(m3.size() == 4);
            }

            SECTION("tensor") {
                extents_type four{8};
                auto new_shape = std::make_unique<shape_type>(four);
                auto tr        = alloc->make_tiled_range(four);
                ta_tensor_type corr(alloc->runtime(), tr,
                                    {1, 2, 3, 4, 5, 6, 7, 8});

                pimpl_type m3(tensor, new_shape->clone(), alloc->clone());
                REQUIRE(m3.allocator() == *alloc);
                REQUIRE(m3.shape() == *new_shape);
                REQUIRE(std::get<0>(m3.variant()) == corr);
                REQUIRE(m3.size() == 8);
            }
        }
    }
#endif
}
