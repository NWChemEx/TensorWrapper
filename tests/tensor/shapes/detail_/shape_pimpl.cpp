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

#include "../make_tot_shape.hpp"
#include "tensorwrapper/tensor/shapes/detail_/shape_pimpl.hpp"
#include <catch2/catch.hpp>

using namespace tensorwrapper::tensor;
using namespace tensorwrapper::sparse_map;

TEST_CASE("ShapePIMPL<Scalar>") {
    using field_type   = field::Scalar;
    using pimpl_type   = tensorwrapper::tensor::detail_::ShapePIMPL<field_type>;
    using extents_type = typename pimpl_type::extents_type;
    using inner_extents_type = typename pimpl_type::inner_extents_type;
    using tiling_type        = typename pimpl_type::tiling_type;

    extents_type scalar_extents;
    extents_type vector_extents{3};
    extents_type matrix_extents{3, 4};

    tiling_type scalar_tiling;
    tiling_type vector_span_tiling{{0, 3}};
    tiling_type matrix_span_tiling{{0, 3}, {0, 4}};
    tiling_type vector_block_tiling{{0, 1, 2, 3}};
    tiling_type matrix_block_tiling{{0, 1, 2, 3}, {0, 1, 2, 3, 4}};

    pimpl_type defaulted;
    pimpl_type scalar(scalar_extents);
    pimpl_type vector_from_extents(vector_extents);
    pimpl_type matrix_from_extents(matrix_extents);
    pimpl_type vector_from_tiling(vector_block_tiling);
    pimpl_type matrix_from_tiling(matrix_block_tiling);

    SECTION("Sanity") {
        using size_type = typename pimpl_type::size_type;
        REQUIRE(std::is_same_v<inner_extents_type, size_type>);
        REQUIRE(vector_from_extents.inner_extents() == 1);
        REQUIRE(vector_from_extents.field_rank() == 0);
    }

    SECTION("CTors") {
        SECTION("Default") {
            REQUIRE(defaulted.extents() == scalar_extents);
            REQUIRE(defaulted.tiling() == scalar_tiling);
        }

        SECTION("Value") {
            REQUIRE(scalar.extents() == scalar_extents);
            REQUIRE(vector_from_extents.extents() == vector_extents);
            REQUIRE(matrix_from_extents.extents() == matrix_extents);
            REQUIRE(vector_from_tiling.extents() == vector_extents);
            REQUIRE(matrix_from_tiling.extents() == matrix_extents);

            REQUIRE(scalar.tiling() == scalar_tiling);
            REQUIRE(vector_from_extents.tiling() == vector_span_tiling);
            REQUIRE(matrix_from_extents.tiling() == matrix_span_tiling);
            REQUIRE(vector_from_tiling.tiling() == vector_block_tiling);
            REQUIRE(matrix_from_tiling.tiling() == matrix_block_tiling);

            // Make sure object is forwarded correctly (i.e. no copy)
            auto pt = matrix_extents.data();
            pimpl_type matrix(std::move(matrix_extents));
            REQUIRE(matrix.extents().data() == pt);
        }
    }

    SECTION("clone()") {
        REQUIRE(*scalar.clone() == scalar);
        REQUIRE(*vector_from_extents.clone() == vector_from_extents);
        REQUIRE(*matrix_from_extents.clone() == matrix_from_extents);
        REQUIRE(*vector_from_tiling.clone() == vector_from_tiling);
        REQUIRE(*matrix_from_tiling.clone() == matrix_from_tiling);
    }

    SECTION("slice()") {
        SECTION("valid") {
            SECTION("tiles span extents") {
                auto vector_slice = vector_from_extents.slice({1}, {3});
                auto matrix_slice = matrix_from_extents.slice({0, 0}, {3, 3});

                pimpl_type corr_vector_slice(extents_type{2});
                pimpl_type corr_matrix_slice({3, 3});

                REQUIRE(*vector_slice == corr_vector_slice);
                REQUIRE(*matrix_slice == corr_matrix_slice);
            }

            SECTION("tiles don't span extents") {
                auto vector_slice = vector_from_tiling.slice({1}, {3});
                auto matrix_slice = matrix_from_tiling.slice({1, 1}, {3, 3});

                pimpl_type corr_vector_slice(tiling_type{{0, 1, 2}});
                pimpl_type corr_matrix_slice({{0, 1, 2}, {0, 1, 2}});

                REQUIRE(*vector_slice == corr_vector_slice);
                REQUIRE(*matrix_slice == corr_matrix_slice);
            }
        }

        SECTION("wrong bounds rank") {
            REQUIRE_THROWS_AS(vector_from_tiling.slice({0}, {0, 1}),
                              std::runtime_error);
            REQUIRE_THROWS_AS(vector_from_tiling.slice({0, 1}, {1}),
                              std::runtime_error);
            REQUIRE_THROWS_AS(vector_from_tiling.slice({0, 1}, {0, 1}),
                              std::runtime_error);
        }

        SECTION("hi < lo") {
            REQUIRE_THROWS_AS(vector_from_tiling.slice({1}, {0}),
                              std::runtime_error);
        }

        SECTION("out of bounds") {
            REQUIRE_THROWS_AS(vector_from_tiling.slice({0}, {4}),
                              std::runtime_error);
            REQUIRE_THROWS_AS(vector_from_tiling.slice({3}, {5}),
                              std::runtime_error);
        }
    }

    SECTION("extents() const") {
        REQUIRE(defaulted.extents() == scalar_extents);
        REQUIRE(scalar.extents() == scalar_extents);
        REQUIRE(vector_from_extents.extents() == vector_extents);
        REQUIRE(matrix_from_extents.extents() == matrix_extents);
        REQUIRE(vector_from_tiling.extents() == vector_extents);
        REQUIRE(matrix_from_tiling.extents() == matrix_extents);
    }

    SECTION("tiling() const") {
        REQUIRE(defaulted.tiling() == scalar_tiling);
        REQUIRE(scalar.tiling() == scalar_tiling);
        REQUIRE(vector_from_extents.tiling() == vector_span_tiling);
        REQUIRE(matrix_from_extents.tiling() == matrix_span_tiling);
        REQUIRE(vector_from_tiling.tiling() == vector_block_tiling);
        REQUIRE(matrix_from_tiling.tiling() == matrix_block_tiling);
    }

    SECTION("hash") {
        using tensorwrapper::detail_::hash_objects;
        REQUIRE(hash_objects(defaulted) == hash_objects(scalar));

        const auto v2_hash = hash_objects(pimpl_type(vector_extents));
        REQUIRE(hash_objects(vector_from_extents) == v2_hash);

        const auto m2_hash = hash_objects(pimpl_type(matrix_extents));
        REQUIRE(hash_objects(matrix_from_extents) == m2_hash);

        REQUIRE_FALSE(hash_objects(defaulted) ==
                      hash_objects(vector_from_extents));

        REQUIRE_FALSE(hash_objects(vector_from_extents) ==
                      hash_objects(matrix_from_extents));

        const auto v3_hash = hash_objects(pimpl_type(extents_type{5}));
        REQUIRE_FALSE(hash_objects(vector_from_tiling) == v3_hash);

        REQUIRE_FALSE(hash_objects(vector_from_extents) ==
                      hash_objects(vector_from_tiling));

        REQUIRE_FALSE(hash_objects(matrix_from_extents) ==
                      hash_objects(matrix_from_tiling));
    }

    SECTION("Equality") {
        REQUIRE(defaulted == scalar);
        REQUIRE(pimpl_type(vector_span_tiling) == pimpl_type(vector_extents));
        REQUIRE(pimpl_type(matrix_span_tiling) == pimpl_type(matrix_extents));

        // default doesn't equal filled
        REQUIRE_FALSE(defaulted == vector_from_extents);
        // different ranks
        REQUIRE_FALSE(vector_from_extents == matrix_from_extents);
        // different size
        REQUIRE_FALSE(vector_from_extents == pimpl_type(extents_type{5}));
        // different tiling
        REQUIRE_FALSE(vector_from_extents == vector_from_tiling);
    }
}

/// TODO: Update for tiling
TEST_CASE("ShapePIMPL<Tensor>") {
    using field_type   = field::Tensor;
    using pimpl_type   = tensorwrapper::tensor::detail_::ShapePIMPL<field_type>;
    using extents_type = typename pimpl_type::extents_type;
    using inner_extents_type = typename pimpl_type::inner_extents_type;
    using tiling_type        = typename pimpl_type::tiling_type;

    extents_type scalar_extents;
    extents_type vector_extents{3};
    extents_type matrix_extents{3, 4};

    tiling_type scalar_tiling;
    tiling_type vector_tiling{{0, 3}};
    tiling_type matrix_tiling{{0, 3}, {0, 4}};

    pimpl_type defaulted;
    auto vov = testing::make_uniform_tot_shape<pimpl_type>(vector_extents,
                                                           vector_extents);
    auto vom = testing::make_uniform_tot_shape<pimpl_type>(vector_extents,
                                                           matrix_extents);
    auto mov = testing::make_uniform_tot_shape<pimpl_type>(matrix_extents,
                                                           vector_extents);
    auto mom = testing::make_uniform_tot_shape<pimpl_type>(matrix_extents,
                                                           matrix_extents);

    SECTION("Sanity") {
        REQUIRE_THROWS_AS(pimpl_type(vector_extents), std::runtime_error);
    }

    SECTION("CTors") {
        SECTION("Default") {
            REQUIRE(defaulted.extents() == scalar_extents);
            REQUIRE(defaulted.tiling() == scalar_tiling);
        }

        SECTION("Uniform Inner Extents") {
            REQUIRE(vov.extents() == vector_extents);
            REQUIRE(vom.extents() == vector_extents);

            REQUIRE(mov.extents() == matrix_extents);
            REQUIRE(mom.extents() == matrix_extents);

            const auto& vov_ie = vov.inner_extents();
            const auto& vom_ie = vom.inner_extents();
            for(auto i = 0ul; i < 3; ++i) {
                Index idx({i});
                REQUIRE(vov_ie.at(idx).extents() == vector_extents);
                REQUIRE(vom_ie.at(idx).extents() == matrix_extents);
            }

            const auto& mov_ie = mov.inner_extents();
            const auto& mom_ie = mom.inner_extents();
            for(auto i = 0ul; i < 3; ++i) {
                for(auto j = 0ul; j < 4; ++j) {
                    Index idx({i, j});
                    REQUIRE(mov_ie.at(idx).extents() == vector_extents);
                    REQUIRE(mom_ie.at(idx).extents() == matrix_extents);
                }
            }

            // Make sure object is forwarded correctly (i.e. no copy)
            auto pt = matrix_extents.data();
            auto _dummy_map =
              testing::make_uniform_tot_map(matrix_extents, vector_extents);
            pimpl_type matrix2(std::move(matrix_extents),
                               std::move(_dummy_map));
            REQUIRE(matrix2.extents().data() == pt);
        }

        SECTION("Non-Uniform Inner Extents") {
            extents_type other_extents{5, 6};
            std::map<Index, Shape<field::Scalar>> inner_map = {
              {Index{0ul}, Shape<field::Scalar>(vector_extents)},
              {Index{1ul}, Shape<field::Scalar>(other_extents)},
              {Index{2ul}, Shape<field::Scalar>(vector_extents)}};
            pimpl_type nu_tot_shape(vector_extents, inner_map);
            REQUIRE(nu_tot_shape.extents() == vector_extents);
            REQUIRE(nu_tot_shape.inner_extents().at(Index{0ul}).extents() ==
                    vector_extents);
            REQUIRE(nu_tot_shape.inner_extents().at(Index{1ul}).extents() ==
                    other_extents);
            REQUIRE(nu_tot_shape.inner_extents().at(Index{0ul}).extents() ==
                    vector_extents);
        }
    }

    SECTION("clone()") {
        REQUIRE(*vov.clone() == vov);
        REQUIRE(*vom.clone() == vom);
        REQUIRE(*mov.clone() == mov);
        REQUIRE(*mom.clone() == mom);
    }

    SECTION("hash") {
        using tensorwrapper::detail_::hash_objects;

        auto test_hash = [&](auto& obj, auto& oe, auto&& ie) {
            const auto hash2 = hash_objects(pimpl_type(oe, ie));
            REQUIRE(hash_objects(obj) == hash2);
        };
        test_hash(
          vov, vector_extents,
          testing::make_uniform_tot_map(vector_extents, vector_extents));
        test_hash(
          vom, vector_extents,
          testing::make_uniform_tot_map(vector_extents, matrix_extents));
        test_hash(
          mov, matrix_extents,
          testing::make_uniform_tot_map(matrix_extents, vector_extents));
        test_hash(
          mom, matrix_extents,
          testing::make_uniform_tot_map(matrix_extents, matrix_extents));

        REQUIRE_FALSE(hash_objects(vov) == hash_objects(mom)); // both diff
        REQUIRE_FALSE(hash_objects(vom) == hash_objects(mov)); // extent swap

        auto test_hash_false = [&](auto& obj, auto oe, auto ie) {
            const auto hash2 = hash_objects(pimpl_type(oe, ie));
            REQUIRE_FALSE(hash_objects(obj) == hash2);
        };

        test_hash_false(
          vov, extents_type{5},
          testing::make_uniform_tot_map(extents_type{5}, vector_extents));
        test_hash_false(
          vov, vector_extents,
          testing::make_uniform_tot_map(vector_extents, extents_type{5}));
        test_hash_false(
          vov, extents_type{5},
          testing::make_uniform_tot_map(extents_type{5}, extents_type{5}));
    }

    SECTION("Equality") {
        REQUIRE(vov == testing::make_uniform_tot_shape<pimpl_type>(
                         vector_extents, vector_extents));
        REQUIRE(vom == testing::make_uniform_tot_shape<pimpl_type>(
                         vector_extents, matrix_extents));
        REQUIRE(mov == testing::make_uniform_tot_shape<pimpl_type>(
                         matrix_extents, vector_extents));
        REQUIRE(mom == testing::make_uniform_tot_shape<pimpl_type>(
                         matrix_extents, matrix_extents));

        REQUIRE_FALSE(defaulted == vov); // default doesn't equal filled
        REQUIRE_FALSE(vov == mom);       // different ranks
        REQUIRE_FALSE(vom == mov);       // swapped extents
        REQUIRE_FALSE(vov == testing::make_uniform_tot_shape<pimpl_type>(
                               extents_type{5}, extents_type{5}));
    }
}
