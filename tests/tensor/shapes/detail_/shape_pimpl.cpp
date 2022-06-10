#include "../make_tot_shape.hpp"
#include "tensorwrapper/tensorshapes/detail_/shape_pimpl.hpp"
#include <catch2/catch.hpp>

using namespace tensorwrapper::tensor;
using namespace tensorwrapper::tensor::novel;
using namespace tensorwrapper::sparse_map;

TEST_CASE("ShapePIMPL<Scalar>") {
    using field_type         = field::Scalar;
    using pimpl_type         = detail_::ShapePIMPL<field_type>;
    using extents_type       = typename pimpl_type::extents_type;
    using inner_extents_type = typename pimpl_type::inner_extents_type;

    extents_type scalar_extents;
    extents_type vector_extents{3};
    extents_type matrix_extents{3, 4};
    extents_type tensor_extents{3, 4, 5};

    pimpl_type defaulted;
    pimpl_type scalar(scalar_extents);
    pimpl_type vector(vector_extents);
    pimpl_type matrix(matrix_extents);
    pimpl_type tensor(tensor_extents);

    SECTION("Sanity") {
        using size_type = typename pimpl_type::size_type;
        REQUIRE(std::is_same_v<inner_extents_type, size_type>);
        REQUIRE(vector.inner_extents() == 1);
        REQUIRE(vector.field_rank() == 0);
    }

    SECTION("CTors") {
        SECTION("Default") { REQUIRE(defaulted.extents() == scalar_extents); }

        SECTION("Value") {
            REQUIRE(scalar.extents() == scalar_extents);
            REQUIRE(vector.extents() == vector_extents);
            REQUIRE(matrix.extents() == matrix_extents);
            REQUIRE(tensor.extents() == tensor_extents);

            // Make sure object is forwarded correctly (i.e. no copy)
            auto pt = tensor_extents.data();
            pimpl_type tensor2(std::move(tensor_extents));
            REQUIRE(tensor2.extents().data() == pt);
        }
    }

    SECTION("clone()") {
        REQUIRE(*scalar.clone() == scalar);
        REQUIRE(*vector.clone() == vector);
        REQUIRE(*matrix.clone() == matrix);
        REQUIRE(*tensor.clone() == tensor);
    }

    SECTION("slice()") {
        SECTION("valid") {
            auto vector_slice = vector.slice({1}, {3});
            auto matrix_slice = matrix.slice({0, 0}, {3, 3});
            auto tensor_slice = tensor.slice({0, 1, 3}, {3, 2, 4});

            pimpl_type corr_vector_slice({2});
            pimpl_type corr_matrix_slice({3, 3});
            pimpl_type corr_tensor_slice({3, 1, 1});

            REQUIRE(*vector_slice == corr_vector_slice);
            REQUIRE(*matrix_slice == corr_matrix_slice);
            REQUIRE(*tensor_slice == corr_tensor_slice);
        }

        SECTION("wrong bounds rank") {
            REQUIRE_THROWS_AS(vector.slice({0}, {0, 1}), std::runtime_error);
            REQUIRE_THROWS_AS(vector.slice({0, 1}, {1}), std::runtime_error);
            REQUIRE_THROWS_AS(vector.slice({0, 1}, {0, 1}), std::runtime_error);
        }

        SECTION("hi < lo") {
            REQUIRE_THROWS_AS(vector.slice({1}, {0}), std::runtime_error);
        }

        SECTION("out of bounds") {
            REQUIRE_THROWS_AS(vector.slice({0}, {4}), std::runtime_error);
            REQUIRE_THROWS_AS(vector.slice({3}, {5}), std::runtime_error);
        }
    }

    SECTION("extents() const") {
        REQUIRE(defaulted.extents() == scalar_extents);
        REQUIRE(scalar.extents() == scalar_extents);
        REQUIRE(vector.extents() == vector_extents);
        REQUIRE(matrix.extents() == matrix_extents);
        REQUIRE(tensor.extents() == tensor_extents);
    }

    SECTION("hash") {
        using tensorwrapper::detail_::hash_objects;
        REQUIRE(hash_objects(defaulted) == hash_objects(scalar));

        const auto v2_hash = hash_objects(pimpl_type(vector_extents));
        REQUIRE(hash_objects(vector) == v2_hash);

        const auto m2_hash = hash_objects(pimpl_type(matrix_extents));
        REQUIRE(hash_objects(matrix) == m2_hash);

        REQUIRE_FALSE(hash_objects(defaulted) == hash_objects(vector));

        REQUIRE_FALSE(hash_objects(vector) == hash_objects(matrix));

        const auto v3_hash = hash_objects(pimpl_type(extents_type{5}));
        REQUIRE_FALSE(hash_objects(vector) == v3_hash);
    }

    SECTION("Equality") {
        REQUIRE(defaulted == scalar);
        REQUIRE(vector == pimpl_type(vector_extents));
        REQUIRE(matrix == pimpl_type(matrix_extents));

        REQUIRE_FALSE(defaulted == vector); // default doesn't equal filled
        REQUIRE_FALSE(vector == matrix);    // different ranks
        REQUIRE_FALSE(vector == pimpl_type(extents_type{5})); // different size
    }
}

TEST_CASE("ShapePIMPL<Tensor>") {
    using field_type         = field::Tensor;
    using pimpl_type         = detail_::ShapePIMPL<field_type>;
    using extents_type       = typename pimpl_type::extents_type;
    using inner_extents_type = typename pimpl_type::inner_extents_type;

    extents_type scalar_extents;
    extents_type vector_extents{3};
    extents_type matrix_extents{3, 4};
    extents_type tensor_extents{3, 4, 5};

    pimpl_type defaulted;
    auto vov = testing::make_uniform_tot_shape<pimpl_type>(vector_extents,
                                                           vector_extents);
    auto vom = testing::make_uniform_tot_shape<pimpl_type>(vector_extents,
                                                           matrix_extents);
    auto vot = testing::make_uniform_tot_shape<pimpl_type>(vector_extents,
                                                           tensor_extents);
    auto mov = testing::make_uniform_tot_shape<pimpl_type>(matrix_extents,
                                                           vector_extents);
    auto mom = testing::make_uniform_tot_shape<pimpl_type>(matrix_extents,
                                                           matrix_extents);
    auto mot = testing::make_uniform_tot_shape<pimpl_type>(matrix_extents,
                                                           tensor_extents);
    auto tov = testing::make_uniform_tot_shape<pimpl_type>(tensor_extents,
                                                           vector_extents);
    auto tom = testing::make_uniform_tot_shape<pimpl_type>(tensor_extents,
                                                           matrix_extents);
    auto tot = testing::make_uniform_tot_shape<pimpl_type>(tensor_extents,
                                                           tensor_extents);

    SECTION("Sanity") {
        // REQUIRE(std::is_same_v<extents_type, inner_extents_type>);
        REQUIRE_THROWS_AS(pimpl_type(vector_extents), std::runtime_error);
    }

    SECTION("CTors") {
        SECTION("Default") { REQUIRE(defaulted.extents() == scalar_extents); }

        SECTION("Uniform Inner Extents") {
            REQUIRE(vov.extents() == vector_extents);
            REQUIRE(vom.extents() == vector_extents);
            REQUIRE(vot.extents() == vector_extents);

            REQUIRE(mov.extents() == matrix_extents);
            REQUIRE(mom.extents() == matrix_extents);
            REQUIRE(mot.extents() == matrix_extents);

            REQUIRE(tov.extents() == tensor_extents);
            REQUIRE(tom.extents() == tensor_extents);
            REQUIRE(tot.extents() == tensor_extents);

            const auto& vov_ie = vov.inner_extents();
            const auto& vom_ie = vom.inner_extents();
            const auto& vot_ie = vot.inner_extents();
            for(auto i = 0ul; i < 3; ++i) {
                Index idx({i});
                REQUIRE(vov_ie.at(idx).extents() == vector_extents);
                REQUIRE(vom_ie.at(idx).extents() == matrix_extents);
                REQUIRE(vot_ie.at(idx).extents() == tensor_extents);
            }

            const auto& mov_ie = mov.inner_extents();
            const auto& mom_ie = mom.inner_extents();
            const auto& mot_ie = mot.inner_extents();
            for(auto i = 0ul; i < 3; ++i)
                for(auto j = 0ul; j < 4; ++j) {
                    Index idx({i, j});
                    REQUIRE(mov_ie.at(idx).extents() == vector_extents);
                    REQUIRE(mom_ie.at(idx).extents() == matrix_extents);
                    REQUIRE(mot_ie.at(idx).extents() == tensor_extents);
                }

            const auto& tov_ie = tov.inner_extents();
            const auto& tom_ie = tom.inner_extents();
            const auto& tot_ie = tot.inner_extents();
            for(auto i = 0ul; i < 3; ++i)
                for(auto j = 0ul; j < 4; ++j)
                    for(auto k = 0ul; k < 5; ++k) {
                        Index idx({i, j, k});
                        REQUIRE(tov_ie.at(idx).extents() == vector_extents);
                        REQUIRE(tom_ie.at(idx).extents() == matrix_extents);
                        REQUIRE(tot_ie.at(idx).extents() == tensor_extents);
                    }

            // Make sure object is forwarded correctly (i.e. no copy)
            auto pt = tensor_extents.data();
            auto _dummy_map =
              testing::make_uniform_tot_map(tensor_extents, vector_extents);
            pimpl_type tensor2(std::move(tensor_extents),
                               std::move(_dummy_map));
            REQUIRE(tensor2.extents().data() == pt);
            // REQUIRE(tensor2.inner_extents().data() == pv);
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
        REQUIRE(*vot.clone() == vot);
        REQUIRE(*mov.clone() == mov);
        REQUIRE(*mom.clone() == mom);
        REQUIRE(*mot.clone() == mot);
        REQUIRE(*tov.clone() == tov);
        REQUIRE(*tom.clone() == tom);
        REQUIRE(*tot.clone() == tot);
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
          vot, vector_extents,
          testing::make_uniform_tot_map(vector_extents, tensor_extents));
        test_hash(
          mov, matrix_extents,
          testing::make_uniform_tot_map(matrix_extents, vector_extents));
        test_hash(
          mom, matrix_extents,
          testing::make_uniform_tot_map(matrix_extents, matrix_extents));
        test_hash(
          mot, matrix_extents,
          testing::make_uniform_tot_map(matrix_extents, tensor_extents));
        test_hash(
          tov, tensor_extents,
          testing::make_uniform_tot_map(tensor_extents, vector_extents));
        test_hash(
          tom, tensor_extents,
          testing::make_uniform_tot_map(tensor_extents, matrix_extents));
        test_hash(
          tot, tensor_extents,
          testing::make_uniform_tot_map(tensor_extents, tensor_extents));

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
        REQUIRE(vot == testing::make_uniform_tot_shape<pimpl_type>(
                         vector_extents, tensor_extents));
        REQUIRE(mov == testing::make_uniform_tot_shape<pimpl_type>(
                         matrix_extents, vector_extents));
        REQUIRE(mom == testing::make_uniform_tot_shape<pimpl_type>(
                         matrix_extents, matrix_extents));
        REQUIRE(mot == testing::make_uniform_tot_shape<pimpl_type>(
                         matrix_extents, tensor_extents));
        REQUIRE(tov == testing::make_uniform_tot_shape<pimpl_type>(
                         tensor_extents, vector_extents));
        REQUIRE(tom == testing::make_uniform_tot_shape<pimpl_type>(
                         tensor_extents, matrix_extents));
        REQUIRE(tot == testing::make_uniform_tot_shape<pimpl_type>(
                         tensor_extents, tensor_extents));

        REQUIRE_FALSE(defaulted == vov); // default doesn't equal filled
        REQUIRE_FALSE(vov == mom);       // different ranks
        REQUIRE_FALSE(vom == mov);       // swapped extents
        REQUIRE_FALSE(vov == testing::make_uniform_tot_shape<pimpl_type>(
                               extents_type{5}, extents_type{5}));
    }
}
