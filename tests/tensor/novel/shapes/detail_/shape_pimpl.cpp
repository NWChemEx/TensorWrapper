#include "tensorwrapper/tensor/novel/shapes/detail_/shape_pimpl.hpp"
#include <catch2/catch.hpp>

using namespace tensorwrapper::tensor;
using namespace tensorwrapper::tensor::novel;

TEST_CASE("ShapePIMPL<Scalar>") {
    using field_type         = field::Scalar;
    using pimpl_type         = novel::detail_::ShapePIMPL<field_type>;
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
    using pimpl_type         = novel::detail_::ShapePIMPL<field_type>;
    using extents_type       = typename pimpl_type::extents_type;
    using inner_extents_type = typename pimpl_type::inner_extents_type;

    extents_type scalar_extents;
    extents_type vector_extents{3};
    extents_type matrix_extents{3, 4};
    extents_type tensor_extents{3, 4, 5};

    pimpl_type defaulted;
    pimpl_type vov(vector_extents, vector_extents);
    pimpl_type vom(vector_extents, matrix_extents);
    pimpl_type vot(vector_extents, tensor_extents);
    pimpl_type mov(matrix_extents, vector_extents);
    pimpl_type mom(matrix_extents, matrix_extents);
    pimpl_type mot(matrix_extents, tensor_extents);
    pimpl_type tov(tensor_extents, vector_extents);
    pimpl_type tom(tensor_extents, matrix_extents);
    pimpl_type tot(tensor_extents, tensor_extents);

    SECTION("Sanity") {
        REQUIRE(std::is_same_v<extents_type, inner_extents_type>);
        REQUIRE_THROWS_AS(pimpl_type(vector_extents), std::runtime_error);
    }

    SECTION("CTors") {
        SECTION("Default") { REQUIRE(defaulted.extents() == scalar_extents); }

        SECTION("Value") {
            REQUIRE(vov.extents() == vector_extents);
            REQUIRE(vom.extents() == vector_extents);
            REQUIRE(vot.extents() == vector_extents);

            REQUIRE(mov.extents() == matrix_extents);
            REQUIRE(mom.extents() == matrix_extents);
            REQUIRE(mot.extents() == matrix_extents);

            REQUIRE(tov.extents() == tensor_extents);
            REQUIRE(tom.extents() == tensor_extents);
            REQUIRE(tot.extents() == tensor_extents);

            REQUIRE(vov.inner_extents() == vector_extents);
            REQUIRE(mov.inner_extents() == vector_extents);
            REQUIRE(tov.inner_extents() == vector_extents);

            REQUIRE(vom.inner_extents() == matrix_extents);
            REQUIRE(mom.inner_extents() == matrix_extents);
            REQUIRE(tom.inner_extents() == matrix_extents);

            REQUIRE(vot.inner_extents() == tensor_extents);
            REQUIRE(mot.inner_extents() == tensor_extents);
            REQUIRE(tot.inner_extents() == tensor_extents);

            // Make sure object is forwarded correctly (i.e. no copy)
            auto pt = tensor_extents.data();
            auto pv = vector_extents.data();
            pimpl_type tensor2(std::move(tensor_extents),
                               std::move(vector_extents));
            REQUIRE(tensor2.extents().data() == pt);
            REQUIRE(tensor2.inner_extents().data() == pv);
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

        auto test_hash = [&](auto& obj, auto& oe, auto& ie) {
            const auto hash2 = hash_objects(pimpl_type(oe, ie));
            REQUIRE(hash_objects(obj) == hash2);
        };
        test_hash(vov, vector_extents, vector_extents);
        test_hash(vom, vector_extents, matrix_extents);
        test_hash(vot, vector_extents, tensor_extents);
        test_hash(mov, matrix_extents, vector_extents);
        test_hash(mom, matrix_extents, matrix_extents);
        test_hash(mot, matrix_extents, tensor_extents);
        test_hash(tov, tensor_extents, vector_extents);
        test_hash(tom, tensor_extents, matrix_extents);
        test_hash(tot, tensor_extents, tensor_extents);

        REQUIRE_FALSE(hash_objects(vov) == hash_objects(mom)); // both diff
        REQUIRE_FALSE(hash_objects(vom) == hash_objects(mov)); // extent swap

        auto test_hash_false = [&](auto& obj, auto oe, auto ie) {
            const auto hash2 = hash_objects(pimpl_type(oe, ie));
            REQUIRE_FALSE(hash_objects(obj) == hash2);
        };

        test_hash_false(vov, extents_type{5}, vector_extents);
        test_hash_false(vov, vector_extents, extents_type{5});
        test_hash_false(vov, extents_type{5}, extents_type{5});
    }

    SECTION("Equality") {
        REQUIRE(vov == pimpl_type(vector_extents, vector_extents));
        REQUIRE(vom == pimpl_type(vector_extents, matrix_extents));
        REQUIRE(vot == pimpl_type(vector_extents, tensor_extents));
        REQUIRE(mov == pimpl_type(matrix_extents, vector_extents));
        REQUIRE(mom == pimpl_type(matrix_extents, matrix_extents));
        REQUIRE(mot == pimpl_type(matrix_extents, tensor_extents));
        REQUIRE(tov == pimpl_type(tensor_extents, vector_extents));
        REQUIRE(tom == pimpl_type(tensor_extents, matrix_extents));
        REQUIRE(tot == pimpl_type(tensor_extents, tensor_extents));

        REQUIRE_FALSE(defaulted == vov); // default doesn't equal filled
        REQUIRE_FALSE(vov == mom);       // different ranks
        REQUIRE_FALSE(vom == mov);       // swapped extents
        REQUIRE_FALSE(vov == pimpl_type(extents_type{5}, extents_type{5}));
    }
}
