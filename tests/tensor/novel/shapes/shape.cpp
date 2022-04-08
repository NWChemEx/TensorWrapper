#include "tensorwrapper/tensor/novel/shapes/shapes.hpp"
#include <catch2/catch.hpp>

using namespace tensorwrapper::tensor;
using namespace tensorwrapper::tensor::novel;

/* Testing Strategy:
 *
 *
 * For both specializations we assume that the underlying PIMPLs work. Thus for
 * functions which forward to the PIMPL we only need to check if the forwarding
 * works, which can be done with one sample input. For polymorphic functions we
 * leave it to derived classes to ensure they interact correctly with the base
 * class, and only test functionality explicitly implemented in the base class
 * in these tests.
 *
 */

TEST_CASE("Shape<Scalar>") {
    using field_type         = field::Scalar;
    using other_field        = field::Tensor;
    using shape_type         = Shape<field_type>;
    using extents_type       = typename shape_type::extents_type;
    using inner_extents_type = typename shape_type::inner_extents_type;

    extents_type vector_extents{4};
    extents_type matrix_extents{3, 5};

    shape_type defaulted;
    shape_type vector(vector_extents);
    shape_type matrix(matrix_extents);

    SECTION("Sanity") {
        using size_type = typename shape_type::size_type;
        REQUIRE(std::is_same_v<inner_extents_type, size_type>);
        REQUIRE(vector.inner_extents() == 1);
        REQUIRE(vector.field_rank() == 0);
    }

    SECTION("CTors") {
        SECTION("Value") {
            REQUIRE(vector.extents() == vector_extents);
            REQUIRE(matrix.extents() == matrix_extents);

            // Ensure that extents are properly moved
            auto* vp = vector_extents.data();
            shape_type v2(std::move(vector_extents));
            REQUIRE(v2.extents().data() == vp);
        }

        SECTION("Clone") {
            auto pv = vector.clone();
            REQUIRE(*pv == vector);
        }
    }

    SECTION("extents") {
        REQUIRE_THROWS_AS(defaulted.extents(), std::runtime_error);
        REQUIRE(vector.extents() == vector_extents);
        REQUIRE(matrix.extents() == matrix_extents);
    }

    SECTION("is_zero") {
        // Everything is non-zero for non-sparse shape
        REQUIRE_FALSE(vector.is_zero({0}, {1}));
        REQUIRE_FALSE(vector.is_zero({0}, {2}));
        REQUIRE_FALSE(vector.is_zero({0}, {4}));
        REQUIRE_FALSE(vector.is_zero({2}, {2}));
        REQUIRE_FALSE(vector.is_zero({2}, {4}));

        REQUIRE_FALSE(matrix.is_zero({0, 0}, {3, 5}));
    }

    SECTION("Comparisons") {
        // LHS is defaulted
        REQUIRE(defaulted == shape_type{});
        REQUIRE_FALSE(defaulted != shape_type{});
        REQUIRE_FALSE(defaulted == vector);
        REQUIRE(defaulted != vector);
        REQUIRE_FALSE(defaulted == matrix);
        REQUIRE(defaulted != matrix);

        // LHS is vector
        REQUIRE(vector == shape_type(vector_extents));
        REQUIRE_FALSE(vector != shape_type(vector_extents));
        REQUIRE_FALSE(vector == matrix);
        REQUIRE(vector != matrix);

        // Different Fields
        REQUIRE(defaulted != Shape<other_field>{});
        REQUIRE_FALSE(defaulted == Shape<other_field>{});
    }

    SECTION("hash") {
        using tensorwrapper::detail_::hash_objects;

        SECTION("LHS is defaulted") {
            auto lhs = hash_objects(defaulted);

            REQUIRE(lhs == hash_objects(shape_type{}));
            REQUIRE(lhs != hash_objects(vector));
            REQUIRE(lhs != hash_objects(matrix));
            // TODO: enable when hashing properly accounts for types
            // REQUIRE(lhs != hash_objects(Shape<other_field>{}));
        }

        SECTION("LHS is vector") {
            auto lhs = hash_objects(vector);

            REQUIRE(lhs == hash_objects(shape_type(vector_extents)));
            REQUIRE(lhs != hash_objects(matrix));
        }
    }
}

TEST_CASE("Shape<Tensor>") {
    using field_type         = field::Tensor;
    using other_field        = field::Scalar;
    using shape_type         = Shape<field_type>;
    using extents_type       = typename shape_type::extents_type;
    using inner_extents_type = typename shape_type::inner_extents_type;

    extents_type vector_extents{3};
    extents_type matrix_extents{3, 4};

    shape_type defaulted;
    shape_type vov(vector_extents, vector_extents);
    shape_type vom(vector_extents, matrix_extents);
    shape_type mom(matrix_extents, matrix_extents);

    SECTION("Sanity") {
        REQUIRE(std::is_same_v<extents_type, inner_extents_type>);
        REQUIRE_THROWS_AS(shape_type(vector_extents), std::runtime_error);
    }

    SECTION("CTors") {
        SECTION("Value") {
            REQUIRE(vov.extents() == vector_extents);
            REQUIRE(vom.extents() == vector_extents);
            REQUIRE(mom.extents() == matrix_extents);

            REQUIRE(vov.inner_extents() == vector_extents);
            REQUIRE(vom.inner_extents() == matrix_extents);
            REQUIRE(mom.inner_extents() == matrix_extents);

            // Make sure object is forwarded correctly (i.e. no copy)
            auto pm = matrix_extents.data();
            auto pv = vector_extents.data();
            shape_type tensor2(std::move(matrix_extents),
                               std::move(vector_extents));
            REQUIRE(tensor2.extents().data() == pm);
            REQUIRE(tensor2.inner_extents().data() == pv);
        }

        SECTION("Clone") {
            auto pvov = vov.clone();
            REQUIRE(*pvov == vov);
        }
    }

    SECTION("extents") {
        REQUIRE_THROWS_AS(defaulted.extents(), std::runtime_error);
        REQUIRE_THROWS_AS(defaulted.inner_extents(), std::runtime_error);
        REQUIRE(vov.extents() == vector_extents);
        REQUIRE(vom.extents() == vector_extents);
        REQUIRE(mom.extents() == matrix_extents);
        REQUIRE(vov.inner_extents() == vector_extents);
        REQUIRE(vom.inner_extents() == matrix_extents);
        REQUIRE(mom.inner_extents() == matrix_extents);
    }

    SECTION("Comparisons") {
        // LHS is defaulted
        REQUIRE(defaulted == shape_type{});
        REQUIRE_FALSE(defaulted != shape_type{});
        REQUIRE_FALSE(defaulted == vov);
        REQUIRE(defaulted != vov);
        REQUIRE_FALSE(defaulted == vom);
        REQUIRE(defaulted != vom);

        // LHS is vector
        REQUIRE(vov == shape_type(vector_extents, vector_extents));
        REQUIRE_FALSE(vov != shape_type(vector_extents, vector_extents));
        REQUIRE_FALSE(vov == vom);
        REQUIRE_FALSE(vov == mom);
        REQUIRE(vov != mom);
        REQUIRE(vom != mom);

        // Different Fields
        REQUIRE(defaulted != Shape<other_field>{});
        REQUIRE_FALSE(defaulted == Shape<other_field>{});
    }

    SECTION("hash") {
        using tensorwrapper::detail_::hash_objects;

        SECTION("LHS is defaulted") {
            auto lhs = hash_objects(defaulted);

            REQUIRE(lhs == hash_objects(shape_type{}));
            REQUIRE(lhs != hash_objects(vov));
            REQUIRE(lhs != hash_objects(vom));
            REQUIRE(lhs != hash_objects(mom));
            // TODO: enable when hashing properly accounts for types
            // REQUIRE(lhs != hash_objects(Shape<other_field>{}));
        }

        SECTION("LHS is vector") {
            auto lhs = hash_objects(vov);

            REQUIRE(lhs ==
                    hash_objects(shape_type(vector_extents, vector_extents)));
            REQUIRE(lhs != hash_objects(vom));
            REQUIRE(lhs != hash_objects(mom));
        }
    }
}
