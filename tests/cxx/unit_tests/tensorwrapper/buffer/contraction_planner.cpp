/*
 * Copyright 2025 NWChemEx-Project
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

#include "../testing/testing.hpp"
#include <tensorwrapper/buffer/contraction_planner.hpp>
#include <tensorwrapper/buffer/eigen.hpp>

using namespace tensorwrapper;
using namespace buffer;

TEST_CASE("ContractionPlanner") {
    // All scalar
    ContractionPlanner cp___("", "", "");

    // Vector times vector
    ContractionPlanner cp__i_i("", "i", "i");
    ContractionPlanner cp_ij_i_j("i,j", "i", "j");
    ContractionPlanner cp_ji_i_j("j,i", "i", "j");

    // Vector times matrix
    ContractionPlanner cp_j_i_ij("j", "i", "i,j");
    ContractionPlanner cp_j_i_ji("j", "i", "j,i");
    ContractionPlanner cp_ijk_i_jk("i,j,k", "i", "j,k");
    ContractionPlanner cp_ijk_i_kj("i,j,k", "i", "k,j");

    // Matrix times matrix
    ContractionPlanner cp_ij_ik_kj("i,j", "i,k", "k,j");
    ContractionPlanner cp_ji_ik_kj("j,i", "i,k", "k,j");
    ContractionPlanner cp_ij_ik_jk("i,j", "i,k", "j,k");
    ContractionPlanner cp_ji_ik_jk("j,i", "i,k", "j,k");

    // 3 times 3
    ContractionPlanner cp__ijk_ijk("", "i,j,k", "i,j,k");
    ContractionPlanner cp__ijk_jik("", "i,j,k", "j,i,k");
    ContractionPlanner cp_il_ijk_jkl("i,l", "i,j,k", "j,k,l");
    ContractionPlanner cp_il_ijk_klj("i,l", "i,j,k", "k,l,j");

    SECTION("Ctors") {
        using error_t = std::runtime_error;

        // Can't contain repeated indices
        REQUIRE_THROWS_AS(ContractionPlanner("j", "i,i", "j"), error_t);
        REQUIRE_THROWS_AS(ContractionPlanner("j", "j", "i,i"), error_t);

        // Can't require trace of a tensor
        REQUIRE_THROWS_AS(ContractionPlanner("", "", "i"), error_t);

        // Can't contain Hadamard
        REQUIRE_THROWS_AS(ContractionPlanner("i", "i", "i"), error_t);
    }

    SECTION("lhs_free") {
        REQUIRE(cp___.lhs_free() == "");

        REQUIRE(cp__i_i.lhs_free() == "");
        REQUIRE(cp_ij_i_j.lhs_free() == "i");
        REQUIRE(cp_ji_i_j.lhs_free() == "i");

        REQUIRE(cp_j_i_ij.lhs_free() == "");
        REQUIRE(cp_j_i_ji.lhs_free() == "");
        REQUIRE(cp_ijk_i_jk.lhs_free() == "i");
        REQUIRE(cp_ijk_i_kj.lhs_free() == "i");

        REQUIRE(cp_ij_ik_kj.lhs_free() == "i");
        REQUIRE(cp_ji_ik_kj.lhs_free() == "i");
        REQUIRE(cp_ij_ik_jk.lhs_free() == "i");
        REQUIRE(cp_ji_ik_jk.lhs_free() == "i");

        REQUIRE(cp__ijk_ijk.lhs_free() == "");
        REQUIRE(cp__ijk_jik.lhs_free() == "");
        REQUIRE(cp_il_ijk_jkl.lhs_free() == "i");
        REQUIRE(cp_il_ijk_klj.lhs_free() == "i");
    }

    SECTION("rhs_free") {
        REQUIRE(cp___.rhs_free() == "");

        REQUIRE(cp__i_i.rhs_free() == "");
        REQUIRE(cp_ij_i_j.rhs_free() == "j");
        REQUIRE(cp_ji_i_j.rhs_free() == "j");

        REQUIRE(cp_j_i_ij.rhs_free() == "j");
        REQUIRE(cp_j_i_ji.rhs_free() == "j");
        REQUIRE(cp_ijk_i_jk.rhs_free() == "j,k");
        REQUIRE(cp_ijk_i_kj.rhs_free() == "k,j");

        REQUIRE(cp_ij_ik_kj.rhs_free() == "j");
        REQUIRE(cp_ji_ik_kj.rhs_free() == "j");
        REQUIRE(cp_ij_ik_jk.rhs_free() == "j");
        REQUIRE(cp_ji_ik_jk.rhs_free() == "j");

        REQUIRE(cp__ijk_ijk.rhs_free() == "");
        REQUIRE(cp__ijk_jik.rhs_free() == "");
        REQUIRE(cp_il_ijk_jkl.rhs_free() == "l");
        REQUIRE(cp_il_ijk_klj.rhs_free() == "l");
    }

    SECTION("lhs_dummy") {
        REQUIRE(cp___.lhs_dummy() == "");

        REQUIRE(cp__i_i.lhs_dummy() == "i");
        REQUIRE(cp_ij_i_j.lhs_dummy() == "");
        REQUIRE(cp_ji_i_j.lhs_dummy() == "");

        REQUIRE(cp_j_i_ij.lhs_dummy() == "i");
        REQUIRE(cp_j_i_ji.lhs_dummy() == "i");
        REQUIRE(cp_ijk_i_jk.lhs_dummy() == "");
        REQUIRE(cp_ijk_i_kj.lhs_dummy() == "");

        REQUIRE(cp_ij_ik_kj.lhs_dummy() == "k");
        REQUIRE(cp_ji_ik_kj.lhs_dummy() == "k");
        REQUIRE(cp_ij_ik_jk.lhs_dummy() == "k");
        REQUIRE(cp_ji_ik_jk.lhs_dummy() == "k");

        REQUIRE(cp__ijk_ijk.lhs_dummy() == "i,j,k");
        REQUIRE(cp__ijk_jik.lhs_dummy() == "i,j,k");
        REQUIRE(cp_il_ijk_jkl.lhs_dummy() == "j,k");
        REQUIRE(cp_il_ijk_klj.lhs_dummy() == "j,k");
    }

    SECTION("rhs_dummy") {
        REQUIRE(cp___.rhs_dummy() == "");

        REQUIRE(cp__i_i.rhs_dummy() == "i");
        REQUIRE(cp_ij_i_j.rhs_dummy() == "");
        REQUIRE(cp_ji_i_j.rhs_dummy() == "");

        REQUIRE(cp_j_i_ij.rhs_dummy() == "i");
        REQUIRE(cp_j_i_ji.rhs_dummy() == "i");
        REQUIRE(cp_ijk_i_jk.rhs_dummy() == "");
        REQUIRE(cp_ijk_i_kj.rhs_dummy() == "");

        REQUIRE(cp_ij_ik_kj.rhs_dummy() == "k");
        REQUIRE(cp_ji_ik_kj.rhs_dummy() == "k");
        REQUIRE(cp_ij_ik_jk.rhs_dummy() == "k");
        REQUIRE(cp_ji_ik_jk.rhs_dummy() == "k");

        REQUIRE(cp__ijk_ijk.rhs_dummy() == "i,j,k");
        REQUIRE(cp__ijk_jik.rhs_dummy() == "j,i,k");
        REQUIRE(cp_il_ijk_jkl.rhs_dummy() == "j,k");
        REQUIRE(cp_il_ijk_klj.rhs_dummy() == "k,j");
    }

    SECTION("lhs_permutation") {
        REQUIRE(cp___.lhs_permutation() == "");

        REQUIRE(cp__i_i.lhs_permutation() == "i");
        REQUIRE(cp_ij_i_j.lhs_permutation() == "i");
        REQUIRE(cp_ji_i_j.lhs_permutation() == "i");

        REQUIRE(cp_j_i_ij.lhs_permutation() == "i");
        REQUIRE(cp_j_i_ji.lhs_permutation() == "i");
        REQUIRE(cp_ijk_i_jk.lhs_permutation() == "i");
        REQUIRE(cp_ijk_i_kj.lhs_permutation() == "i");

        REQUIRE(cp_ij_ik_kj.lhs_permutation() == "i,k");
        REQUIRE(cp_ji_ik_kj.lhs_permutation() == "i,k");
        REQUIRE(cp_ij_ik_jk.lhs_permutation() == "i,k");
        REQUIRE(cp_ji_ik_jk.lhs_permutation() == "i,k");

        REQUIRE(cp__ijk_ijk.lhs_permutation() == "i,j,k");
        REQUIRE(cp__ijk_jik.lhs_permutation() == "i,j,k");
        REQUIRE(cp_il_ijk_jkl.lhs_permutation() == "i,j,k");
        REQUIRE(cp_il_ijk_klj.lhs_permutation() == "i,j,k");
    }

    SECTION("rhs_permutation") {
        REQUIRE(cp___.rhs_permutation() == "");

        REQUIRE(cp__i_i.rhs_permutation() == "i");
        REQUIRE(cp_ij_i_j.rhs_permutation() == "j");
        REQUIRE(cp_ji_i_j.rhs_permutation() == "j");

        REQUIRE(cp_j_i_ij.rhs_permutation() == "i,j");
        REQUIRE(cp_j_i_ji.rhs_permutation() == "i,j");
        REQUIRE(cp_ijk_i_jk.rhs_permutation() == "j,k");
        REQUIRE(cp_ijk_i_kj.rhs_permutation() == "j,k");

        REQUIRE(cp_ij_ik_kj.rhs_permutation() == "k,j");
        REQUIRE(cp_ji_ik_kj.rhs_permutation() == "k,j");
        REQUIRE(cp_ij_ik_jk.rhs_permutation() == "k,j");
        REQUIRE(cp_ji_ik_jk.rhs_permutation() == "k,j");

        REQUIRE(cp__ijk_ijk.rhs_permutation() == "i,j,k");
        REQUIRE(cp__ijk_jik.rhs_permutation() == "i,j,k");
        REQUIRE(cp_il_ijk_jkl.rhs_permutation() == "j,k,l");
        REQUIRE(cp_il_ijk_klj.rhs_permutation() == "j,k,l");
    }
}
