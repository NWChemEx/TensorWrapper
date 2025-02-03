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

    // Scalar times vector
    ContractionPlanner cp___i("", "", "i");
    ContractionPlanner cp__i_("", "i", "");

    // Vector times vector
    ContractionPlanner cp__i_i("", "i", "i");
    ContractionPlanner cp_i_i_i("i", "i", "i");
    ContractionPlanner cp_i_i_j("i", "i", "j");
    ContractionPlanner cp_ij_i_j("i,j", "i", "j");
    ContractionPlanner cp_ji_i_j("j,i", "i", "j");
    ContractionPlanner cp_i_j_i("i", "j", "i");

    // Vector times matrix
    ContractionPlanner cp_i_i_ij("i", "i", "i,j");
    ContractionPlanner cp_j_i_ij("j", "i", "i,j");
    ContractionPlanner cp_i_i_ji("i", "i", "j,i");
    ContractionPlanner cp_j_i_ji("j", "i", "j,i");
    ContractionPlanner cp_ij_i_ij("i,j", "i", "i,j");
    ContractionPlanner cp_ji_i_ij("j,i", "i", "i,j");
    ContractionPlanner cp_ij_i_ji("i,j", "i", "j,i");
    ContractionPlanner cp_ji_i_ji("j,i", "i", "j,i");

    // Tensor times tensor
    ContractionPlanner cp_ij_ijk_ikj("i,j", "i,j,k", "i,k,j");
    ContractionPlanner cp_iljm_ikj_lmk("i,l,j,m", "i,k,j", "l,m,k");

    // These are invalid
    // ContractionPlanner cpi__("i", "", "");

    SECTION("lhs_free") {
        REQUIRE(cp___.lhs_free() == "");

        REQUIRE(cp___i.lhs_free() == "");
        REQUIRE(cp__i_.lhs_free() == "");

        REQUIRE(cp__i_i.lhs_free() == "");
        REQUIRE(cp_i_i_i.lhs_free() == "i");
        REQUIRE(cp_i_i_j.lhs_free() == "i");
        REQUIRE(cp_ij_i_j.lhs_free() == "i");
        REQUIRE(cp_ji_i_j.lhs_free() == "i");
        REQUIRE(cp_i_j_i.lhs_free() == "");

        REQUIRE(cp_i_i_ij.lhs_free() == "i");
        REQUIRE(cp_j_i_ij.lhs_free() == "");
        REQUIRE(cp_i_i_ji.lhs_free() == "i");
        REQUIRE(cp_j_i_ji.lhs_free() == "");
        REQUIRE(cp_ij_i_ij.lhs_free() == "i");
        REQUIRE(cp_ji_i_ij.lhs_free() == "i");
        REQUIRE(cp_ij_i_ji.lhs_free() == "i");
        REQUIRE(cp_ji_i_ji.lhs_free() == "i");

        REQUIRE(cp_ij_ijk_ikj.lhs_free() == "i,j");
        REQUIRE(cp_iljm_ikj_lmk.lhs_free() == "i,j");
    }

    SECTION("rhs_free") {
        REQUIRE(cp___.rhs_free() == "");

        REQUIRE(cp___i.rhs_free() == "");
        REQUIRE(cp__i_.rhs_free() == "");

        REQUIRE(cp__i_i.rhs_free() == "");
        REQUIRE(cp_i_i_i.rhs_free() == "i");
        REQUIRE(cp_i_i_j.rhs_free() == "");
        REQUIRE(cp_ij_i_j.rhs_free() == "j");
        REQUIRE(cp_ji_i_j.rhs_free() == "j");
        REQUIRE(cp_i_j_i.rhs_free() == "i");

        REQUIRE(cp_i_i_ij.rhs_free() == "i");
        REQUIRE(cp_j_i_ij.rhs_free() == "j");
        REQUIRE(cp_i_i_ji.rhs_free() == "i");
        REQUIRE(cp_j_i_ji.rhs_free() == "j");
        REQUIRE(cp_ij_i_ij.rhs_free() == "i,j");
        REQUIRE(cp_ji_i_ij.rhs_free() == "i,j");
        REQUIRE(cp_ij_i_ji.rhs_free() == "j,i");
        REQUIRE(cp_ji_i_ji.rhs_free() == "j,i");

        REQUIRE(cp_ij_ijk_ikj.rhs_free() == "i,j");
        REQUIRE(cp_iljm_ikj_lmk.rhs_free() == "l,m");
    }
}
