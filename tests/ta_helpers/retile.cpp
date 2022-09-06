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

#include "tensorwrapper/ta_helpers/retile.hpp"
#include <catch2/catch.hpp>

using namespace tensorwrapper::ta_helpers;
using ei = tensorwrapper::sparse_map::Index;

TEST_CASE("insert_tile_boundaries(TiledRange,vector)") {
    using vector_t = std::vector<ei>;
    TA::TiledRange tr0{{0, 2, 4, 6, 8, 10}};
    TA::TiledRange tr1{{0, 2, 4}, {0, 3, 9}};

    SECTION("Add no boundaries") {
        SECTION("vector") {
            auto r = insert_tile_boundaries(tr0, vector_t{});
            REQUIRE(r == tr0);
        }

        SECTION("matrix") {
            auto r = insert_tile_boundaries(tr1, vector_t{});
            REQUIRE(r == tr1);
        }
    }

    SECTION("Add one boundary") {
        SECTION("vector") {
            auto r = insert_tile_boundaries(tr0, vector_t{ei{1}});
            TA::TiledRange corr{{0, 1, 2, 4, 6, 8, 10}};
            REQUIRE(r == corr);
        }

        SECTION("matrix") {
            auto r = insert_tile_boundaries(tr1, vector_t{ei{1, 1}});
            TA::TiledRange corr{{0, 1, 2, 4}, {0, 1, 3, 9}};
            REQUIRE(r == corr);
        }
    }

    SECTION("Add two boundaries") {
        SECTION("vector") {
            auto r = insert_tile_boundaries(tr0, vector_t{ei{1}, ei{3}});
            TA::TiledRange corr{{0, 1, 2, 3, 4, 6, 8, 10}};
            REQUIRE(r == corr);
        }

        SECTION("matrix") {
            auto r = insert_tile_boundaries(tr1, vector_t{ei{1, 3}, ei{3, 2}});
            TA::TiledRange corr{{0, 1, 2, 3, 4}, {0, 2, 3, 9}};
            REQUIRE(r == corr);
        }
    }
}

TEST_CASE("insert_tile_boundaries(TiledRange, ElementIndex, ...)") {
    TA::TiledRange tr0{{0, 2, 4, 6, 8, 10}};
    TA::TiledRange tr1{{0, 2, 4}, {0, 3, 9}};

    SECTION("Add one boundary") {
        SECTION("vector") {
            auto r = insert_tile_boundaries(tr0, ei{1});
            TA::TiledRange corr{{0, 1, 2, 4, 6, 8, 10}};
            REQUIRE(r == corr);
        }

        SECTION("matrix") {
            auto r = insert_tile_boundaries(tr1, ei{1, 1});
            TA::TiledRange corr{{0, 1, 2, 4}, {0, 1, 3, 9}};
            REQUIRE(r == corr);
        }
    }

    SECTION("Add two boundaries") {
        SECTION("vector") {
            auto r = insert_tile_boundaries(tr0, ei{1}, ei{3});
            TA::TiledRange corr{{0, 1, 2, 3, 4, 6, 8, 10}};
            REQUIRE(r == corr);
        }

        SECTION("matrix") {
            auto r = insert_tile_boundaries(tr1, ei{1, 3}, ei{3, 2});
            TA::TiledRange corr{{0, 1, 2, 3, 4}, {0, 2, 3, 9}};
            REQUIRE(r == corr);
        }
    }
}
