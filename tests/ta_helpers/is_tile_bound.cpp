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

#include "tensorwrapper/ta_helpers/is_tile_bound.hpp"
#include <catch2/catch.hpp>

using namespace tensorwrapper::ta_helpers;
using tensorwrapper::sparse_map::Index;
using index_type = typename Index::index_type;

TEST_CASE("is_tile_lower_bound") {
    TA::TiledRange tr0{{0, 2, 4, 6, 8, 10}};
    TA::TiledRange tr1{{0, 2, 4}, {0, 3, 6}};

    SECTION("is a lower bound") {
        SECTION("vector tiling") {
            for(std::size_t i : {0, 2, 4, 6, 8}) {
                REQUIRE(is_tile_lower_bound(tr0, Index{i}));
            }
        }

        SECTION("matrix tiling") {
            for(std::size_t i : {0, 2})
                for(std::size_t j : {0, 3})
                    REQUIRE(is_tile_lower_bound(tr1, Index{i, j}));
        }
    }

    SECTION("is not a lower bound") {
        SECTION("vector tiling") {
            for(std::size_t i : {1, 3, 5, 7, 9, 10}) {
                REQUIRE_FALSE(is_tile_lower_bound(tr0, Index{i}));
            }
        }

        SECTION("matrix tiling") {
            SECTION("mode 0 is lower bound, mode 1 isn't") {
                for(std::size_t i : {0, 2}) {
                    for(std::size_t j : {1, 2, 4, 5, 6}) {
                        const Index eij{i, j};
                        REQUIRE_FALSE(is_tile_lower_bound(tr1, eij));
                    }
                }
            }

            SECTION("mode 0 isn't, but mode 1 is") {
                for(std::size_t i : {1, 3, 4}) {
                    for(std::size_t j : {0, 3}) {
                        const Index eij{i, j};
                        REQUIRE_FALSE(is_tile_lower_bound(tr1, eij));
                    }
                }
            }

            SECTION("neither are") {
                for(std::size_t i : {1, 3, 4}) {
                    for(std::size_t j : {1, 2, 4, 5, 6}) {
                        const Index eij{i, j};
                        REQUIRE_FALSE(is_tile_lower_bound(tr1, eij));
                    }
                }
            }
        }
    }
}

TEST_CASE("is_tile_upper_bound") {
    TA::TiledRange tr0{{0, 2, 4, 6, 8, 10}};
    TA::TiledRange tr1{{0, 2, 4}, {0, 3, 6}};

    SECTION("is a upper bound") {
        SECTION("vector tiling") {
            for(std::size_t i : {2, 4, 6, 8, 10}) {
                REQUIRE(is_tile_upper_bound(tr0, Index{i}));
            }
        }

        SECTION("matrix tiling") {
            for(std::size_t i : {2, 4})
                for(std::size_t j : {3, 6})
                    REQUIRE(is_tile_upper_bound(tr1, Index{i, j}));
        }
    }

    SECTION("is not an upper bound") {
        SECTION("vector tiling") {
            for(std::size_t i : {0, 1, 3, 5, 7, 9, 11}) {
                REQUIRE_FALSE(is_tile_upper_bound(tr0, Index{i}));
            }
        }

        SECTION("matrix tiling") {
            SECTION("mode 0 is upper bound, mode 1 isn't") {
                for(std::size_t i : {2, 4}) {
                    for(std::size_t j : {0, 1, 2, 4, 5, 7}) {
                        const Index eij{i, j};
                        REQUIRE_FALSE(is_tile_upper_bound(tr1, eij));
                    }
                }
            }

            SECTION("mode 0 isn't, but mode 1 is") {
                for(std::size_t i : {0, 1, 3, 5}) {
                    for(std::size_t j : {3, 6}) {
                        const Index eij{i, j};
                        REQUIRE_FALSE(is_tile_upper_bound(tr1, eij));
                    }
                }
            }

            SECTION("neither are") {
                for(std::size_t i : {0, 1, 3, 5}) {
                    for(std::size_t j : {0, 1, 2, 4, 5, 7}) {
                        const Index eij{i, j};
                        REQUIRE_FALSE(is_tile_upper_bound(tr1, eij));
                    }
                }
            }
        }
    }
}
