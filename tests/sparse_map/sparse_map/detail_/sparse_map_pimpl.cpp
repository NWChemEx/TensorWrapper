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

#include "tensorwrapper/sparse_map/index.hpp"
#include "tensorwrapper/sparse_map/sparse_map/detail_/sparse_map_pimpl.hpp"
#include <catch2/catch.hpp>

using namespace tensorwrapper::sparse_map;
using namespace tensorwrapper::sparse_map::detail_;

/* General notes on testing:
 *
 * - We know that the Domain class works from unit testing it. We use a variety
 *   of Domains in these unit tests, but do not attempt to be exhaustive for
 *   that reason. For testing, what matters more is the interaction of the
 *   Domain with the SparseMap class.
 *
 */

TEST_CASE("SparseMapPIMPL") {
    Index i0{}, i1{1}, i12{2}, i2{1, 2}, i22{2, 3};
    Index d0{}, d1{1}, d12{2}, d2{1, 2}, d22{2, 3};

    std::map<std::string, SparseMapPIMPL> sms;
    sms["Empty"];
    sms["Ind == rank 0"].add_to_domain(i0, d1);
    sms["Ind == rank 0"].add_to_domain(i0, d12);
    sms["Ind == rank 1"].add_to_domain(i1, d1);
    sms["Ind == rank 2"].add_to_domain(i2, d2);
    sms["Ind == rank 2"].add_to_domain(i22, d22);

    SECTION("Typedefs") {
        SECTION("size_type") {
            using corr_t = std::size_t;
            using the_t  = SparseMapPIMPL::size_type;
            STATIC_REQUIRE(std::is_same_v<corr_t, the_t>);
        }

        SECTION("key_type") {
            using corr_t = Index;
            using the_t  = SparseMapPIMPL::key_type;
            STATIC_REQUIRE(std::is_same_v<corr_t, the_t>);
        }

        SECTION("mapped_type") {
            using corr_t = Domain;
            using the_t  = SparseMapPIMPL::mapped_type;
            STATIC_REQUIRE(std::is_same_v<corr_t, the_t>);
        }
    }

    SECTION("CTors") {
        SECTION("Default Ctor") {
            auto& sm = sms.at("Empty");
            REQUIRE(sm.size() == 0);
            REQUIRE(sm.ind_rank() == 0);
            REQUIRE(sm.dep_rank() == 0);
        }

        SECTION("Copy Ctor") {
            SparseMapPIMPL p0;
            p0.add_to_domain(i1, d1);
            SparseMapPIMPL p1(p0);
            REQUIRE(p0 == p1);
            SECTION("Is deep copy") {
                p0.add_to_domain(i12, d12);
                REQUIRE_THROWS_AS(p1.at(1), std::out_of_range);
            }
        }

        SECTION("Copy Assignment") {
            SparseMapPIMPL p0, p1;
            p0.add_to_domain(i1, d1);
            auto pp1 = &(p1 = p0);
            SECTION("Returns *this") { REQUIRE(pp1 == &p1); }
            REQUIRE(p0 == p1);
            SECTION("Is deep copy") {
                p0.add_to_domain(i12, d12);
                REQUIRE_THROWS_AS(p1.at(1), std::out_of_range);
            }
        }

        SECTION("Move Ctor") {
            SparseMapPIMPL p0, p1;
            p0.add_to_domain(i1, d1);
            p1.add_to_domain(i1, d1);
            SparseMapPIMPL p2(std::move(p0));
            REQUIRE(p1 == p2);
            REQUIRE_FALSE(p0 == p2);
        }

        SECTION("Move Assignment") {
            SparseMapPIMPL p0, p1, p2;
            p0.add_to_domain(i1, d1);
            p2.add_to_domain(i1, d1);
            auto pp1 = &(p1 = std::move(p0));
            SECTION("Returns *this") { REQUIRE(pp1 == &p1); }
            REQUIRE(p1 == p2);
            REQUIRE_FALSE(p0 == p2);
        }
    } // SECTION("CTORS")

    SECTION("size") {
        SECTION("Empty") { REQUIRE(sms.at("Empty").size() == 0); }

        SECTION("Ind == rank 0") {
            auto& sm0 = sms.at("Ind == rank 0");
            REQUIRE(sm0.size() == 1);
        }

        SECTION("Ind == rank 1") {
            auto& sm1 = sms.at("Ind == rank 1");
            REQUIRE(sm1.size() == 1);
        }

        SECTION("Ind == rank 2") {
            auto& sm2 = sms.at("Ind == rank 2");
            REQUIRE(sm2.size() == 2);
        }
    }

    SECTION("count") {
        SECTION("Empty") { REQUIRE_FALSE(sms.at("Empty").count(i0)); }

        SECTION("Ind == rank 0") {
            auto& sm0 = sms.at("Ind == rank 0");
            SECTION("Has index") { REQUIRE(sm0.count(i0)); }
            SECTION("Doesn't have") { REQUIRE_FALSE(sm0.count(i1)); }
        }

        SECTION("Ind == rank 1") {
            auto& sm1 = sms.at("Ind == rank 1");
            SECTION("Has index") { REQUIRE(sm1.count(i1)); }
            SECTION("Doesn't have") { REQUIRE_FALSE(sm1.count(i0)); }
        }

        SECTION("Ind == rank 2") {
            auto& sm2 = sms.at("Ind == rank 2");
            SECTION("Has index") { REQUIRE(sm2.count(i2)); }
            SECTION("Doesn't have") { REQUIRE_FALSE(sm2.count(i0)); }
        }
    }

    SECTION("ind_rank") {
        SECTION("Empty") { REQUIRE(sms.at("Empty").ind_rank() == 0); }

        SECTION("Ind == rank 0") {
            auto& sm0 = sms.at("Ind == rank 0");
            REQUIRE(sm0.ind_rank() == 0);
        }

        SECTION("Ind == rank 1") {
            auto& sm1 = sms.at("Ind == rank 1");
            REQUIRE(sm1.ind_rank() == 1);
        }

        SECTION("Ind == rank 2") {
            auto& sm2 = sms.at("Ind == rank 2");
            REQUIRE(sm2.ind_rank() == 2);
        }
    }

    SECTION("dep_rank") {
        SECTION("Empty") { REQUIRE(sms.at("Empty").dep_rank() == 0); }

        SECTION("Ind == rank 0") {
            auto& sm0 = sms.at("Ind == rank 0");
            REQUIRE(sm0.dep_rank() == 1);
        }

        SECTION("Ind == rank 1") {
            auto& sm1 = sms.at("Ind == rank 1");
            REQUIRE(sm1.dep_rank() == 1);
        }

        SECTION("Ind == rank 2") {
            auto& sm2 = sms.at("Ind == rank 2");
            REQUIRE(sm2.dep_rank() == 2);
        }
    }

    SECTION("add_to_domain") {
        SECTION("Empty") {
            auto& sm = sms.at("Empty");
            sm.add_to_domain(i0, d0);
            REQUIRE(sm.size() == 1);
            REQUIRE(sm.ind_rank() == 0);
            REQUIRE(sm.dep_rank() == 0);
            REQUIRE(sm.at(0).first == i0);
            REQUIRE(sm.at(0).second == Domain{d0});
        }

        SECTION("Ind == rank 0") {
            auto& sm0 = sms.at("Ind == rank 0");
            SECTION("Throws if independent rank is wrong") {
                REQUIRE_THROWS_AS(sm0.add_to_domain(i1, d1),
                                  std::runtime_error);
            }
            SECTION("Throws if dependent rank is wrong") {
                REQUIRE_THROWS_AS(sm0.add_to_domain(i0, d0),
                                  std::runtime_error);
            }
            SECTION("Add to existing independent index") {
                sm0.add_to_domain(i0, Index{3});
                REQUIRE(sm0.size() == 1);
                REQUIRE(sm0.ind_rank() == 0);
                REQUIRE(sm0.dep_rank() == 1);
                REQUIRE(sm0.at(0).first == i0);
                REQUIRE(sm0.at(0).second == Domain{d1, d12, Index{3}});
            }
        }

        SECTION("Ind == rank 1") {
            auto& sm1 = sms.at("Ind == rank 1");
            SECTION("Throws if independent rank is wrong") {
                REQUIRE_THROWS_AS(sm1.add_to_domain(i0, d1),
                                  std::runtime_error);
            }
            SECTION("Throws if dependent rank is wrong") {
                REQUIRE_THROWS_AS(sm1.add_to_domain(i1, d0),
                                  std::runtime_error);
            }
            SECTION("Add to existing independent index") {
                sm1.add_to_domain(i1, d12);
                REQUIRE(sm1.size() == 1);
                REQUIRE(sm1.ind_rank() == 1);
                REQUIRE(sm1.dep_rank() == 1);
                REQUIRE(sm1.at(0).first == i1);
                REQUIRE(sm1.at(0).second == Domain{d1, d12});
            }
            SECTION("Add to non-existing independent index") {
                sm1.add_to_domain(Index{4}, d12);
                REQUIRE(sm1.size() == 2);
                REQUIRE(sm1.ind_rank() == 1);
                REQUIRE(sm1.dep_rank() == 1);
                REQUIRE(sm1.at(0).first == i1);
                REQUIRE(sm1.at(0).second == Domain{d1});
                REQUIRE(sm1.at(1).first == Index{4});
                REQUIRE(sm1.at(1).second == Domain{d12});
            }
        }

        SECTION("Ind == rank 2") {
            auto& sm2 = sms.at("Ind == rank 2");
            SECTION("Throws if independent rank is wrong") {
                REQUIRE_THROWS_AS(sm2.add_to_domain(i0, d2),
                                  std::runtime_error);
            }
            SECTION("Throws if dependent rank is wrong") {
                REQUIRE_THROWS_AS(sm2.add_to_domain(i2, d0),
                                  std::runtime_error);
            }
            SECTION("Add to existing independent index") {
                sm2.add_to_domain(i2, Index{3, 4});
                REQUIRE(sm2.size() == 2);
                REQUIRE(sm2.ind_rank() == 2);
                REQUIRE(sm2.dep_rank() == 2);
                REQUIRE(sm2.at(0).first == i2);
                REQUIRE(sm2.at(0).second == Domain{d2, Index{3, 4}});
                REQUIRE(sm2.at(1).first == i22);
                REQUIRE(sm2.at(1).second == Domain{d22});
            }
            SECTION("Add to non-existing independent index") {
                sm2.add_to_domain(Index{3, 4}, d2);
                SparseMapPIMPL corr;
                corr.add_to_domain(i2, d2);
                corr.add_to_domain(i22, d22);
                corr.add_to_domain(Index{3, 4}, d2);
                REQUIRE(sm2.size() == 3);
                REQUIRE(sm2.ind_rank() == 2);
                REQUIRE(sm2.dep_rank() == 2);
                REQUIRE(sm2.at(0).first == i2);
                REQUIRE(sm2.at(0).second == Domain{d2});
                REQUIRE(sm2.at(1).first == i22);
                REQUIRE(sm2.at(1).second == Domain{d22});
                REQUIRE(sm2.at(2).first == Index{3, 4});
                REQUIRE(sm2.at(2).second == Domain{d2});
            }
        }
    }

    SECTION("at() const") {
        SECTION("Empty") {
            const auto& sm = sms.at("Empty");
            REQUIRE_THROWS_AS(sm.at(i0), std::out_of_range);
        }

        SECTION("Ind == rank 0") {
            const auto& sm = sms.at("Ind == rank 0");
            SECTION("Throws if wrong ind rank") {
                REQUIRE_THROWS_AS(sm.at(i1), std::runtime_error);
            }
            SECTION("Value") { REQUIRE(sm.at(i0) == Domain{d1, d12}); }
        }

        SECTION("Ind == rank 1") {
            const auto& sm = sms.at("Ind == rank 1");
            SECTION("Throws if wrong ind rank") {
                REQUIRE_THROWS_AS(sm.at(i0), std::runtime_error);
            }
            SECTION("Throws if value is not present") {
                REQUIRE_THROWS_AS(sm.at(Index{4}), std::out_of_range);
            }
            SECTION("Value") { REQUIRE(sm.at(i1) == Domain{d1}); }
        }

        SECTION("Ind == rank 2") {
            const auto& sm = sms.at("Ind == rank 2");
            SECTION("Throws if wrong ind rank") {
                REQUIRE_THROWS_AS(sm.at(i1), std::runtime_error);
            }
            SECTION("Throws if value is not present") {
                Index i23{3, 4};
                REQUIRE_THROWS_AS(sm.at(i23), std::out_of_range);
            }
            SECTION("Value") { REQUIRE(sm.at(i2) == Domain{d2}); }
        }

    } // SECTION("at() const")

    SECTION("direct_product_assign") {
        SECTION("LHS == Empty") {
            auto& lhs = sms.at("Empty");
            SparseMapPIMPL corr;

            for(auto& [key, rhs] : sms) {
                SECTION("RHS == " + key) {
                    auto plhs = &(lhs.direct_product_assign(rhs));
                    SECTION("returns *this") { REQUIRE(plhs == &lhs); }
                    SECTION("Value") { REQUIRE(lhs == corr); }
                }
            }
        }

        SECTION("LHS == rank 0") {
            auto& lhs = sms.at("Ind == rank 0");

            SECTION("RHS == Empty") {
                auto& rhs = sms.at("Empty");
                auto plhs = &(lhs.direct_product_assign(rhs));
                SECTION("Returns *this") { REQUIRE(plhs == &lhs); }
                SECTION("Value") { REQUIRE(lhs == rhs); }
            }

            SECTION("RHS == Ind == rank 0") {
                auto& rhs = sms.at("Ind == rank 0");
                auto plhs = &(lhs.direct_product_assign(rhs));
                SECTION("Returns *this") { REQUIRE(plhs == &lhs); }
                SECTION("Value") {
                    SparseMapPIMPL corr;
                    corr.add_to_domain(i0, Index{1, 1});
                    corr.add_to_domain(i0, Index{1, 2});
                    corr.add_to_domain(i0, Index{2, 1});
                    corr.add_to_domain(i0, Index{2, 2});
                    REQUIRE(lhs == corr);
                }
            }

            SECTION("RHS == Ind == rank 1") {
                auto& rhs = sms.at("Ind == rank 1");
                auto plhs = &(lhs.direct_product_assign(rhs));
                SECTION("returns *this") { REQUIRE(plhs == &lhs); }
                SECTION("value") {
                    SparseMapPIMPL corr;
                    corr.add_to_domain(i1, Index{1, 1});
                    corr.add_to_domain(i1, Index{2, 1});
                    REQUIRE(lhs == corr);
                }
            }

            SECTION("RHS == Ind == rank 2") {
                auto& rhs = sms.at("Ind == rank 2");
                auto plhs = &(lhs.direct_product_assign(rhs));
                SECTION("returns *this") { REQUIRE(plhs == &lhs); }
                SECTION("value") {
                    SparseMapPIMPL corr;
                    corr.add_to_domain(i2, Index{1, 1, 2});
                    corr.add_to_domain(i2, Index{2, 1, 2});
                    corr.add_to_domain(i22, Index{1, 2, 3});
                    corr.add_to_domain(i22, Index{2, 2, 3});
                    REQUIRE(lhs == corr);
                }
            }
        }

        SECTION("LHS == rank 1") {
            auto& lhs = sms.at("Ind == rank 1");

            SECTION("RHS == empty") {
                auto& rhs = sms.at("Empty");
                auto plhs = &(lhs.direct_product_assign(rhs));
                SECTION("returns *this") { REQUIRE(plhs == &lhs); }
                SECTION("value") { REQUIRE(lhs == rhs); }
            }

            SECTION("RHS == Ind == rank 0") {
                auto& rhs = sms.at("Ind == rank 0");
                auto plhs = &(lhs.direct_product_assign(rhs));
                SECTION("returns *this") { REQUIRE(plhs == &lhs); }
                SECTION("value") {
                    SparseMapPIMPL corr;
                    corr.add_to_domain(i1, Index{1, 1});
                    corr.add_to_domain(i1, Index{1, 2});
                    REQUIRE(lhs == corr);
                }
            }

            SECTION("RHS == Ind == rank 1") {
                auto& rhs = sms.at("Ind == rank 1");
                auto plhs = &(lhs.direct_product_assign(rhs));
                SECTION("returns *this") { REQUIRE(plhs == &lhs); }
                SECTION("value") {
                    SparseMapPIMPL corr;
                    corr.add_to_domain(Index{1, 1}, Index{1, 1});
                    REQUIRE(lhs == corr);
                }
            }

            SECTION("RHS == Ind == rank 2") {
                auto& rhs = sms.at("Ind == rank 2");
                auto plhs = &(lhs.direct_product_assign(rhs));
                SECTION("returns *this") { REQUIRE(plhs == &lhs); }
                SECTION("value") {
                    SparseMapPIMPL corr;
                    corr.add_to_domain(Index{1, 1, 2}, Index{1, 1, 2});
                    corr.add_to_domain(Index{1, 2, 3}, Index{1, 2, 3});
                    REQUIRE(lhs == corr);
                }
            }
        }

        SECTION("LHS == rank 2") {
            auto& lhs = sms.at("Ind == rank 2");

            SECTION("RHS == empty") {
                auto& rhs = sms.at("Empty");
                auto plhs = &(lhs.direct_product_assign(rhs));
                SECTION("returns *this") { REQUIRE(plhs == &lhs); }
                SECTION("value") { REQUIRE(lhs == rhs); }
            }

            SECTION("RHS == Ind == rank 0") {
                auto& rhs = sms.at("Ind == rank 0");
                auto plhs = &(lhs.direct_product_assign(rhs));
                SECTION("returns *this") { REQUIRE(plhs == &lhs); }
                SECTION("value") {
                    SparseMapPIMPL corr;
                    corr.add_to_domain(i2, Index{1, 2, 1});
                    corr.add_to_domain(i2, Index{1, 2, 2});
                    corr.add_to_domain(i22, Index{2, 3, 1});
                    corr.add_to_domain(i22, Index{2, 3, 2});
                    REQUIRE(lhs == corr);
                }
            }

            SECTION("RHS == Ind == rank 1") {
                auto& rhs = sms.at("Ind == rank 1");
                auto plhs = &(lhs.direct_product_assign(rhs));
                SECTION("returns *this") { REQUIRE(plhs == &lhs); }
                SECTION("value") {
                    SparseMapPIMPL corr;
                    corr.add_to_domain(Index{1, 2, 1}, Index{1, 2, 1});
                    corr.add_to_domain(Index{2, 3, 1}, Index{2, 3, 1});
                    REQUIRE(lhs == corr);
                }
            }

            SECTION("RHS == Ind == rank 2") {
                auto& rhs = sms.at("Ind == rank 2");
                auto plhs = &(lhs.direct_product_assign(rhs));
                SECTION("returns *this") { REQUIRE(plhs == &lhs); }
                SECTION("value") {
                    SparseMapPIMPL corr;
                    corr.add_to_domain(Index{1, 2, 1, 2}, Index{1, 2, 1, 2});
                    corr.add_to_domain(Index{1, 2, 2, 3}, Index{1, 2, 2, 3});
                    corr.add_to_domain(Index{2, 3, 1, 2}, Index{2, 3, 1, 2});
                    corr.add_to_domain(Index{2, 3, 2, 3}, Index{2, 3, 2, 3});
                    REQUIRE(lhs == corr);
                }
            }
        }
    }

    /* With respect to direct product operator*= does all the work and
     * operator* simply calls operator*= on a copy. Therefore we test
     * operator*= in depth and simply make sure operator* works for one
     * scenario.
     */
    SECTION("operator*=") {
        SECTION("LHS == empty") {
            auto& lhs = sms.at("Empty");

            SECTION("RHS == empty") {
                SparseMapPIMPL rhs;

                SECTION("lhs *= rhs") {
                    auto plhs = &(lhs *= rhs);
                    SECTION("Value") { REQUIRE(lhs == rhs); }
                    SECTION("Returns *this") { REQUIRE(plhs == &lhs); }
                }

                SECTION("rhs *= lhs") {
                    auto prhs = &(rhs *= lhs);
                    SECTION("Value") { REQUIRE(rhs == lhs); }
                    SECTION("Returns *this") { REQUIRE(prhs == &rhs); }
                }
            }

            SECTION("RHS == non-empty") {
                SparseMapPIMPL rhs;
                rhs.add_to_domain(Index{1}, Index{2});

                SECTION("lhs *= rhs") {
                    auto plhs = &(lhs *= rhs);
                    SECTION("Value") { REQUIRE(lhs == SparseMapPIMPL{}); }
                    SECTION("Returns *this") { REQUIRE(plhs == &lhs); }
                }

                SECTION("rhs *= lhs") {
                    auto prhs = &(rhs *= lhs);
                    SECTION("Value") { REQUIRE(rhs == lhs); }
                    SECTION("Returns *this") { REQUIRE(prhs == &rhs); }
                }
            }
        }

        SECTION("LHS == non-empty") {
            SparseMapPIMPL lhs;
            lhs.add_to_domain(Index{1}, Index{1});

            SECTION("RHS same independent, single element domain") {
                SparseMapPIMPL rhs;
                rhs.add_to_domain(Index{1}, Index{2});

                SECTION("lhs *= rhs") {
                    SparseMapPIMPL corr;
                    corr.add_to_domain(Index{1}, Index{1, 2});
                    auto plhs = &(lhs *= rhs);
                    SECTION("Value") { REQUIRE(lhs == corr); }
                    SECTION("Returns *this") { REQUIRE(plhs == &lhs); }
                }

                SECTION("rhs *= lhs") {
                    SparseMapPIMPL corr;
                    corr.add_to_domain(Index{1}, Index{2, 1});
                    auto prhs = &(rhs *= lhs);
                    SECTION("Value") { REQUIRE(rhs == corr); }
                    SECTION("Returns *this") { REQUIRE(prhs == &rhs); }
                }
            }

            SECTION("RHS same independent, two element domain") {
                SparseMapPIMPL rhs;
                rhs.add_to_domain(Index{1}, Index{2});
                rhs.add_to_domain(Index{1}, Index{3});

                SECTION("lhs *= rhs") {
                    SparseMapPIMPL corr;
                    corr.add_to_domain(Index{1}, Index{1, 2});
                    corr.add_to_domain(Index{1}, Index{1, 3});
                    auto plhs = &(lhs *= rhs);
                    SECTION("Value") { REQUIRE(lhs == corr); }
                    SECTION("Returns *this") { REQUIRE(plhs == &lhs); }
                }

                SECTION("rhs *= lhs") {
                    SparseMapPIMPL corr;
                    corr.add_to_domain(Index{1}, Index{2, 1});
                    corr.add_to_domain(Index{1}, Index{3, 1});
                    auto prhs = &(rhs *= lhs);
                    SECTION("Value") { REQUIRE(rhs == corr); }
                    SECTION("Returns *this") { REQUIRE(prhs == &rhs); }
                }
            }

            SECTION("RHS different independent, single element domain") {
                SparseMapPIMPL rhs;
                rhs.add_to_domain(Index{2}, Index{2});

                SECTION("lhs *= rhs") {
                    auto plhs = &(lhs *= rhs);
                    SECTION("Value") { REQUIRE(lhs == SparseMapPIMPL{}); }
                    SECTION("Returns *this") { REQUIRE(plhs == &lhs); }
                }

                SECTION("rhs *= lhs") {
                    auto prhs = &(rhs *= lhs);
                    SECTION("Value") { REQUIRE(rhs == SparseMapPIMPL{}); }
                    SECTION("Returns *this") { REQUIRE(prhs == &rhs); }
                }
            }

            SECTION("RHS multiple independent") {
                SparseMapPIMPL rhs;
                rhs.add_to_domain(Index{1}, Index{2});
                rhs.add_to_domain(Index{2}, Index{2});

                SECTION("lhs * rhs") {
                    auto plhs = &(lhs *= rhs);
                    SparseMapPIMPL corr;
                    corr.add_to_domain(Index{1}, Index{1, 2});
                    SECTION("Value") { REQUIRE(lhs == corr); }
                    SECTION("Returns *this") { REQUIRE(plhs == &lhs); }
                }

                SECTION("rhs * lhs") {
                    auto prhs = &(rhs *= lhs);
                    SparseMapPIMPL corr;
                    corr.add_to_domain(Index{1}, Index{2, 1});
                    SECTION("Value") { REQUIRE(rhs == corr); }
                    SECTION("Returns *this") { REQUIRE(prhs == &rhs); }
                }
            }
        }

        SECTION("Incompatible independent indices") {
            auto& lhs = sms.at("Ind == rank 1");
            auto& rhs = sms.at("Ind == rank 2");
            REQUIRE_THROWS_AS(lhs *= rhs, std::runtime_error);
        }
    }

    /* With respect to union operator+= does all the work and operator+
     * simply calls operator+= on a copy. Therefore we test operator+= in depth
     * and make sure operator+ works for one scenario.
     */
    SECTION("operator+=") {
        SECTION("Empty / Empty") {
            SparseMapPIMPL sm, sm2;
            auto psm = &(sm += sm2);
            SECTION("Value") { REQUIRE(sm == sm2); }
            SECTION("Returns *this") { REQUIRE(psm == &sm); }
        }

        SECTION("Empty / Non-empty") {
            SparseMapPIMPL sm;
            SparseMapPIMPL sm2;
            sm2.add_to_domain(Index{1}, Index{0});
            sm2.add_to_domain(Index{1}, Index{3});
            sm2.add_to_domain(Index{2}, Index{1});
            sm2.add_to_domain(Index{2}, Index{2});
            SECTION("sm += sm2") {
                auto psm = &(sm += sm2);
                REQUIRE(sm == sm2);
                SECTION("Returns *this") { REQUIRE(psm == &sm); }
            }

            SECTION("sm += sm2") {
                auto psm = &(sm += sm2);
                REQUIRE(sm == sm2);
                SECTION("Returns *this") { REQUIRE(psm == &sm); }
            }
        }

        SECTION("Non-empty / Non-empty") {
            SparseMapPIMPL sm;
            sm.add_to_domain(Index{1}, Index{0});
            sm.add_to_domain(Index{1}, Index{3});
            sm.add_to_domain(Index{2}, Index{1});
            sm.add_to_domain(Index{2}, Index{2});

            SECTION("Compatible") {
                SparseMapPIMPL sm2;
                sm2.add_to_domain(Index{0}, Index{0});
                sm2.add_to_domain(Index{0}, Index{3});
                sm2.add_to_domain(Index{1}, Index{1});
                sm2.add_to_domain(Index{1}, Index{2});
                sm2.add_to_domain(Index{2}, Index{1});
                sm2.add_to_domain(Index{2}, Index{2});
                sm2.add_to_domain(Index{3}, Index{1});
                sm2.add_to_domain(Index{3}, Index{2});

                SparseMapPIMPL corr;
                corr.add_to_domain(Index{0}, Index{0});
                corr.add_to_domain(Index{0}, Index{3});
                corr.add_to_domain(Index{1}, Index{0});
                corr.add_to_domain(Index{1}, Index{1});
                corr.add_to_domain(Index{1}, Index{2});
                corr.add_to_domain(Index{1}, Index{3});
                corr.add_to_domain(Index{2}, Index{1});
                corr.add_to_domain(Index{2}, Index{2});
                corr.add_to_domain(Index{3}, Index{1});
                corr.add_to_domain(Index{3}, Index{2});
                SECTION("sm += sm2") {
                    auto psm = &(sm += sm2);
                    SECTION("Value") { REQUIRE(sm == corr); }
                    SECTION("Returns *this") { REQUIRE(psm == &sm); }
                }
                SECTION("sm += sm2") {
                    auto psm2 = &(sm2 += sm);
                    SECTION("Value") { REQUIRE(sm2 == corr); }
                    SECTION("Returns *this") { REQUIRE(psm2 == &sm2); }
                }
                SECTION("sm += corr") {
                    auto psm = &(sm += corr);
                    SECTION("Value") { REQUIRE(sm == corr); }
                    SECTION("Returns *this") { REQUIRE(psm == &sm); }
                }
            }

            SECTION("Incompatible independent indices") {
                SparseMapPIMPL incompatible;
                incompatible.add_to_domain(Index{1, 2}, Index{0});
                incompatible.add_to_domain(Index{1, 2}, Index{3});
                incompatible.add_to_domain(Index{2, 3}, Index{1});
                incompatible.add_to_domain(Index{2, 3}, Index{2});
                REQUIRE_THROWS_AS(sm += incompatible, std::runtime_error);
            }

            SECTION("Incompatible dependent indices") {
                SparseMapPIMPL incompatible;
                incompatible.add_to_domain(Index{1}, Index{0, 1});
                incompatible.add_to_domain(Index{1}, Index{3, 4});
                incompatible.add_to_domain(Index{2}, Index{1, 2});
                incompatible.add_to_domain(Index{2}, Index{2, 3});
                REQUIRE_THROWS_AS(sm += incompatible, std::runtime_error);
            }
        }
    }

    SECTION("operator^=") {
        SECTION("Empty / Empty") {
            SparseMapPIMPL sm;
            auto psm = &(sm ^= sm);
            SECTION("Value") { REQUIRE(sm == SparseMapPIMPL{}); }
            SECTION("Returns *this") { REQUIRE(psm == &sm); }
        }

        SECTION("Empty / Non-empty") {
            SparseMapPIMPL sm;
            SparseMapPIMPL sm2;
            sm2.add_to_domain(Index{1}, Index{0});
            sm2.add_to_domain(Index{1}, Index{3});
            sm2.add_to_domain(Index{2}, Index{1});
            sm2.add_to_domain(Index{2}, Index{2});

            SECTION("sm ^= sm2") {
                auto psm = &(sm ^= sm2);
                SECTION("Value") { REQUIRE(sm == SparseMapPIMPL{}); }
                SECTION("Returns *this") { REQUIRE(psm == &sm); }
            }

            SECTION("sm2 ^= sm") {
                auto psm2 = &(sm2 ^= sm);
                SECTION("Value") { REQUIRE(sm2 == SparseMapPIMPL{}); }
                SECTION("Returns *this") { REQUIRE(psm2 == &sm2); }
            }
        }

        SECTION("Non-empty / Non-empty") {
            SparseMapPIMPL sm;
            sm.add_to_domain(Index{1}, Index{0});
            sm.add_to_domain(Index{1}, Index{3});
            sm.add_to_domain(Index{2}, Index{1});
            sm.add_to_domain(Index{2}, Index{2});
            SparseMapPIMPL sm2;
            sm2.add_to_domain(Index{0}, Index{0});
            sm2.add_to_domain(Index{0}, Index{3});
            sm2.add_to_domain(Index{1}, Index{1});
            sm2.add_to_domain(Index{1}, Index{2});
            sm2.add_to_domain(Index{2}, Index{1});
            sm2.add_to_domain(Index{2}, Index{2});
            sm2.add_to_domain(Index{3}, Index{1});
            sm2.add_to_domain(Index{3}, Index{2});

            SparseMapPIMPL corr;
            corr.add_to_domain(Index{2}, Index{1});
            corr.add_to_domain(Index{2}, Index{2});

            SECTION("sm ^= sm2") {
                auto psm = &(sm ^= sm2);
                SECTION("Value") { REQUIRE(sm == corr); }
                SECTION("Returns *this") { REQUIRE(psm == &sm); }
            }

            SECTION("sm2 ^= sm") {
                auto psm2 = &(sm2 ^= sm);
                SECTION("Value") { REQUIRE(sm2 == corr); }
                SECTION("Returns *this") { REQUIRE(psm2 == &sm2); }
            }

            SECTION("sm ^= corr") {
                auto psm = &(sm ^= corr);
                SECTION("Value") { REQUIRE(sm == corr); }
                SECTION("Returns *this") { REQUIRE(psm == &sm); }
            }

            SECTION("different ranks") {
                auto psm = &(sm ^= sms.at("Ind == rank 2"));
                SECTION("Value") { REQUIRE(sm == SparseMapPIMPL{}); }
                SECTION("Returns *this") { REQUIRE(psm == &sm); }
            }
        }
    }

    SECTION("comparisons") {
        SECTION("Empty == Empty") {
            REQUIRE(sms.at("Empty") == SparseMapPIMPL{});
            REQUIRE_FALSE(sms.at("Empty") != SparseMapPIMPL{});
        }

        SECTION("Empty != non-empty") {
            auto& lhs = sms.at("Empty");
            for(std::size_t i = 0; i < 3; ++i) {
                std::string key = "Ind == rank " + std::to_string(i);
                auto& rhs       = sms.at(key);
                SECTION(key) {
                    REQUIRE_FALSE(lhs == rhs);
                    REQUIRE(lhs != rhs);
                }
            }
        }

        SECTION("Same non-empty") {
            auto& lhs = sms.at("Ind == rank 0");
            SparseMapPIMPL copy(lhs);
            REQUIRE(lhs == copy);
            REQUIRE_FALSE(lhs != copy);
        }

        SECTION("Domain is subset/superset") {
            auto& lhs = sms.at("Ind == rank 0");
            SparseMapPIMPL copy(lhs);
            copy.add_to_domain(i0, Index{3});
            REQUIRE_FALSE(lhs == copy);
            REQUIRE(lhs != copy);
        }

        SECTION("Different independent indices") {
            auto& lhs = sms.at("Ind == rank 1");
            SparseMapPIMPL copy(lhs);
            copy.add_to_domain(Index{3}, Index{3});
            REQUIRE_FALSE(lhs == copy);
            REQUIRE(lhs != copy);
        }
    }

    SECTION("print") {
        std::stringstream ss;

        SECTION("empty") {
            auto pss = &(sms.at("Empty").print(ss));
            SECTION("Value") { REQUIRE(ss.str() == "{}"); }
            SECTION("Returns ostream") { REQUIRE(pss == &ss); }
        }

        SECTION("Non-empty") {
            auto pss = &(sms.at("Ind == rank 0").print(ss));
            SECTION("Value") {
                std::string corr = "{({} : {{1}, {2}})}";
                REQUIRE(ss.str() == corr);
            }
            SECTION("Returns ostream") { REQUIRE(pss == &ss); }
        }
    }

    SECTION("hash") {
        using tensorwrapper::detail_::hash_objects;
        SECTION("Empty == Empty") {
            auto h  = hash_objects(sms.at("Empty"));
            auto h2 = hash_objects(SparseMapPIMPL{});
            REQUIRE(h == h2);
        }

        SECTION("Empty != non-empty") {
            auto h = hash_objects(sms.at("Empty"));
            for(std::size_t i = 0; i < 3; ++i) {
                std::string key = "Ind == rank " + std::to_string(i);
                auto& rhs       = sms.at(key);
                SECTION(key) {
                    auto h2 = hash_objects(rhs);
                    REQUIRE(h != h2);
                }
            }
        }

        SECTION("Same non-empty") {
            auto& lhs = sms.at("Ind == rank 0");
            SparseMapPIMPL copy(lhs);
            auto h  = hash_objects(lhs);
            auto h2 = hash_objects(copy);
            REQUIRE(h == h2);
        }

        SECTION("Domain is subset/superset") {
            auto& lhs = sms.at("Ind == rank 0");
            auto h    = hash_objects(lhs);
            SparseMapPIMPL copy(lhs);
            copy.add_to_domain(i0, Index{3});
            auto h2 = hash_objects(copy);
            REQUIRE(h != h2);
        }

        SECTION("Different independent indices") {
            auto& lhs = sms.at("Ind == rank 1");
            auto h    = hash_objects(lhs);
            SparseMapPIMPL copy(lhs);
            copy.add_to_domain(Index{3}, Index{3});
            auto h2 = hash_objects(copy);
            REQUIRE(h != h2);
        }
    }
}

/* operator<< just calls SparseMap::print. So as long as that works, this will
 *  work too. We just test an empty SparseMap to make sure it gets forwarded
 *  correctly and the ostream is returend.
 */
TEST_CASE("operator<<(std::ostream, SparseMapPIMPL)") {
    std::stringstream ss;
    SparseMapPIMPL sm;
    auto pss = &(ss << sm);
    REQUIRE(pss == &ss);
    REQUIRE(ss.str() == "{}");
}
