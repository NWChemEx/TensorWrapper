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

#include "tensorwrapper/sparse_map/sparse_map/sparse_map.hpp"
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

TEST_CASE("SparseMap") {
    Index i0{}, i1{1}, i12{2}, i2{1, 2}, i22{2, 3};
    Index d0{}, d1{1}, d12{2}, d2{1, 2}, d22{2, 3};

    std::map<std::string, SparseMap> sms;
    sms["Empty"];
    sms["Ind == rank 0"] = SparseMap{{i0, {d1, d12}}};
    sms["Ind == rank 1"] = SparseMap{{i12, {}}, {i1, {d1}}};
    sms["Ind == rank 2"] = SparseMap{{i2, {d2}}, {i22, {d22}}};
    SparseMap temp(std::move(sms["No PIMPL"]));

    SECTION("CTors") {
        SECTION("Typedefs") {
            SECTION("size_type") {
                using corr_t = std::size_t;
                using the_t  = SparseMap::size_type;
                STATIC_REQUIRE(std::is_same_v<corr_t, the_t>);
            }

            SECTION("key_type") {
                using corr_t = Index;
                using the_t  = SparseMap::key_type;
                STATIC_REQUIRE(std::is_same_v<corr_t, the_t>);
            }

            SECTION("mapped_type") {
                using corr_t = Domain;
                using the_t  = SparseMap::mapped_type;
                STATIC_REQUIRE(std::is_same_v<corr_t, the_t>);
            }

            SECTION("const_iterator") {
                using corr_t =
                  utilities::iterators::OffsetIterator<const SparseMap>;
                using the_t = SparseMap::const_iterator;
                STATIC_REQUIRE(std::is_same_v<corr_t, the_t>);
            }
        }

        SECTION("Default Ctor") {
            auto& sm = sms.at("Empty");
            REQUIRE(sm.size() == 0);
            REQUIRE(sm.empty());
            REQUIRE(sm.ind_rank() == 0);
            REQUIRE(sm.dep_rank() == 0);
        }

        SECTION("Initializer list") {
            SECTION("Empty") {
                SparseMap sm_empty({});
                REQUIRE(sm_empty == sms.at("Empty"));
            }

            SECTION("Ind == rank 0") {
                auto& sm0 = sms.at("Ind == rank 0");
                REQUIRE(sm0.size() == 1);
                REQUIRE_FALSE(sm0.empty());
                REQUIRE(sm0.ind_rank() == 0);
                REQUIRE(sm0.dep_rank() == 1);
            }

            SECTION("Ind == rank 1") {
                auto& sm1 = sms.at("Ind == rank 1");
                REQUIRE(sm1.size() == 1);
                REQUIRE_FALSE(sm1.empty());
                REQUIRE(sm1.ind_rank() == 1);
                REQUIRE(sm1.dep_rank() == 1);
            }

            SECTION("Ind == rank 2") {
                auto& sm2 = sms.at("Ind == rank 2");
                REQUIRE(sm2.size() == 2);
                REQUIRE_FALSE(sm2.empty());
                REQUIRE(sm2.ind_rank() == 2);
                REQUIRE(sm2.dep_rank() == 2);
            }
        }

        SECTION("Copy ctor") {
            for(auto [k, v] : sms) {
                SECTION(k) {
                    SparseMap copy(v);
                    REQUIRE(copy == v);
                }
            }
        }

        SECTION("Move ctor") {
            for(auto [k, v] : sms) {
                SECTION(k) {
                    SparseMap corr(v);
                    SparseMap moved2(std::move(v));
                    REQUIRE(moved2 == corr);
                }
            }
        }

        SECTION("Copy assignment") {
            for(auto [k, v] : sms) {
                SECTION(k) {
                    SparseMap copy;
                    auto pcopy = &(copy = v);
                    SECTION("Value") { REQUIRE(copy == v); }
                    SECTION("Returns *this") { REQUIRE(pcopy == &copy); }
                }
            }
        }

        SECTION("Move ctor") {
            for(auto [k, v] : sms) {
                SECTION(k) {
                    SparseMap corr(v);
                    SparseMap moved2;
                    auto pmoved = &(moved2 = std::move(v));
                    SECTION("Value") { REQUIRE(moved2 == corr); }
                    SECTION("Returns *this") { REQUIRE(pmoved == &moved2); }
                }
            }
        }

    } // SECTION("CTORS")

    SECTION("swap") {
        for(auto& [lhs_k, lhs_v] : sms) {
            for(auto& [rhs_k, rhs_v] : sms) {
                SECTION(lhs_k + " swapped with " + rhs_k) {
                    SparseMap corr_lhs(rhs_v);
                    SparseMap corr_rhs(lhs_v);
                    lhs_v.swap(rhs_v);
                    REQUIRE(lhs_v == corr_lhs);
                    REQUIRE(rhs_v == corr_rhs);
                }
            }
        }
    }

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

        SECTION("No PIMPL") {
            auto& mf = sms.at("No PIMPL");
            REQUIRE(mf.size() == 0);
        }
    }

    SECTION("empty") {
        for(auto [k, v] : sms) {
            SECTION(k) {
                bool is_empty = (v.size() == 0);
                REQUIRE(v.empty() == is_empty);
            }
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

        SECTION("No PIMPL") { REQUIRE_FALSE(sms.at("No PIMPL").count(i0)); }
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

        SECTION("No PIMPL") {
            auto& mf = sms.at("No PIMPL");
            REQUIRE(mf.ind_rank() == 0);
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

        SECTION("No PIMPL") {
            auto& mf = sms.at("No PIMPL");
            REQUIRE(mf.dep_rank() == 0);
        }
    }

    SECTION("add_to_domain") {
        SECTION("Empty") {
            auto& sm = sms.at("Empty");
            sm.add_to_domain(i0, d0);
            SparseMap corr{{i0, {d0}}};
            REQUIRE(sm == corr);
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
                SparseMap corr{{i0, {d1, d12, Index{3}}}};
                REQUIRE(sm0 == corr);
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
                sm1.add_to_domain(i12, d12);
                SparseMap corr{{i1, {d1}}, {i12, {d12}}};
                REQUIRE(sm1 == corr);
            }
            SECTION("Add to non-existing independent index") {
                sm1.add_to_domain(Index{4}, d12);
                SparseMap corr{{i12, {}}, {i1, {d1}}, {Index{4}, {d12}}};
                REQUIRE(sm1 == corr);
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
                SparseMap corr{{i2, {d2, Index{3, 4}}}, {i22, {d22}}};
                REQUIRE(sm2 == corr);
            }
            SECTION("Add to non-existing independent index") {
                sm2.add_to_domain(Index{3, 4}, d2);
                SparseMap corr{{i2, {d2}}, {i22, {d22}}, {Index{3, 4}, {d2}}};
                REQUIRE(sm2 == corr);
            }
        }

        SECTION("No PIMPL") {
            auto& mf = sms.at("No PIMPL");
            mf.add_to_domain(i0, d0);
            REQUIRE(mf == SparseMap{{i0, {d0}}});
        }
    }

    SECTION("operator[] const") {
        SECTION("Empty") {
            const auto& sm = sms.at("Empty");
            REQUIRE_THROWS_AS(sm[i0], std::out_of_range);
        }

        SECTION("Ind == rank 0") {
            const auto& sm = sms.at("Ind == rank 0");
            SECTION("Throws if wrong ind rank") {
                REQUIRE_THROWS_AS(sm[i1], std::runtime_error);
            }
            SECTION("Value") { REQUIRE(sm[i0] == Domain{d1, d12}); }
        }

        SECTION("Ind == rank 1") {
            const auto& sm = sms.at("Ind == rank 1");
            SECTION("Throws if wrong ind rank") {
                REQUIRE_THROWS_AS(sm[i0], std::runtime_error);
            }
            SECTION("Throws if value is not present") {
                REQUIRE_THROWS_AS(sm[Index{4}], std::out_of_range);
            }
            SECTION("Value") { REQUIRE(sm[i1] == Domain{d1}); }
        }

        SECTION("Ind == rank 2") {
            const auto& sm = sms.at("Ind == rank 2");
            SECTION("Throws if wrong ind rank") {
                REQUIRE_THROWS_AS(sm[i1], std::runtime_error);
            }
            SECTION("Throws if value is not present") {
                Index i23{3, 4};
                REQUIRE_THROWS_AS(sm[i23], std::out_of_range);
            }
            SECTION("Value") { REQUIRE(sm[i2] == Domain{d2}); }
        }

        SECTION("No PIMPL") {
            const auto& sm = sms.at("No PIMPL");
            REQUIRE_THROWS_AS(sm[i0], std::out_of_range);
        }
    } // SECTION("operator[]const")

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

        SECTION("No PIMPL") {
            const auto& sm = sms.at("No PIMPL");
            REQUIRE_THROWS_AS(sm.at(i0), std::out_of_range);
        }
    } // SECTION("operator[]const")

    SECTION("direct_product") {
        SECTION("LHS == Empty") {
            auto& lhs = sms.at("Empty");
            SparseMap corr(lhs);

            for(auto [key, rhs] : sms) {
                SECTION("RHS == " + key) {
                    auto result = lhs.direct_product(rhs);
                    REQUIRE(result == corr);
                }
            }
        }

        SECTION("LHS == rank 0") {
            auto& lhs = sms.at("Ind == rank 0");

            SECTION("RHS == Empty") {
                auto& rhs   = sms.at("Empty");
                auto result = lhs.direct_product(rhs);
                REQUIRE(result == rhs);
            }

            SECTION("RHS == Ind == rank 0") {
                auto& rhs = sms.at("Ind == rank 0");
                SparseMap corr{
                  {i0, {Index{1, 1}, Index{1, 2}, Index{2, 1}, Index{2, 2}}}};
                auto result = lhs.direct_product(rhs);
                REQUIRE(result == corr);
            }

            SECTION("RHS == Ind == rank 1") {
                auto& rhs = sms.at("Ind == rank 1");
                SparseMap corr{{i1, {Index{1, 1}, Index{2, 1}}}};
                auto result = lhs.direct_product(rhs);
                REQUIRE(result == corr);
            }

            SECTION("RHS == Ind == rank 2") {
                auto& rhs = sms.at("Ind == rank 2");
                SparseMap corr{{i2, {Index{1, 1, 2}, Index{2, 1, 2}}},
                               {i22, {Index{1, 2, 3}, Index{2, 2, 3}}}};
                auto result = lhs.direct_product(rhs);
                REQUIRE(result == corr);
            }

            SECTION("RHS == No PIMPL") {
                auto& rhs   = sms.at("No PIMPL");
                auto result = lhs.direct_product(rhs);
                REQUIRE(result == rhs);
            }
        }

        SECTION("LHS == rank 1") {
            auto& lhs = sms.at("Ind == rank 1");

            SECTION("RHS == empty") {
                auto& rhs   = sms.at("Empty");
                auto result = lhs.direct_product(rhs);
                REQUIRE(result == rhs);
            }

            SECTION("RHS == Ind == rank 0") {
                auto& rhs = sms.at("Ind == rank 0");
                SparseMap corr{{i1, {Index{1, 1}, Index{1, 2}}}};
                auto result = lhs.direct_product(rhs);
                REQUIRE(result == corr);
            }

            SECTION("RHS == Ind == rank 1") {
                auto& rhs = sms.at("Ind == rank 1");
                SparseMap corr{{Index{1, 1}, {Index{1, 1}}},
                               {Index{2, 1}, {}},
                               {Index{2, 2}, {}}};
                auto result = lhs.direct_product(rhs);
                REQUIRE(result == corr);
            }

            SECTION("RHS == Ind == rank 2") {
                auto& rhs = sms.at("Ind == rank 2");
                SparseMap corr{{Index{1, 1, 2}, {Index{1, 1, 2}}},
                               {Index{1, 2, 3}, {Index{1, 2, 3}}},
                               {Index{2, 1, 2}, {}},
                               {Index{2, 2, 3}, {}}};
                auto result = lhs.direct_product(rhs);
                REQUIRE(result == corr);
            }

            SECTION("RHS == No PIMPL") {
                auto& rhs   = sms.at("No PIMPL");
                auto result = lhs.direct_product(rhs);
                REQUIRE(result == rhs);
            }
        }

        SECTION("LHS == rank 2") {
            auto& lhs = sms.at("Ind == rank 2");

            SECTION("RHS == empty") {
                auto& rhs   = sms.at("Empty");
                auto result = lhs.direct_product(rhs);
                REQUIRE(result == rhs);
            }

            SECTION("RHS == Ind == rank 0") {
                auto& rhs = sms.at("Ind == rank 0");
                SparseMap corr{{i2, {Index{1, 2, 1}, Index{1, 2, 2}}},
                               {i22, {Index{2, 3, 1}, Index{2, 3, 2}}}};
                auto result = lhs.direct_product(rhs);
                REQUIRE(result == corr);
            }

            SECTION("RHS == Ind == rank 1") {
                auto& rhs = sms.at("Ind == rank 1");
                SparseMap corr{{Index{1, 2, 1}, {Index{1, 2, 1}}},
                               {Index{1, 2, 2}, {}},
                               {Index{2, 3, 1}, {Index{2, 3, 1}}},
                               {Index{2, 3, 2}, {}}};
                auto result = lhs.direct_product(rhs);
                REQUIRE(result == corr);
            }

            SECTION("RHS == Ind == rank 2") {
                auto& rhs = sms.at("Ind == rank 2");
                SparseMap corr{{Index{1, 2, 1, 2}, {Index{1, 2, 1, 2}}},
                               {Index{1, 2, 2, 3}, {Index{1, 2, 2, 3}}},
                               {Index{2, 3, 1, 2}, {Index{2, 3, 1, 2}}},
                               {Index{2, 3, 2, 3}, {Index{2, 3, 2, 3}}}};
                auto result = lhs.direct_product(rhs);
                REQUIRE(result == corr);
            }

            SECTION("RHS == No PIMPL") {
                auto& rhs   = sms.at("No PIMPL");
                auto result = lhs.direct_product(rhs);
                REQUIRE(result == rhs);
            }
        }

        SECTION("LHS == No PIMPL") {
            auto& lhs = sms.at("No PIMPL");
            SparseMap corr(lhs);

            for(auto [key, rhs] : sms) {
                SECTION("RHS == " + key) {
                    auto result = lhs.direct_product(rhs);
                    REQUIRE(result == corr);
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
                SparseMap rhs;

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
                SparseMap rhs{{Index{1}, {Index{2}}}};

                SECTION("lhs *= rhs") {
                    auto plhs = &(lhs *= rhs);
                    SECTION("Value") { REQUIRE(lhs == SparseMap{}); }
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
            SparseMap lhs{{Index{1}, {Index{1}}}};

            SECTION("RHS same independent, single element domain") {
                SparseMap rhs{{Index{1}, {Index{2}}}};

                SECTION("lhs *= rhs") {
                    SparseMap corr{{Index{1}, {Index{1, 2}}}};
                    auto plhs = &(lhs *= rhs);
                    SECTION("Value") { REQUIRE(lhs == corr); }
                    SECTION("Returns *this") { REQUIRE(plhs == &lhs); }
                }

                SECTION("rhs *= lhs") {
                    SparseMap corr{{Index{1}, {Index{2, 1}}}};
                    auto prhs = &(rhs *= lhs);
                    SECTION("Value") { REQUIRE(rhs == corr); }
                    SECTION("Returns *this") { REQUIRE(prhs == &rhs); }
                }
            }

            SECTION("RHS same independent, two element domain") {
                SparseMap rhs{{Index{1}, {Index{2}, Index{3}}}};

                SECTION("lhs *= rhs") {
                    SparseMap corr{{Index{1}, {Index{1, 2}, Index{1, 3}}}};
                    auto plhs = &(lhs *= rhs);
                    SECTION("Value") { REQUIRE(lhs == corr); }
                    SECTION("Returns *this") { REQUIRE(plhs == &lhs); }
                }

                SECTION("rhs *= lhs") {
                    SparseMap corr{{Index{1}, {Index{2, 1}, Index{3, 1}}}};
                    auto prhs = &(rhs *= lhs);
                    SECTION("Value") { REQUIRE(rhs == corr); }
                    SECTION("Returns *this") { REQUIRE(prhs == &rhs); }
                }
            }

            SECTION("RHS different independent, single element domain") {
                SparseMap rhs{{Index{2}, {Index{2}}}};

                SECTION("lhs *= rhs") {
                    auto plhs = &(lhs *= rhs);
                    SECTION("Value") { REQUIRE(lhs == SparseMap{}); }
                    SECTION("Returns *this") { REQUIRE(plhs == &lhs); }
                }

                SECTION("rhs *= lhs") {
                    auto prhs = &(rhs *= lhs);
                    SECTION("Value") { REQUIRE(rhs == SparseMap{}); }
                    SECTION("Returns *this") { REQUIRE(prhs == &rhs); }
                }
            }

            SECTION("RHS multiple independent") {
                SparseMap rhs{{Index{1}, {Index{2}}}, {Index{2}, {Index{2}}}};

                SECTION("lhs * rhs") {
                    auto plhs = &(lhs *= rhs);
                    SparseMap corr{{Index{1}, {Index{1, 2}}}};
                    SECTION("Value") { REQUIRE(lhs == corr); }
                    SECTION("Returns *this") { REQUIRE(plhs == &lhs); }
                }

                SECTION("rhs * lhs") {
                    auto prhs = &(rhs *= lhs);
                    SparseMap corr{{Index{1}, {Index{2, 1}}}};
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

    SECTION("operator*") {
        auto& lhs = sms.at("Ind == rank 2");
        auto& rhs = sms.at("Ind == rank 2");
        SparseMap corr{{Index{1, 2}, {Index{1, 2, 1, 2}}},
                       {Index{2, 3}, {Index{2, 3, 2, 3}}}};
        auto r = lhs * rhs;
        REQUIRE(r == corr);
    }

    /* With respect to union operator+= does all the work and operator+
     * simply calls operator+= on a copy. Therefore we test operator+= in depth
     * and make sure operator+ works for one scenario.
     */
    SECTION("operator+=") {
        SECTION("Empty / Empty") {
            SparseMap sm, sm2;
            auto psm = &(sm += sm2);
            SECTION("Value") { REQUIRE(sm == sm2); }
            SECTION("Returns *this") { REQUIRE(psm == &sm); }
        }

        SECTION("Empty / Non-empty") {
            SparseMap sm;
            SparseMap sm2{{Index{1}, {Index{0}, Index{3}}},
                          {Index{2}, {Index{1}, Index{2}}}};
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
            SparseMap sm{{Index{1}, {Index{0}, Index{3}}},
                         {Index{2}, {Index{1}, Index{2}}}};

            SECTION("Compatible") {
                SparseMap sm2{{Index{0}, {Index{0}, Index{3}}},
                              {Index{1}, {Index{1}, Index{2}}},
                              {Index{2}, {Index{1}, Index{2}}},
                              {Index{3}, {Index{1}, Index{2}}}};
                SparseMap corr{
                  {Index{0}, {Index{0}, Index{3}}},
                  {Index{1}, {Index{0}, Index{1}, Index{2}, Index{3}}},
                  {Index{2}, {Index{1}, Index{2}}},
                  {Index{3}, {Index{1}, Index{2}}}};
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
                SparseMap incompatible{{Index{1, 2}, {Index{0}, Index{3}}},
                                       {Index{2, 3}, {Index{1}, Index{2}}}};
                REQUIRE_THROWS_AS(sm += incompatible, std::runtime_error);
            }

            SECTION("Incompatible dependent indices") {
                SparseMap incompatible{{Index{1}, {Index{0, 1}, Index{3, 4}}},
                                       {Index{2}, {Index{1, 2}, Index{2, 3}}}};
                REQUIRE_THROWS_AS(sm += incompatible, std::runtime_error);
            }
        }
    }

    SECTION("operator+") {
        auto& lhs = sms.at("Empty");
        auto& rhs = sms.at("Ind == rank 0");
        auto r    = lhs + rhs;
        REQUIRE(r == rhs);
    }

    SECTION("operator^=") {
        SECTION("Empty / Empty") {
            SparseMap sm;
            auto psm = &(sm ^= sm);
            SECTION("Value") { REQUIRE(sm == SparseMap{}); }
            SECTION("Returns *this") { REQUIRE(psm == &sm); }
        }

        SECTION("Empty / Non-empty") {
            SparseMap sm;
            SparseMap sm2{{Index{1}, {Index{0}, Index{3}}},
                          {Index{2}, {Index{1}, Index{2}}}};

            SECTION("sm ^= sm2") {
                auto psm = &(sm ^= sm2);
                SECTION("Value") { REQUIRE(sm == SparseMap{}); }
                SECTION("Returns *this") { REQUIRE(psm == &sm); }
            }

            SECTION("sm2 ^= sm") {
                auto psm2 = &(sm2 ^= sm);
                SECTION("Value") { REQUIRE(sm2 == SparseMap{}); }
                SECTION("Returns *this") { REQUIRE(psm2 == &sm2); }
            }
        }

        SECTION("Non-empty / Non-empty") {
            SparseMap sm{{Index{1}, {Index{0}, Index{3}}},
                         {Index{2}, {Index{1}, Index{2}}}};
            SparseMap sm2{{Index{0}, {Index{0}, Index{3}}},
                          {Index{1}, {Index{1}, Index{2}}},
                          {Index{2}, {Index{1}, Index{2}}},
                          {Index{3}, {Index{1}, Index{2}}}};

            SparseMap corr{{Index{2}, {Index{1}, Index{2}}}};

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
                SECTION("Value") { REQUIRE(sm == SparseMap{}); }
                SECTION("Returns *this") { REQUIRE(psm == &sm); }
            }
        }
    }

    SECTION("operator^") {
        auto& sm1 = sms.at("Ind == rank 1");
        auto r    = sm1 ^ sm1;
        REQUIRE(r == sm1);
    }

    SECTION("inverse") {
        SECTION("Empty") {
            SparseMap sm;
            SparseMap corr;
            REQUIRE(sm.inverse() == corr);
        }

        SECTION("Non-empty") {
            SparseMap sm{{Index{1}, {Index{0}, Index{3}}},
                         {Index{2}, {Index{1}, Index{2}}}};
            SparseMap corr{{Index{0}, {Index{1}}},
                           {Index{3}, {Index{1}}},
                           {Index{1}, {Index{2}}},
                           {Index{2}, {Index{2}}}};
            REQUIRE(sm.inverse() == corr);
            REQUIRE(sm.inverse().inverse() == sm);
        }
    }

    SECTION("chain") {
        SparseMap lsm1{{Index{1}, {Index{0}, Index{3}}},
                       {Index{2}, {Index{1}, Index{2}}}};
        SparseMap rsm1{{Index{1}, {Index{0}, Index{3}}},
                       {Index{2}, {Index{1}, Index{2}}}};

        SECTION("Empty / Empty") {
            SparseMap sm;
            SparseMap rhs;
            SparseMap corr;
            REQUIRE(sm.chain(rhs) == corr);
        }

        SECTION("Empty / Non-empty") {
            SparseMap sm;
            REQUIRE_THROWS_AS(sm.chain(rsm1), std::runtime_error);
        }

        SECTION("Non-empty / Non-empty") {
            SparseMap rsm2{{Index{0}, {Index{0}, Index{3}}},
                           {Index{1}, {Index{1}, Index{2}}},
                           {Index{2}, {Index{1}, Index{2}}},
                           {Index{3}, {Index{1}, Index{2}}}};
            SparseMap corr{{Index{1}, {Index{0}, Index{1}, Index{2}, Index{3}}},
                           {Index{2}, {Index{1}, Index{2}}}};
            REQUIRE(lsm1.chain(rsm2) == corr);
        }

        SECTION("Non-empty / incompatible") {
            SparseMap incompatible{{Index{1, 2}, {Index{0}, Index{3}}},
                                   {Index{2, 3}, {Index{1}, Index{2}}}};
            REQUIRE_THROWS_AS(lsm1.chain(incompatible), std::runtime_error);
        }
    }

    SECTION("comparisons") {
        SECTION("Empty == Empty") {
            REQUIRE(sms.at("Empty") == SparseMap{});
            REQUIRE_FALSE(sms.at("Empty") != SparseMap{});
        }

        SECTION("Empty == No PIMPL") {
            REQUIRE(sms.at("Empty") == sms.at("No PIMPL"));
            REQUIRE_FALSE(sms.at("Empty") != sms.at("No PIMPL"));
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
            SparseMap copy(lhs);
            REQUIRE(lhs == copy);
            REQUIRE_FALSE(lhs != copy);
        }

        SECTION("Domain is subset/superset") {
            auto& lhs = sms.at("Ind == rank 0");
            SparseMap copy(lhs);
            copy.add_to_domain(i0, Index{3});
            REQUIRE_FALSE(lhs == copy);
            REQUIRE(lhs != copy);
        }

        SECTION("Different independent indices") {
            auto& lhs = sms.at("Ind == rank 1");
            SparseMap copy(lhs);
            copy.add_to_domain(Index{3}, Index{3});
            REQUIRE_FALSE(lhs == copy);
            REQUIRE(lhs != copy);
        }

        SECTION("No PIMPL != non-empty") {
            auto& lhs = sms.at("No PIMPL");
            for(std::size_t i = 0; i < 3; ++i) {
                std::string key = "Ind == rank " + std::to_string(i);
                auto& rhs       = sms.at(key);
                SECTION(key) {
                    REQUIRE_FALSE(lhs == rhs);
                    REQUIRE(lhs != rhs);
                }
            }
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
            auto h2 = hash_objects(SparseMap{});
            REQUIRE(h == h2);
        }

        SECTION("Empty == No PIMPL") {
            auto h  = hash_objects(sms.at("Empty"));
            auto h2 = hash_objects(sms.at("No PIMPL"));
#ifdef BPHASH_USE_TYPEID
            REQUIRE_FALSE(h == h2);
#else
            REQUIRE(h == h2);
#endif
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
            auto h    = hash_objects(lhs);
            auto h2   = hash_objects(SparseMap(lhs));
            REQUIRE(h == h2);
        }

        SECTION("Domain is subset/superset") {
            auto& lhs = sms.at("Ind == rank 0");
            auto h    = hash_objects(lhs);
            SparseMap copy(lhs);
            copy.add_to_domain(i0, Index{3});
            auto h2 = hash_objects(copy);
            REQUIRE(h != h2);
        }

        SECTION("Different independent indices") {
            auto& lhs = sms.at("Ind == rank 1");
            auto h    = hash_objects(lhs);
            SparseMap copy(lhs);
            copy.add_to_domain(Index{3}, Index{3});
            auto h2 = hash_objects(copy);
            REQUIRE(h != h2);
        }

        SECTION("No PIMPL != non-empty") {
            auto& lhs = sms.at("No PIMPL");
            auto h    = hash_objects(lhs);
            for(std::size_t i = 0; i < 3; ++i) {
                std::string key = "Ind == rank " + std::to_string(i);
                auto& rhs       = sms.at(key);
                auto h2         = hash_objects(rhs);
                SECTION(key) { REQUIRE(h != h2); }
            }
        }
    }
}

/* operator<< just calls SparseMap::print. So as long as that works, this will
 *  work too. We just test an empty SparseMap to make sure it gets forwarded
 *  correctly and the ostream is returend.
 */
TEST_CASE("operator<<(std::ostream, SparseMapBase)") {
    std::stringstream ss;
    SparseMap sm;
    auto pss = &(ss << sm);
    REQUIRE(pss == &ss);
    REQUIRE(ss.str() == "{}");
}
