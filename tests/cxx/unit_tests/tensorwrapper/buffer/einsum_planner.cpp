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
#include <tensorwrapper/buffer/einsum_planner.hpp>

using namespace tensorwrapper;
using namespace buffer;

/*
 * Let "t" stand for a set of trace indices, "f" for a set of free indices,
 * "d" for a set of dummy indices, and "b" for a set of batch indices. Then any
 * given label can be described as a combination of these four categories. In
 * the event that a label is empty we label it "s" for scalar.
 *
 * For the tensor operation A = B * C the possible categorization of the labels
 * for A, B, and C can respectively be:
 *  - s s s
 *  - s s t
 *  - s t s
 *  - s t t
 *  - s d d
 *  - s d dt
 *  - s dt d
 *  - s dt dt
 *  - f t f
 *  - f f t
 *  - f f f
 *  - f t ft
 *  - f ft t
 *  - f ft ft
 *  - f d df
 *  - f df d
 *  - f df df
 *  - f dt df
 *  - f df dt
 *  - f df df
 *  - f dt dft
 *  - f dft dt
 *  - f dft dft
 *  - bf bt bf
 *  - bf bf bt
 *  - bf bf bf
 *  - bf bd bdf
 *  - bf bdf bd
 *  - bf bdf bdf
 *  - bf bt bft
 *  - bf bft bt
 *  - bf bft bft
 *  - bf bdt bdf
 *  - bf bdf bdt
 *  - bf bdf bdf
 *  - bf bdt bdft
 *  - bf bdft bdt
 *  - bf bdft bdft
 *
 *  (these enumerations ignore permuting the categories within a label)
 *
 *  The following are NOT possible:
 *
 *  - labels that are scalar and something else (e.g., trace). Scalar is by
 *    definition the lack of the four index categories.
 *  - trace in the result (would require result to have a mode that is
 *    independent of the inputs)
 *  - dummy in the result (dummy can only appear in the inputs)
 *  - dummy in only one of the inputs
 *  - free indices when the result is a scalar
 *  - batch indices when the result is a scalar
 *
 */

TEST_CASE("EinsumPlanner") {
    SECTION("Result in scalars") {
        SECTION("s s s") {
            EinsumPlanner ep___("", "", "");
            REQUIRE(ep___.lhs_trace() == "");
            REQUIRE(ep___.rhs_trace() == "");
            REQUIRE(ep___.lhs_dummy() == "");
            REQUIRE(ep___.rhs_dummy() == "");
            REQUIRE(ep___.lhs_free() == "");
            REQUIRE(ep___.rhs_free() == "");
            REQUIRE(ep___.result_batch() == "");
            REQUIRE(ep___.lhs_batch() == "");
            REQUIRE(ep___.rhs_batch() == "");
        }
        SECTION("s s t") {
            EinsumPlanner ep___kl("", "", "k,l");
            REQUIRE(ep___kl.lhs_trace() == "");
            REQUIRE(ep___kl.rhs_trace() == "k,l");
            REQUIRE(ep___kl.lhs_dummy() == "");
            REQUIRE(ep___kl.rhs_dummy() == "");
            REQUIRE(ep___kl.lhs_free() == "");
            REQUIRE(ep___kl.rhs_free() == "");
            REQUIRE(ep___kl.result_batch() == "");
            REQUIRE(ep___kl.lhs_batch() == "");
            REQUIRE(ep___kl.rhs_batch() == "");
        }
        SECTION("s t s") {
            EinsumPlanner ep__ij_("", "i,j", "");
            REQUIRE(ep__ij_.lhs_trace() == "i,j");
            REQUIRE(ep__ij_.rhs_trace() == "");
            REQUIRE(ep__ij_.lhs_dummy() == "");
            REQUIRE(ep__ij_.rhs_dummy() == "");
            REQUIRE(ep__ij_.lhs_free() == "");
            REQUIRE(ep__ij_.rhs_free() == "");
            REQUIRE(ep__ij_.result_batch() == "");
            REQUIRE(ep__ij_.lhs_batch() == "");
            REQUIRE(ep__ij_.rhs_batch() == "");
        }
        SECTION("s t t") {
            EinsumPlanner ep__ij_klm("", "i,j", "k,l,m");
            REQUIRE(ep__ij_klm.lhs_trace() == "i,j");
            REQUIRE(ep__ij_klm.rhs_trace() == "k,l,m");
            REQUIRE(ep__ij_klm.lhs_dummy() == "");
            REQUIRE(ep__ij_klm.rhs_dummy() == "");
            REQUIRE(ep__ij_klm.lhs_free() == "");
            REQUIRE(ep__ij_klm.rhs_free() == "");
            REQUIRE(ep__ij_klm.result_batch() == "");
            REQUIRE(ep__ij_klm.lhs_batch() == "");
            REQUIRE(ep__ij_klm.rhs_batch() == "");
        }
        SECTION("s d d") {
            EinsumPlanner ep__ij_ji("", "i,j", "j,i");
            REQUIRE(ep__ij_ji.lhs_trace() == "");
            REQUIRE(ep__ij_ji.rhs_trace() == "");
            REQUIRE(ep__ij_ji.lhs_dummy() == "i,j");
            REQUIRE(ep__ij_ji.rhs_dummy() == "j,i");
            REQUIRE(ep__ij_ji.lhs_free() == "");
            REQUIRE(ep__ij_ji.rhs_free() == "");
            REQUIRE(ep__ij_ji.result_batch() == "");
            REQUIRE(ep__ij_ji.lhs_batch() == "");
            REQUIRE(ep__ij_ji.rhs_batch() == "");
        }
        SECTION("s d dt") {
            EinsumPlanner ep__ij_jik("", "i,j", "j,i,k");
            REQUIRE(ep__ij_jik.lhs_trace() == "");
            REQUIRE(ep__ij_jik.rhs_trace() == "k");
            REQUIRE(ep__ij_jik.lhs_dummy() == "i,j");
            REQUIRE(ep__ij_jik.rhs_dummy() == "j,i");
            REQUIRE(ep__ij_jik.lhs_free() == "");
            REQUIRE(ep__ij_jik.rhs_free() == "");
            REQUIRE(ep__ij_jik.result_batch() == "");
            REQUIRE(ep__ij_jik.lhs_batch() == "");
            REQUIRE(ep__ij_jik.rhs_batch() == "");
        }
        SECTION("s dt d") {
            EinsumPlanner ep__jik_ik("", "j,i,k", "i,k");
            REQUIRE(ep__jik_ik.lhs_trace() == "j");
            REQUIRE(ep__jik_ik.rhs_trace() == "");
            REQUIRE(ep__jik_ik.lhs_dummy() == "i,k");
            REQUIRE(ep__jik_ik.rhs_dummy() == "i,k");
            REQUIRE(ep__jik_ik.lhs_free() == "");
            REQUIRE(ep__jik_ik.rhs_free() == "");
            REQUIRE(ep__jik_ik.result_batch() == "");
            REQUIRE(ep__jik_ik.lhs_batch() == "");
            REQUIRE(ep__jik_ik.rhs_batch() == "");
        }
        SECTION("s dt dt") {
            EinsumPlanner ep__jik_ikm("", "j,i,k", "i,k,m");
            REQUIRE(ep__jik_ikm.lhs_trace() == "j");
            REQUIRE(ep__jik_ikm.rhs_trace() == "m");
            REQUIRE(ep__jik_ikm.lhs_dummy() == "i,k");
            REQUIRE(ep__jik_ikm.rhs_dummy() == "i,k");
            REQUIRE(ep__jik_ikm.lhs_free() == "");
            REQUIRE(ep__jik_ikm.rhs_free() == "");
            REQUIRE(ep__jik_ikm.result_batch() == "");
            REQUIRE(ep__jik_ikm.lhs_batch() == "");
            REQUIRE(ep__jik_ikm.rhs_batch() == "");
        }
    }

    SECTION("Result in free indices") {
        SECTION("f t f") {
            EinsumPlanner ep_ik_jl_ik("i,k", "j,l", "i,k");
            REQUIRE(ep_ik_jl_ik.lhs_trace() == "j,l");
            REQUIRE(ep_ik_jl_ik.rhs_trace() == "");
            REQUIRE(ep_ik_jl_ik.lhs_dummy() == "");
            REQUIRE(ep_ik_jl_ik.rhs_dummy() == "");
            REQUIRE(ep_ik_jl_ik.lhs_free() == "");
            REQUIRE(ep_ik_jl_ik.rhs_free() == "i,k");
            REQUIRE(ep_ik_jl_ik.result_batch() == "");
            REQUIRE(ep_ik_jl_ik.lhs_batch() == "");
            REQUIRE(ep_ik_jl_ik.rhs_batch() == "");
        }
        SECTION("f f t") {
            EinsumPlanner ep_ij_ji_kl("i,j", "j,i", "k,l");
            REQUIRE(ep_ij_ji_kl.lhs_trace() == "");
            REQUIRE(ep_ij_ji_kl.rhs_trace() == "k,l");
            REQUIRE(ep_ij_ji_kl.lhs_dummy() == "");
            REQUIRE(ep_ij_ji_kl.rhs_dummy() == "");
            REQUIRE(ep_ij_ji_kl.lhs_free() == "j,i");
            REQUIRE(ep_ij_ji_kl.rhs_free() == "");
            REQUIRE(ep_ij_ji_kl.result_batch() == "");
            REQUIRE(ep_ij_ji_kl.lhs_batch() == "");
            REQUIRE(ep_ij_ji_kl.rhs_batch() == "");
        }
        SECTION("f f f") {
            EinsumPlanner ep_ijkl_kl_ji("i,j,k,l", "k,l", "j,i");
            REQUIRE(ep_ijkl_kl_ji.lhs_trace() == "");
            REQUIRE(ep_ijkl_kl_ji.rhs_trace() == "");
            REQUIRE(ep_ijkl_kl_ji.lhs_dummy() == "");
            REQUIRE(ep_ijkl_kl_ji.rhs_dummy() == "");
            REQUIRE(ep_ijkl_kl_ji.lhs_free() == "k,l");
            REQUIRE(ep_ijkl_kl_ji.rhs_free() == "j,i");
            REQUIRE(ep_ijkl_kl_ji.result_batch() == "");
            REQUIRE(ep_ijkl_kl_ji.lhs_batch() == "");
            REQUIRE(ep_ijkl_kl_ji.rhs_batch() == "");
        }
        SECTION("f t ft") {
            EinsumPlanner ep_ik_jl_kmi("i,k", "j,l", "k,m,i");
            REQUIRE(ep_ik_jl_kmi.lhs_trace() == "j,l");
            REQUIRE(ep_ik_jl_kmi.rhs_trace() == "m");
            REQUIRE(ep_ik_jl_kmi.lhs_dummy() == "");
            REQUIRE(ep_ik_jl_kmi.rhs_dummy() == "");
            REQUIRE(ep_ik_jl_kmi.lhs_free() == "");
            REQUIRE(ep_ik_jl_kmi.rhs_free() == "k,i");
            REQUIRE(ep_ik_jl_kmi.result_batch() == "");
            REQUIRE(ep_ik_jl_kmi.lhs_batch() == "");
            REQUIRE(ep_ik_jl_kmi.rhs_batch() == "");
        }
        SECTION("f ft t") {
            EinsumPlanner ep_jl_ljm_i("j,l", "l,j,m", "i");
            REQUIRE(ep_jl_ljm_i.lhs_trace() == "m");
            REQUIRE(ep_jl_ljm_i.rhs_trace() == "i");
            REQUIRE(ep_jl_ljm_i.lhs_dummy() == "");
            REQUIRE(ep_jl_ljm_i.rhs_dummy() == "");
            REQUIRE(ep_jl_ljm_i.lhs_free() == "l,j");
            REQUIRE(ep_jl_ljm_i.rhs_free() == "");
            REQUIRE(ep_jl_ljm_i.result_batch() == "");
            REQUIRE(ep_jl_ljm_i.lhs_batch() == "");
            REQUIRE(ep_jl_ljm_i.rhs_batch() == "");
        }
        SECTION("f ft ft") {
            EinsumPlanner ep_ik_kl_im("i,k", "k,l", "i,m");
            REQUIRE(ep_ik_kl_im.lhs_trace() == "l");
            REQUIRE(ep_ik_kl_im.rhs_trace() == "m");
            REQUIRE(ep_ik_kl_im.lhs_dummy() == "");
            REQUIRE(ep_ik_kl_im.rhs_dummy() == "");
            REQUIRE(ep_ik_kl_im.lhs_free() == "k");
            REQUIRE(ep_ik_kl_im.rhs_free() == "i");
            REQUIRE(ep_ik_kl_im.result_batch() == "");
            REQUIRE(ep_ik_kl_im.lhs_batch() == "");
            REQUIRE(ep_ik_kl_im.rhs_batch() == "");
        }
        SECTION("f d df") {
            EinsumPlanner ep_i_kj_jki("i", "k,j", "j,k,i");
            REQUIRE(ep_i_kj_jki.lhs_trace() == "");
            REQUIRE(ep_i_kj_jki.rhs_trace() == "");
            REQUIRE(ep_i_kj_jki.lhs_dummy() == "k,j");
            REQUIRE(ep_i_kj_jki.rhs_dummy() == "j,k");
            REQUIRE(ep_i_kj_jki.lhs_free() == "");
            REQUIRE(ep_i_kj_jki.rhs_free() == "i");
            REQUIRE(ep_i_kj_jki.result_batch() == "");
            REQUIRE(ep_i_kj_jki.lhs_batch() == "");
            REQUIRE(ep_i_kj_jki.rhs_batch() == "");
        }
        SECTION("f df d") {
            EinsumPlanner ep_ij_jikl_kl("i,j", "j,i,k,l", "k,l");
            REQUIRE(ep_ij_jikl_kl.lhs_trace() == "");
            REQUIRE(ep_ij_jikl_kl.rhs_trace() == "");
            REQUIRE(ep_ij_jikl_kl.lhs_dummy() == "k,l");
            REQUIRE(ep_ij_jikl_kl.rhs_dummy() == "k,l");
            REQUIRE(ep_ij_jikl_kl.lhs_free() == "j,i");
            REQUIRE(ep_ij_jikl_kl.rhs_free() == "");
            REQUIRE(ep_ij_jikl_kl.result_batch() == "");
            REQUIRE(ep_ij_jikl_kl.lhs_batch() == "");
            REQUIRE(ep_ij_jikl_kl.rhs_batch() == "");
        }
        SECTION("f df df") {
            EinsumPlanner ep_jm_im_ij("j,m", "i,m", "i,j");
            REQUIRE(ep_jm_im_ij.lhs_trace() == "");
            REQUIRE(ep_jm_im_ij.rhs_trace() == "");
            REQUIRE(ep_jm_im_ij.lhs_dummy() == "i");
            REQUIRE(ep_jm_im_ij.rhs_dummy() == "i");
            REQUIRE(ep_jm_im_ij.lhs_free() == "m");
            REQUIRE(ep_jm_im_ij.rhs_free() == "j");
            REQUIRE(ep_jm_im_ij.result_batch() == "");
            REQUIRE(ep_jm_im_ij.lhs_batch() == "");
            REQUIRE(ep_jm_im_ij.rhs_batch() == "");
        }
        SECTION("f dt df") {
            EinsumPlanner ep_lm_ij_iml("l,m", "i,j", "i,m,l");
            REQUIRE(ep_lm_ij_iml.lhs_trace() == "j");
            REQUIRE(ep_lm_ij_iml.rhs_trace() == "");
            REQUIRE(ep_lm_ij_iml.lhs_dummy() == "i");
            REQUIRE(ep_lm_ij_iml.rhs_dummy() == "i");
            REQUIRE(ep_lm_ij_iml.lhs_free() == "");
            REQUIRE(ep_lm_ij_iml.rhs_free() == "m,l");
            REQUIRE(ep_lm_ij_iml.result_batch() == "");
            REQUIRE(ep_lm_ij_iml.lhs_batch() == "");
            REQUIRE(ep_lm_ij_iml.rhs_batch() == "");
        }
        SECTION("f df dt") {
            EinsumPlanner ep_i_ij_jk("i", "i,j", "j,k");
            REQUIRE(ep_i_ij_jk.lhs_trace() == "");
            REQUIRE(ep_i_ij_jk.rhs_trace() == "k");
            REQUIRE(ep_i_ij_jk.lhs_dummy() == "j");
            REQUIRE(ep_i_ij_jk.rhs_dummy() == "j");
            REQUIRE(ep_i_ij_jk.lhs_free() == "i");
            REQUIRE(ep_i_ij_jk.rhs_free() == "");
            REQUIRE(ep_i_ij_jk.result_batch() == "");
            REQUIRE(ep_i_ij_jk.lhs_batch() == "");
            REQUIRE(ep_i_ij_jk.rhs_batch() == "");
        }
        SECTION("f df df") {
            EinsumPlanner ep_ijk_klm_jlmi("i,j,k", "k,l,m", "j,l,m,i");
            REQUIRE(ep_ijk_klm_jlmi.lhs_trace() == "");
            REQUIRE(ep_ijk_klm_jlmi.rhs_trace() == "");
            REQUIRE(ep_ijk_klm_jlmi.lhs_dummy() == "l,m");
            REQUIRE(ep_ijk_klm_jlmi.rhs_dummy() == "l,m");
            REQUIRE(ep_ijk_klm_jlmi.lhs_free() == "k");
            REQUIRE(ep_ijk_klm_jlmi.rhs_free() == "j,i");
            REQUIRE(ep_ijk_klm_jlmi.result_batch() == "");
            REQUIRE(ep_ijk_klm_jlmi.lhs_batch() == "");
            REQUIRE(ep_ijk_klm_jlmi.rhs_batch() == "");
        }
        SECTION("f dt dft") {
            EinsumPlanner ep_il_jm_jlis("i,l", "j,m", "j,l,i,s");
            REQUIRE(ep_il_jm_jlis.lhs_trace() == "m");
            REQUIRE(ep_il_jm_jlis.rhs_trace() == "s");
            REQUIRE(ep_il_jm_jlis.lhs_dummy() == "j");
            REQUIRE(ep_il_jm_jlis.rhs_dummy() == "j");
            REQUIRE(ep_il_jm_jlis.lhs_free() == "");
            REQUIRE(ep_il_jm_jlis.rhs_free() == "l,i");
            REQUIRE(ep_il_jm_jlis.result_batch() == "");
            REQUIRE(ep_il_jm_jlis.lhs_batch() == "");
            REQUIRE(ep_il_jm_jlis.rhs_batch() == "");
        }
        SECTION("f dft dt") {
            EinsumPlanner ep_i_jikm_kjn("i", "j,i,k,m", "k,j,n");
            REQUIRE(ep_i_jikm_kjn.lhs_trace() == "m");
            REQUIRE(ep_i_jikm_kjn.rhs_trace() == "n");
            REQUIRE(ep_i_jikm_kjn.lhs_dummy() == "j,k");
            REQUIRE(ep_i_jikm_kjn.rhs_dummy() == "k,j");
            REQUIRE(ep_i_jikm_kjn.lhs_free() == "i");
            REQUIRE(ep_i_jikm_kjn.rhs_free() == "");
            REQUIRE(ep_i_jikm_kjn.result_batch() == "");
            REQUIRE(ep_i_jikm_kjn.lhs_batch() == "");
            REQUIRE(ep_i_jikm_kjn.rhs_batch() == "");
        }
        SECTION("f dft dft") {
            EinsumPlanner ep_ijk_nilsk_sammjl("i,j,k", "n,i,l,s,k",
                                              "s,a,m,m,j,l");
            REQUIRE(ep_ijk_nilsk_sammjl.lhs_trace() == "n");
            REQUIRE(ep_ijk_nilsk_sammjl.rhs_trace() == "a,m");
            REQUIRE(ep_ijk_nilsk_sammjl.lhs_dummy() == "l,s");
            REQUIRE(ep_ijk_nilsk_sammjl.rhs_dummy() == "s,l");
            REQUIRE(ep_ijk_nilsk_sammjl.lhs_free() == "i,k");
            REQUIRE(ep_ijk_nilsk_sammjl.rhs_free() == "j");
            REQUIRE(ep_ijk_nilsk_sammjl.result_batch() == "");
            REQUIRE(ep_ijk_nilsk_sammjl.lhs_batch() == "");
            REQUIRE(ep_ijk_nilsk_sammjl.rhs_batch() == "");
        }
    }

    SECTION("Result in batched free indices") {
        SECTION("bf bt bf") {
            EinsumPlanner ep_ibk_bjl_bik("i,b,k", "b,j,l", "b,i,k");
            REQUIRE(ep_ibk_bjl_bik.lhs_trace() == "j,l");
            REQUIRE(ep_ibk_bjl_bik.rhs_trace() == "");
            REQUIRE(ep_ibk_bjl_bik.lhs_dummy() == "");
            REQUIRE(ep_ibk_bjl_bik.rhs_dummy() == "");
            REQUIRE(ep_ibk_bjl_bik.lhs_free() == "");
            REQUIRE(ep_ibk_bjl_bik.rhs_free() == "i,k");
            REQUIRE(ep_ibk_bjl_bik.result_batch() == "b");
            REQUIRE(ep_ibk_bjl_bik.lhs_batch() == "b");
            REQUIRE(ep_ibk_bjl_bik.rhs_batch() == "b");
        }
        SECTION("bf bf bt") {
            EinsumPlanner ep_bij_jib_kbl("b,i,j", "j,i,b", "k,b,l");
            REQUIRE(ep_bij_jib_kbl.lhs_trace() == "");
            REQUIRE(ep_bij_jib_kbl.rhs_trace() == "k,l");
            REQUIRE(ep_bij_jib_kbl.lhs_dummy() == "");
            REQUIRE(ep_bij_jib_kbl.rhs_dummy() == "");
            REQUIRE(ep_bij_jib_kbl.lhs_free() == "j,i");
            REQUIRE(ep_bij_jib_kbl.rhs_free() == "");
            REQUIRE(ep_bij_jib_kbl.result_batch() == "b");
            REQUIRE(ep_bij_jib_kbl.lhs_batch() == "b");
            REQUIRE(ep_bij_jib_kbl.rhs_batch() == "b");
        }
        SECTION("bf bf bf") {
            EinsumPlanner ep_iajkbl_kbla_ajbi("i,a,j,k,b,l", "k,b,l,a",
                                              "a,j,b,i");
            REQUIRE(ep_iajkbl_kbla_ajbi.lhs_trace() == "");
            REQUIRE(ep_iajkbl_kbla_ajbi.rhs_trace() == "");
            REQUIRE(ep_iajkbl_kbla_ajbi.lhs_dummy() == "");
            REQUIRE(ep_iajkbl_kbla_ajbi.rhs_dummy() == "");
            REQUIRE(ep_iajkbl_kbla_ajbi.lhs_free() == "k,l");
            REQUIRE(ep_iajkbl_kbla_ajbi.rhs_free() == "j,i");
            REQUIRE(ep_iajkbl_kbla_ajbi.result_batch() == "a,b");
            REQUIRE(ep_iajkbl_kbla_ajbi.lhs_batch() == "b,a");
            REQUIRE(ep_iajkbl_kbla_ajbi.rhs_batch() == "a,b");
        }
        SECTION("bf bt bft") {
            EinsumPlanner ep_ibk_jbl_kbmi("i,b,k", "j,b,l", "k,b,m,i");
            REQUIRE(ep_ibk_jbl_kbmi.lhs_trace() == "j,l");
            REQUIRE(ep_ibk_jbl_kbmi.rhs_trace() == "m");
            REQUIRE(ep_ibk_jbl_kbmi.lhs_dummy() == "");
            REQUIRE(ep_ibk_jbl_kbmi.rhs_dummy() == "");
            REQUIRE(ep_ibk_jbl_kbmi.lhs_free() == "");
            REQUIRE(ep_ibk_jbl_kbmi.rhs_free() == "k,i");
            REQUIRE(ep_ibk_jbl_kbmi.result_batch() == "b");
            REQUIRE(ep_ibk_jbl_kbmi.lhs_batch() == "b");
            REQUIRE(ep_ibk_jbl_kbmi.rhs_batch() == "b");
        }
        SECTION("bf bft bt") {
            EinsumPlanner ep_jlb_ljmb_ib("j,l,b", "l,j,m,b", "i,b");
            REQUIRE(ep_jlb_ljmb_ib.lhs_trace() == "m");
            REQUIRE(ep_jlb_ljmb_ib.rhs_trace() == "i");
            REQUIRE(ep_jlb_ljmb_ib.lhs_dummy() == "");
            REQUIRE(ep_jlb_ljmb_ib.rhs_dummy() == "");
            REQUIRE(ep_jlb_ljmb_ib.lhs_free() == "l,j");
            REQUIRE(ep_jlb_ljmb_ib.rhs_free() == "");
            REQUIRE(ep_jlb_ljmb_ib.result_batch() == "b");
            REQUIRE(ep_jlb_ljmb_ib.lhs_batch() == "b");
            REQUIRE(ep_jlb_ljmb_ib.rhs_batch() == "b");
        }
        SECTION("bf bft bft") {
            EinsumPlanner ep_ibk_bkl_bim("i,b,k", "b,k,l", "b,i,m");
            REQUIRE(ep_ibk_bkl_bim.lhs_trace() == "l");
            REQUIRE(ep_ibk_bkl_bim.rhs_trace() == "m");
            REQUIRE(ep_ibk_bkl_bim.lhs_dummy() == "");
            REQUIRE(ep_ibk_bkl_bim.rhs_dummy() == "");
            REQUIRE(ep_ibk_bkl_bim.lhs_free() == "k");
            REQUIRE(ep_ibk_bkl_bim.rhs_free() == "i");
            REQUIRE(ep_ibk_bkl_bim.result_batch() == "b");
            REQUIRE(ep_ibk_bkl_bim.lhs_batch() == "b");
            REQUIRE(ep_ibk_bkl_bim.rhs_batch() == "b");
        }
        SECTION("bf bd bdf") {
            EinsumPlanner ep_ib_bkj_bjki("i,b", "b,k,j", "b,j,k,i");
            REQUIRE(ep_ib_bkj_bjki.lhs_trace() == "");
            REQUIRE(ep_ib_bkj_bjki.rhs_trace() == "");
            REQUIRE(ep_ib_bkj_bjki.lhs_dummy() == "k,j");
            REQUIRE(ep_ib_bkj_bjki.rhs_dummy() == "j,k");
            REQUIRE(ep_ib_bkj_bjki.lhs_free() == "");
            REQUIRE(ep_ib_bkj_bjki.rhs_free() == "i");
            REQUIRE(ep_ib_bkj_bjki.result_batch() == "b");
            REQUIRE(ep_ib_bkj_bjki.lhs_batch() == "b");
            REQUIRE(ep_ib_bkj_bjki.rhs_batch() == "b");
        }
        SECTION("bf bdf bd") {
            EinsumPlanner ep_ibj_jikbl_klb("i,b,j", "j,i,k,b,l", "k,l,b");
            REQUIRE(ep_ibj_jikbl_klb.lhs_trace() == "");
            REQUIRE(ep_ibj_jikbl_klb.rhs_trace() == "");
            REQUIRE(ep_ibj_jikbl_klb.lhs_dummy() == "k,l");
            REQUIRE(ep_ibj_jikbl_klb.rhs_dummy() == "k,l");
            REQUIRE(ep_ibj_jikbl_klb.lhs_free() == "j,i");
            REQUIRE(ep_ibj_jikbl_klb.rhs_free() == "");
            REQUIRE(ep_ibj_jikbl_klb.result_batch() == "b");
            REQUIRE(ep_ibj_jikbl_klb.lhs_batch() == "b");
            REQUIRE(ep_ibj_jikbl_klb.rhs_batch() == "b");
        }
        SECTION("bf bdf bdf") {
            EinsumPlanner ep_jmb_imb_ijb("j,m,b", "i,m,b", "i,j,b");
            REQUIRE(ep_jmb_imb_ijb.lhs_trace() == "");
            REQUIRE(ep_jmb_imb_ijb.rhs_trace() == "");
            REQUIRE(ep_jmb_imb_ijb.lhs_dummy() == "i");
            REQUIRE(ep_jmb_imb_ijb.rhs_dummy() == "i");
            REQUIRE(ep_jmb_imb_ijb.lhs_free() == "m");
            REQUIRE(ep_jmb_imb_ijb.rhs_free() == "j");
            REQUIRE(ep_jmb_imb_ijb.result_batch() == "b");
            REQUIRE(ep_jmb_imb_ijb.lhs_batch() == "b");
            REQUIRE(ep_jmb_imb_ijb.rhs_batch() == "b");
        }
        SECTION("bf bdt bdf") {
            EinsumPlanner ep_lbqm_iqbj_iqbml("l,b,q,m", "i,q,b,j", "i,q,b,m,l");
            REQUIRE(ep_lbqm_iqbj_iqbml.lhs_trace() == "j");
            REQUIRE(ep_lbqm_iqbj_iqbml.rhs_trace() == "");
            REQUIRE(ep_lbqm_iqbj_iqbml.lhs_dummy() == "i");
            REQUIRE(ep_lbqm_iqbj_iqbml.rhs_dummy() == "i");
            REQUIRE(ep_lbqm_iqbj_iqbml.lhs_free() == "");
            REQUIRE(ep_lbqm_iqbj_iqbml.rhs_free() == "m,l");
            REQUIRE(ep_lbqm_iqbj_iqbml.result_batch() == "b,q");
            REQUIRE(ep_lbqm_iqbj_iqbml.lhs_batch() == "q,b");
            REQUIRE(ep_lbqm_iqbj_iqbml.rhs_batch() == "q,b");
        }
        SECTION("bf bdf bdt") {
            EinsumPlanner ep_bi_bij_bjk("b,i", "b,i,j", "b,j,k");
            REQUIRE(ep_bi_bij_bjk.lhs_trace() == "");
            REQUIRE(ep_bi_bij_bjk.rhs_trace() == "k");
            REQUIRE(ep_bi_bij_bjk.lhs_dummy() == "j");
            REQUIRE(ep_bi_bij_bjk.rhs_dummy() == "j");
            REQUIRE(ep_bi_bij_bjk.lhs_free() == "i");
            REQUIRE(ep_bi_bij_bjk.rhs_free() == "");
            REQUIRE(ep_bi_bij_bjk.result_batch() == "b");
            REQUIRE(ep_bi_bij_bjk.lhs_batch() == "b");
            REQUIRE(ep_bi_bij_bjk.rhs_batch() == "b");
        }
        SECTION("bf bdf bdf") {
            EinsumPlanner ep_ibjk_kblm_jblmi("i,b,j,k", "k,b,l,m", "j,b,l,m,i");
            REQUIRE(ep_ibjk_kblm_jblmi.lhs_trace() == "");
            REQUIRE(ep_ibjk_kblm_jblmi.rhs_trace() == "");
            REQUIRE(ep_ibjk_kblm_jblmi.lhs_dummy() == "l,m");
            REQUIRE(ep_ibjk_kblm_jblmi.rhs_dummy() == "l,m");
            REQUIRE(ep_ibjk_kblm_jblmi.lhs_free() == "k");
            REQUIRE(ep_ibjk_kblm_jblmi.rhs_free() == "j,i");
            REQUIRE(ep_ibjk_kblm_jblmi.result_batch() == "b");
            REQUIRE(ep_ibjk_kblm_jblmi.lhs_batch() == "b");
            REQUIRE(ep_ibjk_kblm_jblmi.rhs_batch() == "b");
        }
        SECTION("bf bdt bdft") {
            EinsumPlanner ep_ilb_bjm_jlibs("i,l,b", "b,j,m", "j,l,i,b,s");
            REQUIRE(ep_ilb_bjm_jlibs.lhs_trace() == "m");
            REQUIRE(ep_ilb_bjm_jlibs.rhs_trace() == "s");
            REQUIRE(ep_ilb_bjm_jlibs.lhs_dummy() == "j");
            REQUIRE(ep_ilb_bjm_jlibs.rhs_dummy() == "j");
            REQUIRE(ep_ilb_bjm_jlibs.lhs_free() == "");
            REQUIRE(ep_ilb_bjm_jlibs.rhs_free() == "l,i");
            REQUIRE(ep_ilb_bjm_jlibs.result_batch() == "b");
            REQUIRE(ep_ilb_bjm_jlibs.lhs_batch() == "b");
            REQUIRE(ep_ilb_bjm_jlibs.rhs_batch() == "b");
        }
        SECTION("bf bdft bdt") {
            EinsumPlanner ep_bi_jikbm_bkjn("b,i", "j,i,k,b,m", "b,k,j,n");
            REQUIRE(ep_bi_jikbm_bkjn.lhs_trace() == "m");
            REQUIRE(ep_bi_jikbm_bkjn.rhs_trace() == "n");
            REQUIRE(ep_bi_jikbm_bkjn.lhs_dummy() == "j,k");
            REQUIRE(ep_bi_jikbm_bkjn.rhs_dummy() == "k,j");
            REQUIRE(ep_bi_jikbm_bkjn.lhs_free() == "i");
            REQUIRE(ep_bi_jikbm_bkjn.rhs_free() == "");
            REQUIRE(ep_bi_jikbm_bkjn.result_batch() == "b");
            REQUIRE(ep_bi_jikbm_bkjn.lhs_batch() == "b");
            REQUIRE(ep_bi_jikbm_bkjn.rhs_batch() == "b");
        }
        SECTION("bf bdft bdft") {
            EinsumPlanner ep_ijbk_bnilsk_bsammjl("i,j,b,k", "b,n,i,l,s,k",
                                                 "b,s,a,m,m,j,l");
            REQUIRE(ep_ijbk_bnilsk_bsammjl.lhs_trace() == "n");
            REQUIRE(ep_ijbk_bnilsk_bsammjl.rhs_trace() == "a,m");
            REQUIRE(ep_ijbk_bnilsk_bsammjl.lhs_dummy() == "l,s");
            REQUIRE(ep_ijbk_bnilsk_bsammjl.rhs_dummy() == "s,l");
            REQUIRE(ep_ijbk_bnilsk_bsammjl.lhs_free() == "i,k");
            REQUIRE(ep_ijbk_bnilsk_bsammjl.rhs_free() == "j");
            REQUIRE(ep_ijbk_bnilsk_bsammjl.result_batch() == "b");
            REQUIRE(ep_ijbk_bnilsk_bsammjl.lhs_batch() == "b");
            REQUIRE(ep_ijbk_bnilsk_bsammjl.rhs_batch() == "b");
        }
    }
}
