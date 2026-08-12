.. Copyright 2026 NWChemEx-Project
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
.. http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.

.. _tw_designing_uq_contraction:

#####################################
Contracting Uncertainty-Aware Tensors
#####################################

The point of this page is to record why tensor contraction for uncertainty
quantification (UQ) scalar types (``sigma::Uncertain``, ``sigma::Interval``,
``sigma::Affine``, ``sigma::ThresholdedAffine``, and ``sigma::TaylorModel``)
does **not** go through the same code path as contraction for ``float``/
``double``, even though both live in the same
``EigenTensorImpl::contraction_assignment_`` function.

************************************
What is UQ-aware tensor contraction?
************************************

``EigenTensorImpl::contraction_assignment_`` implements tensor contraction via
TTGT (Transpose-Transpose-GEMM-Transpose): the operands are physically
transposed into a 2D matrix layout, multiplied with a dense GEMM, and the
result is transposed back. This works uniformly for any scalar type ``Eigen``
knows how to multiply and add -- which includes UQ scalar types, since each has
an ``Eigen::NumTraits`` specialization (see ``sigma/include/sigma/*/eigen_compat.hpp``).

**************************************
Why do we need to treat it specially?
**************************************

For UQ scalar types, correctness depends on more than the numeric value: a
type like ``sigma::TaylorModel<T>`` also carries a *truncation order*, and
operations between operands of different orders resolve to
``min(lhs.max_order(), rhs.max_order())`` (documented, intentional behavior --
see ``sigma::TaylorModel::operator+=``/``operator*=``).

Eigen's dense GEMM/GEBP kernel seeds each output accumulator cell with an
internal "zero" of the scalar type before accumulating into it. For UQ scalar
types this seed is **not** a genuinely empty/default-constructed value -- it is
a concrete value constructed via an implicit conversion from a numeric literal
(e.g. ``TaylorModel(0)``), which for ``TaylorModel`` resolves to
``default_max_order()`` (compile-time constant, currently 2). Because mixed-
order operations resolve downward via ``min()``, every subsequent accumulation
into that cell is silently clamped to order 2, **regardless of the order the
real operands were built at**. The bug is invisible for ``float``/``double``
because those types carry no such metadata -- only the numeric value is ever
wrong-or-right, and Eigen's zero seed is numerically correct (``0.0``).

We confirmed this is genuinely unreachable through the public customization
point: overriding ``Eigen::NumTraits<TaylorModel<T>>::Zero()`` to return an
explicitly empty ``TaylorModel()`` had no effect -- instrumenting the override
with a call counter showed it is never invoked by Eigen's GEMM/GEBP kernel for
this scalar type. The zero-seed is constructed through some other, non-
interceptable internal mechanism.

**********************
Existing Options
**********************

Patch Eigen directly
   Eigen is a vendored third-party dependency several layers below
   ``TensorWrapper`` that this project does not own or maintain a patched fork
   of. Locating and patching the exact internal zero-construction site inside
   ``GeneralMatrixMatrix.h``/the GEBP kernel would be fragile (version-specific)
   and outside the maintenance boundary this project wants to take on.

Override ``Eigen::NumTraits<Scalar>::Zero()``
   The natural customization point for "what does zero look like for this
   scalar type." Tried and confirmed ineffective for this code path (see
   above) -- Eigen's dense GEMM does not consult it here.

***************
Chosen Strategy
***************

``contraction_assignment_`` branches at compile time on
``tensorwrapper::types::is_uq_type_v<FloatType>`` (an existing trait, already
used the same way elsewhere in this codebase, e.g.
``generate/generate_eigenvalues.cpp``, and throughout ``SCF``'s eigensolver).
For UQ scalar types, instead of ``omatrix = lmatrix * rmatrix``, a hand-rolled
triple loop performs the same GEMM manually:

.. code-block:: c++

   if constexpr(types::is_uq_type_v<FloatType>) {
       for(std::size_t i = 0; i < lrows; ++i) {
           for(std::size_t j = 0; j < rcols; ++j) {
               FloatType acc{}; // genuinely empty/default-constructed
               for(std::size_t k = 0; k < lcols; ++k)
                   acc += lmatrix(i, k) * rmatrix(k, j);
               omatrix(i, j) = acc;
           }
       }
   } else {
       omatrix = lmatrix * rmatrix;
   }

The key detail is that ``acc`` starts as a *real* ``FloatType{}`` -- for
``TaylorModel`` this is genuinely empty (``empty() == true``), never a
concrete order-2 value. UQ scalar types' ``operator+=`` special-cases an empty
left-hand side (``if(empty()) { return *this = other; }``), so the first term
accumulated into ``acc`` adopts the real operand's order unmodified. Every
later ``+=``/``*=`` in the inner loop then combines two operands that already
carry the correct (matching, in the common case) order, so ``min()`` is a
no-op. The configured truncation order survives the contraction intact.

This only changes behavior for ``is_uq_type_v<FloatType>``; ``float``/
``double`` contractions are untouched and keep using Eigen's optimized,
vectorized GEMM.

**********************
Further Considerations
**********************

Performance
   The hand-rolled loop is a plain, unblocked, unvectorized triple loop -- it
   does not have Eigen's cache blocking or SIMD. This is an intentional
   tradeoff: correctness for UQ scalar types over GEMM performance for them.
   **Do not "simplify" this back to** ``omatrix = lmatrix * rmatrix`` **for UQ
   types** -- that regresses this exact bug. If UQ contraction performance
   becomes a bottleneck, the fix should be a UQ-aware blocked/vectorized loop
   that still seeds accumulators from an empty ``FloatType{}``, not a return
   to Eigen's GEMM.

Other backends
   This fix lives in ``EigenTensorImpl`` (the dense Eigen backend) only. Other
   backends (e.g. the CUDA/cuTensor backend) would need the equivalent
   treatment if/when they are extended to support UQ scalar types; nothing
   about this fix propagates automatically to them.

Non-``TaylorModel`` UQ types
   ``sigma::Uncertain``, ``sigma::Interval``, ``sigma::Affine``, and
   ``sigma::ThresholdedAffine`` do not carry a truncation order the way
   ``TaylorModel`` does, so they were not directly affected by the order-
   collapse symptom. They are still routed through the hand-rolled loop
   (via the same ``is_uq_type_v`` gate) since they are also
   ``RequireInitialization = 1`` custom scalar types subject to the same
   unauditable Eigen zero-seed; using a genuinely empty accumulator for them
   is at least as correct as, and no more expensive than, whatever Eigen's
   internal seed happened to be.
