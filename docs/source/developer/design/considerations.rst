.. Copyright 2023 NWChemEx-Project
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

.. _tw_considerations:

############################
TensorWrapper Considerations
############################

Tensors are the domain-specific language (DSL) of physics, since nearly every
law of physics is succinctly summarized by a tensor equation. That said,
naively creating arrays of floating-point values and subjecting them to
the mathematical operations implied by the equations often leaves much
performance on the table. Nonetheless we argue that having a tensor-based DSL is
important for physics-based codes because:

- Facilitates translating theory to code.
- Encapsulates mathematical optimizations.
- Code is easier to read/rationalize about.

The following subsections summarize the considerations our tensor DSL, TW, must
contend with if it is to encapsulate the process of going from tensor
expressions to performant implementations.

***********
Performance
***********

Code written with TW should be competitive with hand-rolled implementations.
When appropriate, this requires TW to take advantage of:

#. Hardware optimizations.

   - Intrinsics.
   - Accelerators.

#. Memory optimizations.

   - On-the-fly recomputation vs. storing.
   - Literal data layout.
   - Distributed vs. replicated.

#. Mathematics optimizations

   - Order of operations.
   - Equation factoring.
   - Common intermediates.

#. Physics optimizations.

   - Sparse nature of equations.
   - Tensor symmetries (e.g., permutational and antisymmetric).


***************
User Experience
***************

#. Easy to express arbitrary tensor operations.

   - Einstein notation for basic math.
   - Slicing for algorithmic data movement.

#. Ideally tell TW the math to do, and TW does it performantly.

   - ParallelZone provides view of runtime, use it to reason about operations.
   - Treating user interface as DSL allows equations to be optimized.

#. Users may provide performant implementations.

   - Mechanism to override underlying optimizations.
