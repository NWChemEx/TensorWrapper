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

.. _designing_the_expression_component:

##################################
Designing the Expression Component
##################################

This page documents the process of designing the expression component of
TensorWrapper.

*********************************
What is the expression component?
*********************************

Users of a tensor library have two main needs: to create tensors and to solve
tensor expressions involving those tensors. The expression component contains
the pieces of the :ref:`term_dsl` required to write succinct tensor expressions.

***************************************
Why do we need an expression component?
***************************************

Most libraries tend to be eagerly evaluated, meaning code is evaluated as it is
encountered. For simple, easily optimized code such a strategy is fine.
Unfortunately, tensor expressions tend to be significantly harder to optimize,
plus the optimal evaluation tends to be highly runtime dependent. A potential
solution to this problem is to evaluate the code lazily, instead of eagerly.
Lazily evaluated code is "recorded" until it is needed. Once it is needed it
is played back.

Most tensor operations are steps in a bigger algorithm. By lazily
evaluating tensor computations we can register the user's intention and
optimize our evaluation strategy accordingly. The point of the expression layer
is to provide a user-friendly DSL for lazy evaluation of tensors.

***********************************
Expression Component Considerations
***********************************

.. _ec_compose_multiple_objects:

compose multiple objects
   While our goal is to ultimately be able to compose ``TensorWrapper`` objects,
   we can decompose the work needed to do so by ensuring we can compose the
   pieces of the ``TensorWrapper`` object. In particular we want to be able to
   compose objects from the shape, symmetry, sparsity, and allocator components.

.. _ec_generalized_einstein_notation:

generalized Einstein notation
   Many of the most common tensor operations are succinctly specified using
   generalized Einstein notation (indices appearing on the right side of
   an equation, but not the left are summed over).

   - For expressions which involve more than one term, this notation may be
     ambiguous (TODO: add an example when I can recall how it's ambiguous).
   - Math operations include: addition, subtraction, scaling, multiplication,
     and division. The latter is missing from a number of existing tensor
     libraries.
   - Note that multiplication actually covers a number of operations including:
     element-wise product, matrix multiplication, contraction, trace, and direct
     product.

.. _ec_non_einstein_math:

non-Einstein math
   There are a number of tensor operations which are not easily expressed using
   generalized Einstein notation, but are still important for tensor algebra.
   In particular:

   - Eigen system and linear equation solvers.
   - Factorizations/decompositions: Cholesky, QR, SVD.
   - Slicing/chipping
   - Matrix powers

.. _ec_identifying_common_intermediates:

identifying common intermediates
   As we build

sparse maps
   The sparsity component :ref:`sparsity_design` realized that sparse maps are
   a mechanism for creating sparsity objects and thus should live above the
   sparsity component. Sparse maps are most naturally expressed as
   relationships among dummy indices, that is given an expression like
   ``C("i,j") = A("i,j,a,b") * B("i,j,b,a");`` we would define a sparse map, say
   which maps a given ``i,j`` pair to a set of corresponding, non-zero ``a,b``
   pairs. From that sparse map we can work out the sparsity of ``A``, ``B``,
   and ``C``. The trick is we need to know how the sparse map's modes map to
   the modes of ``A``, ``B``, and ``C``. This information is
   available in the expression layer.

***************************
Expression Component Design
***************************

To implement lazy evaluation in C++ one typically relies on a C++ template
meta-programming technique known as expression templates. With expression
templates, users write code using expression objects. The expression objects
capture the user's intent and map it to a single, heavily nested, instance of a
class template. The resulting type contains all of the information about the
requested calculation. Then when the object is instantiated at runtime, the
class plays back the type's namesake computation.


- ``IndexedSparsity``, ``IndexedShape``, ``IndexedSymmetry``, etc. should use
  this component.
- Sparse maps got punted here.
- Hooks for linear algebra including eigen solves, etc.
