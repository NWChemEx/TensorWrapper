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

.. _einstein_summation_convention:

#############################
Einstein Summation Convention
#############################

The purpose of this page is to give a primer on the Einstein summation
conventions and to point out why they are useful and when they can't be used.

**********
Motivation
**********

.. |n| replace:: :math:`n`
.. |u| replace:: :math:`\mathbf{u}`
.. |v| replace:: :math:`\mathbf{v}`
.. |M| replace:: :math:`\mathbf{M}`
.. |A| replace:: :math:`\mathbf{A}`
.. |B| replace:: :math:`\mathbf{B}`
.. |C| replace:: :math:`\mathbf{C}`

Many tensor operations involve summing over pairs of repeated indices. For
example, the inner-product between two vectors, |u| and |v|, looks like:

.. math::

   c = \sum_{i} u_i v_i.

Other prominent examples include the product of a vector |v| with a matrix |M|,
:math:`\mathbf{vM}`:

.. math::

   \left[\mathbf{vM}\right]_j = \sum_{i} v_i M_{ij}

and the matrix-matrix product, |C|, between two matrices |A| and |B|:

.. math::

   C_{ij} = \sum_{k} A_{ik}B_{kj}.

Given the prevalence of such tensor operations people wanting to save some
writing/typing often forgo the explicit summation symbols and agree on a
summation convention which says that pairs of repeated indices appearing in a
term are summed over. This summation convention was brought to physics by Albert
Einstein in his work on general relativity and is thus commonly known as the
Einstein summation convention. Using the Einstein summation convention the
above three equations would look like:

.. math::

   c =& u_i v_i\\
   \left[\mathbf{vM}\right]_j =& \sum_{i} v_i M_{ij}\\
   C_{ij} =& \sum_{k} A_{ik}B_{kj}

The traditional Einstein summation convention doesn't allow for element-wise
products like (note there is no implicit summation in the next equation):

.. math::

   C_{ij} = A_{ij}B_{ij}.

Making the observation that indices which are summed over only appear on one
side of the equation. We can define a "generalized Einstein summation
convention" which says that if an index only appears on a single side of an
equation it is implicitly summed over. Note that this also relaxes the "pair"
restriction so summing over a row of a matrix to form a vector, *i.e.*,

.. math::

   v_j = \sum_{i} A_{ij}

could be written using the generalized Einstein summation convention as:

.. math::

   v_j = A_{ij}

similarly the trace of |A| can be written:

.. math::

   Tr\left(\mathbf{A}\right) = A_{ii}.

For the purposes of TensorWrapper, generalized Einstein summation convention
allows us to write many tensor operations in a user-friendly manner. For example
the above equations

.. code-block:: c++

   // Vector-vector inner-product
   TensorWrapper c, u, v;
   c("") = u("i") * v("i");

   // Vector-matrix product
   TensorWrapper M, vM;
   vM("j") = v("i") * M("i,j");

   // Matrix-matrix product
   TensorWrapper C, A, B;
   C("i,j") = A("i,k") * B("k,j");

   // Element-wise matrix product
   C("i,j") = A("i,j") * B("i,j");

   // Summing over a row of a matrix
   v("j") = A("i,j");

   // Trace of a matrix
   c("") = A("i,i");

***********
Limitations
***********

Traditional Einstein summation convention is usually said to be limited to pairs
of repeated indices because repeating an index three or more times is ambiguous
(*e.g.*, see
`here https://math.stackexchange.com/questions/436515/problem-with-free-index-in-einstein-summation-notation`__).
Consider the equation:

.. math::

   u_{i} = v_{i}B_{ii}.

Traditional Einstein summation convention requires that we must sum over pairs
of repeated indices. Since :math:`i` appears more than once we must sum it.
There are three possible ways to sum over pairs of :math:`i`. For clarity,
we perform a dummy index transformation so that we are summing over :math:`j`
instead:

.. math::

   u_{i} = \sum_{j} v_j B_{ji}

or

.. math::

   u_{i} = \sum_{j} v_{j}B_{ij}

or

.. math::

   u_{i} = \sum_{j}  v_{i} B_{jj}.

We however argue that none of these interpretations are in the spirit of
conventional summation notation because changing the value of a dummy index
must be done to all occurrences of the dummy index in order to preserve the
meaning. Even for vectors we can not selectively change dummy indices without
changing the meaning, *i.e.*,

.. math::

   \sum_{i} u_iv_i \neq \sum_{ij} u_iv_j.

The general Einstein summation convention has no ambiguity for three repeated
indices and, consistent with conventional summation conventions, recognizes
:math:`u_i=v_iB_{ii}` as the product of |v| and the diagonal of |B|. In fact,
generalized Einstein summation convention has no ambiguity since
every index is either summed over, or not, based on whether it appears on one or
both sides of an equation respectively. Indices which must have the same values
in each term must be assigned the same letter. Indices which are allowed to
vary independently must be assigned different letters.

That said, in dynamic programming situations it can be hard to ensure indices
are chosen in a manner which adheres to the general Einstein summation
convention. For example, it is not unreasonable to write something like:

.. code-block:: c++

   auto l = [](std::size_t i){
       auto [A, B]  = build_tensors_from_parameter(i);
       return A("i,k")*B("k,j");
   }
   auto rhs = l(0) + l(1);
   TensorWrapper C;
   C("i,j") = rhs;

The idea being we have a function which generates terms for an expression, then
the caller of the function assembles those terms into a larger expression before
ultimately assigning it to an indexed tensor. As written the above would
generate an expression which looks something like (note that ``A`` and ``B``
in the lambda are temporary variable names):

.. code-block:: c++

   C("i,j") = A("i,k")*B("k,j")* D("i,k")*E("k,j");

This is an unambiguous expression, with summations inserted it's equivalent to:

.. math::

   C_{ij} = \sum_{k}\left(A_{ik}B_{kj}D_{ik}E_{kj}\right)

which is not the same as:

.. math::

   C_{ij} = \sum_{kl}\left(A_{ik}B_{kj}D_{il}E_{lj}\right).

Point being, if the intent of the function calls was to return a matrix in a
factorized form, they needed to choose different contraction indices in each
call. Generally speaking, generalized Einstein summation convention is best
applied to binary operations and not to nested expressions.
