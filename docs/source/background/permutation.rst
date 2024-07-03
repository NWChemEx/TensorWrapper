.. Copyright 2024 NWChemEx-Project
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

.. _permutations:

############
Permutations
############

Many tensors possess symmetry such that swapping two (or more modes) results in
the same tensor. For example, consider the matrix :math:`A` given by:

.. math::

   A = \begin{pmatrix}
       1 & 2\\
       2 & 3
       \end{pmatrix}

If we declare another matrix, :math:`B`, such that rows of :math:`B` were the
columns of :math:`A` we get:

.. math::

   B = \begin{pmatrix}
       1 & 2\\
       2 & 3
       \end{pmatrix}

i.e., :math:`A=B` and we say that :math:`A` (or :math:`B`) is symmetric with
respect to permuting modes 0 (the rows) and 1 (the columns). Permutations are a
key component of describing tensor symmetry and this page reviews the math
behind them.

***********
Terminology
***********

Many of these definitions rely on the algorithm needed to reorder a set of
objects :math:`I` into :math:`I'`. Assuming there are :math:`n+1` elements in
:math:`I` the algorithm is:

1. Set :math:`i` to 0.
2. If :math:`i` is :math:`n+1` terminate.
3. If :math:`I_i` (the :math:`i`-th element of :math:`I`) is in the correct spot
   increment :math:`i` and return to step 2.
4. Let :math:`j` be the correct position for :math:`I_i`, swap :math:`I_i` and
   :math:`I_j`. The element which was previously :math:`I_i` is now in its
   correct spot. Return to step 3 with the realization that the current
   :math:`I_i` is now the element which was :math:`I_j`.

.. glossary::

   cycle
      A "cycle" is a series of :term:`transpositions <transposition>` needed
      to get each element in a given :term:`orbit` into its correct position.
      Cycles involving more than two elements are not unique as the transposes
      can be done in different orders.

   cyclic permutation
      A :term:`permutation` which can be written as a single :term:`cycle`. All
      permutations can be decomposed into one or more cyclic permutations.

   noncyclic permutation
      A :term:`permutation` which can not be written as a single :term:`cycle`.

   orbit
      For a given :math:`i` the algorithm above will cause one or more elements
      to be labeled as :math:`I_i`. The set of elements which can be labeled as
      :math:`I_i` defines the orbit of :math:`I_i`.

   permutation
     A particular ordering of a set of objects. "Permuting" is the act of going
     from one permutation to another.

   transposition
     Swapping two objects.

********
Notation
********

Having to write "permute mode 0 with mode 1" is already tedious and becomes more
tedious when more modes are involved. One common notation for writing a
permutation is "cycle notation". For a :term:`cyclic permutation` involving
:math:`n+1` objects the :term:`cycle` is written like
:math:`(e_0, e_1,..., e_n)` where :math:`e_i` is the index of the :term:`i`-th
object BEFORE the permutation and :math:`e_{i+1}` is the index of :math:`e_{i}`
AFTER the permutation (except for :math:`e_{n}` which becomes :math:`e_0`).
