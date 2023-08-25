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

.. _tw_terminology:

#########################
TensorWrapper Terminology
#########################

.. |n| replace:: :math:`n`
.. |m| replace:: :math:`m`

When discussing tensors, different researchers use different terminology. This
section provides a glossary of the tensor terminology we will use throughout
TensorWrapper.

******************
Tensor Terminology
******************

Terms are listed alphabetically.

.. _term_block:

block
=====

We use the term block to generically refer to a sub-tensor. Blocks can
be either :ref:`term_chip` or :ref:`term_slice` depending on how their rank
compares to that of the parent tensor.

.. _term_chip:

chip
====

A chip of a tensor is similar to a :ref:`term_slice`, but the resulting tensor
has lower rank. The distinction between chip and slice is important because of
the ambiguity associated with taking slices with extents of length one along
one or more modes. For example, say we ask for the first row of a matrix with
|n| columns. Does the user want a 1 by |n| matrix or an |n|-element vector? Chip
vs. slice resolves this ambiguity. If the user asked for row as a slice, they
get back a matrix, if they asked for the row as a chip they get back a vector.

.. _term_element:

element
=======

Tensors are typically thought of as hyper-rectangular arrays of floating-point
values. These individual values are termed "elements". Elements do not need to
be floating-point values; they can be integers, strings, or even other tensors.
The important part is that the elements form a mathematical :ref:`term_field`.
For a rank |n| tensor, an individual element can be specified by providing the
offset along each of the |n| modes. Other common names for elements include
"components".

.. _term_extent:

extent
======

The number of elements along a :ref:`term_mode`. For a vector, the extent of
the vector is the total number of elements in the vector. A matrix has two
extents: the number of rows and the number of columns. Outside TensorWrapper
other common names for extent are length and dimensionality.

.. _term_jagged:

jagged
======

.. |J| replace:: :math:`\mathbf{J}`
.. |ji| replace:: :math:`j_i`
.. |ki| replace:: :math:`k_i`

Elements of a :ref:`term_nested` tensor |J| are tensors themselves. If the
shapes of the elements of |J| differ, then |J| is jagged. Put another way, let
|ji| and |ki| be :ref:`term_slice` s of |J| along the
:math:`i`-th :ref:`term_mode`,  then if for any mode of |J| there exists
a |ji| and a |ki| with different shapes, |J| is jagged.

.. _term_mode:

mode
====

We typically think of vectors as a column (or row) of numbers, matrices are
thought of as tables of numbers (thus each entry has a row and a column
associated with it), etc. The point is typically we think of tensors as hyper-
rectangular arrays of values. "mode" is the generic term we use for referring
to a row or a column (or a higher-dimension analog). Put another way, if
specifying an element of a tensor requires specifying |n| indices, then that
tensor has |n| modes. Outside TensorWrapper other common names for mode are
dimension.

.. note::

   When discussing the literal geometry of the rectangular array in which the
   values are laid out, we will usually say "|n|-dimensional rectangular
   array." This is because when referring to literal geometric shapes (*i.e.*,
   rectangles, squares, rectangular prisms, cubes, etc.) the use of
   "mode" is not standard practice.

.. _term_nested:

nested
======

Mathematically speaking a tensor is a multilinear map over a field. Practically,
the field defines the set of values the elements of a tensor may have. While we
usually assume that field associated with a tensor is the field of real (or
complex) numbers, mathematically there is no such restrictions. Indeed, we
sometimes find it useful to use other fields (such as fields whose elements
are tensors of a :ref:`term_rank` greater than 0). We say a tensor is nested
if its elements are tensors of rank greater than 0.

.. _term_on_demand:

on-demand
=========

A tensor, or more typically a slice of a tensor, is on-demand if the elements of
the tensor are computed when needed, then discarded. On-demand tensors typically
store a callback which is capable of building a specified slice of a tensor.
The slice will exist in memory, but will not be stored in the tensor.

.. _term_rank:

rank
====

The number of :ref:`term_mode` s a tensor has. A scalar is a rank 0 tensor, a
vector is rank 1 tensor, a matrix is a rank 2 tensor, etc. Outside TensorWrapper
other common names are dimensionality and order.

.. _term_reshape:

reshape
=======

The hyper-rectangular layout of a tensor is not unique. We can just as easily
treat an |m| by |n| matrix as a vector with :math:`nm` elements. When converting
a tensor into a vector, the process is usually termed vectorization. More
generally, this process is termed reshaping as it changes the shape of the
tensor from an |m|-dimensional hyper-rectangular array to an |n|-dimensional
hyper-rectangular array (:math:`m\neq n`).

.. _term_shape:

shape
=====

The shape of a tensor is the set containing the :ref:`term_extent` of each
:ref:`term_mode`. The shape defines the edge lengths of the hyper-rectangular
array the elements are stored in. Since the number of edge lengths is the
:ref:`term_rank` of the tensor, the shape also contains that information.

.. _term_slice:

slice
=====

A sub-tensor of a tensor. A "proper" slice contains less elements than the
tensor it originates from. We require slices to have the same :ref:`term_rank`
as the original tensor. If a sub-tensor has a lower rank it is referred to as
a :ref:`term_chip`.

.. _term_smooth:

smooth
======

.. |S| replace:: :math:`\mathbf{S}`

While not a widely used term, it is helpful to introduce a term to contrast
with :ref:`term_jagged`. We define a "smooth" :ref:`term_nested` tensor to be
a tensor which is not jagged.  Put another way, let |ji| and |ki|
be :ref:`term_slice` s of |S| along the :math:`i`-th :ref:`term_mode`,
then if for all modes of |S| every pair |ji| and a |ki|
has the same shape, |S| is smooth.

****************************
Computer Science Terminology
****************************

.. _term_ast:

abstract syntax tree (AST)
==========================

With respect to source code, an abstract syntax tree (AST) is a representation
of the structure of what the programmer wrote, whereas a :ref:`term_cst`
contains the literal representation of what the programmer wrote. Carrying out
the programmed instructions is easier by traversing an AST, versus a CST,
because extraneous information has been removed.

.. _term_cst:

concrete syntax tree (CST)
==========================

With respect to source code, a concrete syntax tree (CST) is a representation
of the literal source code the programmer wrote. The CST contains all details
of the source code, including how the programmer chose to represent a
particular concept. Distilling out the essential concepts leads to an
:ref:`term_ast`.

.. _term_dsl:

domain specific language (DSL)
==============================

A domain specific language (DSL) is a coding language targeted at a particular
domain of applications. Compared to general-purpose coding languages, DSLs
tend to contain fewer language primitives on account of the DSL only concerning
itself with being general enough to express operations within the target
domain. The DSL in ``TensorWrapper`` targets the domain of tensor math and is
designed to makes it easy to express tensor operations in a performant manner.

***********************
Mathematics Terminology
***********************

.. _term_field:

field
=====

A field is a set of elements along with two operations, usually termed
addition and multiplication. Addition and multiplication behave like the
traditional addition and multiplication operations, *i.e.*, both addition and
multiplication are commutative and associative, and multiplication distributes
over addition. Finally, each non-zero element in the set must also posses an
additive and multiplicative inverse (zero elements will have only an additive
inverse).
