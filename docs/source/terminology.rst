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

When discussing tensors, different researchers use different terminology. This
section provides a glossary of the tensor terminology we will use throughout
TensorWrapper.

******************
Tensor Terminology
******************

Terms are listed alphabetically.

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

Elements of a :ref:`nested` tensor :math:`J` are tensors themselves. If the
shapes of the elements of the :math:`J` can NOT be written as a Cartesian
product of tilings, we say :math:`J` is jagged. Put another way, let
:math:`j_i` and :math:`k_i` be :ref:`term_slice`s of :math:`J` along the
:math:`i`-th :ref:`term_mode`,  then if for any mode of :math:`J` there exists
a :math:`j_i` and a :math:`k_i` with different shapes, :math:`J` is jagged.

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
are tensors of a :ref:`term_rank`` greater than 0). We say a tensor is nested
if its elements are tensors of rank greater than 0.

.. _term_rank:

rank
====

The number of :ref:`term_mode`s a tensor has. A scalar is a rank 0 tensor, a
vector is rank 1 tensor, a matrix is a rank 2 tensor, etc. Outside TensorWrapper
other common names are dimensionality and order.

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
tensor it originates from. We use the term slice for any sub-tensor regardless
of whether the sub-tensor has the same :ref:`term_rank` as the original tensor.

.. _term_smooth:

smooth
======

While not a widely used term, it is helpful to introduce a term to contrast
with :ref:`term_jagged`. We define a "smooth" :ref:`term_nested` tensor to be
a tensor which is not jagged.  Put another way, let :math:`j_i` and :math:`k_i`
be :ref:`term_slice`s of :math:`S` along the :math:`i`-th :ref:`term_mode`,
then if for all modes of :math:`S` every pair :math:`j_i` and a :math:`k_i`
has the same shape, :math:`S` is smooth.
