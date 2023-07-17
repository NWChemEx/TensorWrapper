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

.. _shape_design:

###################
Tensor Shape Design
###################

This page captures the design process of TensorWrapper's ``Shape`` class.

*************************
What is a tensor's shape?
*************************

.. |n| replace:: :math:`n`

For computing purposes tensors are really nothing more than a bunch of floating
point values and meta-data associated with those values. Conceptually, the
floating point values are typically arranged into |n|-dimensional rectangular
arrays, where |n| is the number of modes in the tensor. A tensor's shape
represents how the values are conceptually, and physically, laid out.

********************************
Why do we need a tensor's shape?
********************************

A tensor's shape is arguably the most primitive meta-data associated with the
tensor. Without the shape of the tensor we do not know how to access elements
or lay them out in memory.

********************
Shape Considerations
********************

Rank and extents
   The main data in the shape is the :ref:`term_rank` and the
   :ref:`term_extent`s of each mode.

Nested
   While a :ref:`term_nested` tensor may seem exotic, in practice, distributed
   tensors are often implemented by nesting (ideally the user need not be aware
   of such nesting aside from possibly specifying it at construction). Nesting,
   also occurs naturally when discussing sparsity.

   - Nestings may be :ref:`term_smooth`  or :ref:`term_jagged`

Logical vs actual
   The user declares the tensor with some shape. That shape usually reflects the
   physical problem being modeled. Internally we may need to store the tensor
   as a different shape, for performance reasons. The shape describing how the
   user wants to interact with the tensor is the "logical" shape.

   - The user should interact with the tensor as if it had the external shape.
   - We should allow the user to override the internal shape if need be.

Non-integral indices
   Sometimes it is useful to index modes with something other than an offset.
   For example, one may want to name sub-ranges and refer to blocks by those
   names.

Symmetry
   In the generalized sense.

*************
Proposed APIs
*************
