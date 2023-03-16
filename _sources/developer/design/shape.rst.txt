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
point values. These floating point values are typically arranged into
|n|-dimensional rectangular arrays, where |n| is the number of modes in the
tensor. A tensor's shape represents the layout of the |n|-dimensional
rectangular array.

********************************
Why do we need a tensor's shape?
********************************

A tensor's shape is arguably the most primitive information about the tensor.
Without the shape it is not possible to even begin laying out the tensor in
memory.

********************
Shape Considerations
********************

#. Element-based API.

   - Algorithms interacting with the DSL reason in terms of elements, not tiles.
   - Tiling is largely an implementation detail needed for performance.
   - Internally, easy to map element-to-tile and vice versa.

#. Contains the number of modes.
#. Contains the extent of each mode.
#. Contains the tiling of each mode.

   - Performance requires some sort of tiling.
   - Tiling is not strictly restricted to distributed computing.

#. Shape may need to be recursive.

   - Elements of tensors typically thought of as scalars, but can be any field.
   - Need to be able to get the shape of the field's elements.
