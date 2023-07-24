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

.. _sparsity_design:

######################
Tensor Sparsity Design
######################

This page documents the process of designing the sparsity component of
TensorWrapper.

************************
What is tensor sparsity?
************************

In the context of tensors, sparsity refers to tensors possessing elements which
are effectively zero. Exactly what defines the effective zero is situational
and problem-specific. The sparsity component is responsible for indicating
which elements are zero or non-zero.

For our purposes tensor sparsity comes in two types: tile and element sparsity.
Element sparsity is the more fundamental sparsity and indicates which elements
of the tensor are zero/non-zero. Tensors are typically tiled, and under the
hood we are thus more interested in tile sparsity, which is whether or not a
tile contains non-zero elements.

*******************************
Why do we need tensor sparsity?
*******************************

If a tensor exhibits a substantial amount of sparsity, then implicitly storing
the zero values leads to a significant space savings. Furthermore, many
operations can be simplified based on whether or not the operation involves
sparsity, *e.g.*, Contracting a non-zero tile with a zero tile results in a
zero-tile (which if the resulting tile is implicitly stored means the
contraction is a no-op). Thus properly exploiting sparsity can result in
substantial space and time savings.

***********************
Sparsity considerations
***********************

#. Element sparsity.

   - TensorWrapper APIs are element-based for user-friendliness.
   - Non-zero tiles can still have element sparsity.

#. Tile sparsity.

   - Most implementations will work with tile sparsity.
   - Convertible to element sparsity with Shape.

#. Interact with sparse maps.

   - Running modes through a series of sparse maps produces sparsity object.

#. Potentially recursive structure.
