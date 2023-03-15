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

.. _distribution_design:

###################
Distribution Design
###################

********************************
What is a (tensor) distribution?
********************************

Each non-null tensor contains one or more elements. These elements must live
somewhere. The distribution component tells us where a particular element lives.

***************************************
Why do we need a (tensor) distribution?
***************************************

For small tensors, one stores the elements in core memory, in a vectorized
format (row or column major) and an object stating where the element lives is
superfluous. For larger tensors, sparse tensors, and/or tensors we don't want
in core memory, keeping track of where elements lives is more complicated.

**********************************
Tensor distribution considerations
**********************************

#. Which tiles are owned by which processes?

   - For replicated tensors every process owns each tile.
   - For distributed tensors each process only owns some tiles.

#. Which tiles are lazy vs eager?

   - Eager tiles are built immediately and stored.
   - Lazy tiles are built on-the-fly and then deleted.
   - Theoretically can mix lazy and eager.

#. Hardware owning each tile.

   - Generalization of 1st consideration.
   - E.g., which tiles live on which GPUs?
