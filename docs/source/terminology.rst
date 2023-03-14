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

******************
Tensor Terminology
******************

Extent
======

The number of elements along a mode. For a vector, the extent of the vector is
the total number of elements in the vector (also called its length). A matrix
has two extents: the number of rows and the number of columns.

Mode
====

Colloquially the number of indices required to specify an element of the tensor.
A scalar is a 0-mode tensor, a vector is a 1-mode tensor, a matrix is a 2-mode
tensor, etc.
