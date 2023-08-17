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


.. _designing_tensor_wrapper_class:

#############################
Designing TensorWrapper Class
#############################

The point of this page is to document the design process of the TensorWrapper
class.

********************************
What is the TensorWrapper class?
********************************




*************
Proposed APIs
*************

Easiest Non-Trivial Construction
================================

.. code-block:: c++

   // Makes a 10 by 20 by 40 rank 3 tensor initialize with 0.0
   TensorWrapper t({10, 20, 40}, 0.0);

Under the hood this looks like:

.. code-block:: c++

   Shape shape(10, 20, 40);
   Allocator a;
   TensorWrapper t(a.allocate(shape));


Where ``Shape`` and ``Allocator`` are the default types (to be decided later).

Declaring a Distributed Tensor
==============================

.. code-block:: c++

    TiledShape shape({2, 4, 6, 8, 10}, {10, 20}, {10, 20, 30, 40});
    TensorWrapper t(shape); // Dispatches to TensorWrapper(Buffer)

c("mu", "occ", "i")
