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

.. _tw_designing_the_allocator:

#######################
Designing the Allocator
#######################

The point of this page is to capture the design process of the

*********************
What is an Allocator?
*********************

In C++, allocators are objects used by containers to secure uninitialized
storage for the elements of the container. Optionally, the allocator may also
initialize the storage. In TensorWrapper, an allocator is responsible for
securing a (possibly uninitialized) instance of the backend's tensor object.

**************************************
Why do We Need an Allocator Component?
**************************************

Using TensorWrapper, users will create objects for describing a tensor's shape,
symmetries, and sparsity. Those objects will then be converted into a layout.
Given the target layout, TensorWrapper will then create a tensor object from
one of the available backends. Since each of these libraries has their own
tensor class (or classes), with their own construction methods, the allocator
component is needed to hide the process of constructing the backend's buffer
objects.

************************
Allocator Considerations
************************

.. _a_backend_aware:

Backend aware
   The allocator's primary purposes are to provide a mechanism for selecting
   the tensor backend and to wrap the process of creating instances for that
   backend.

   - Eventually we would like to automate the backend selection process.
     Ideally, TensorWrapper would know which backend works best in which
     situations and would choose it for the user. This can be done by having a
     super allocator which delegates to the individual backend allocators.

.. _a_runtime_aware:

Runtime aware
   The ``Allocator`` is charged with literally allocating ``Buffer`` objects.
   In order to do this in a performant manner, the ``Allocator`` will need
   access to runtime information such as the number of processes and the
   amount of available memory.

   - Tracking the state of the runtime is the responsibility of ParallelZone.
   - Not all backends resource manage, TensorWrapper will have to do it for
     them.

.. _a_initialization:

Initialization
   Before any tensor operations can be performed some tensors will need to be
   filled in with values. Sometimes the tensor is simply an identity or zero
   tensor, but more often the initial value of an element depends on its
   indices. The ``Allocator`` must have a mechanism for filling in blocks of the
   ``Buffer`` with arbitrary values.

.. _a_propagation:

Propagation
   Tensors contain the ``Allocator`` used to create their ``Buffer``. Reuse of
   that ``Allocator`` will create additional ``Buffer`` objects which rely on
   the same backend. In turn, by grabbing the ``Allocator`` associated with a
   tensor, users can create additional ``Buffer`` objects which are guaranteed
   to be compatible with the tensor backend. This requires the ``Allocator`` to
   propagate through expressions.

.. _a_rebind:

Rebind
   C++ allocators are associated with a specific type. If you want to use the
   allocator to allocate memory for a different type, you have to "rebind" the
   allocator (the mechanism for doing this changed in C++20, but boils down to
   using template meta-programming to work out the type of the other allocator).
   In TensorWrapper's use case, many of the various backends will be able to
   make different types of buffers (*e.g.*, distributed libraries will usually
   be able to minimally make ``DistributedBuffer`` and ``ReplicatedBuffer``
   objects). Being able to rebind TensorWrapper allocators thus allows us to
   create other buffer types which are still compatible with the backend.

.. _a_downcasting:

Downcasting
   Since allocators are tied to a particular type, given the allocator which
   made a buffer, the allocator should know what the actual (not type-erased)
   type of the buffer is.

   - The main use case here is for converting between backends by using the
     allocators to establish which backends the buffers contain.
