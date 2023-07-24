.. _tw_designing_the_buffer:

####################
Designing the Buffer
####################

*******************
What is the Buffer?
*******************

To vastly over simplify, tensors consist of two things: the literal elements of
the tensor and all the additional properties and mathematical structure imposed
on top of those elements. The ``Buffer`` class is responsible for holding the
literal elements of the tensor and being able to describe physically how those
elements are held.

************************
Why do we need a Buffer?
************************

TensorWrapper is ultimately powered by other tensor libraries. The boundary
between those libraries and TensorWrapper is the ``Buffer`` class. The
``Buffer`` class is the interface through which the user's intentions (specified
with the TensorWrapper DSL) are conveyed to the backend.

******************
Buffer Terminology
******************

distributed
   A buffer is distributed if it has both local and remote pieces. By contrast
   remote buffers only contain remote data (no local data).

future (to a buffer)
   A future to a buffer is an object which will eventually be a buffer, but at
   the time of creation may not actually have its data yet. Futures to buffers
   typically arise when a task scheduler is creating buffers and we do not want
   to wait for the scheduler to create the buffer. In this case a backgrounded
   task for creating the buffer is added to the scheduler and control
   continues with only a future to the result. Once the backgrounded task
   has completed creating the buffer, the buffer can be accessed directly
   from the future to the buffer. If control requests the buffer before it has
   been created, then control must wait until the buffer is ready before
   continuing.

lazy
   A buffer is lazy if it does not store the values which live in it, but
   instead creates them on-the-fly. This differs from a future in that the
   values of a lazy tensor are "immediately available" (after the delay required
   to compute them).

local
   A buffer is "local" if the current process can access the state of the
   buffer without communicating with another process. A local buffer is the
   opposite of a remote buffer

remote
   A buffer is "remote" if none of the state can be accessed without
   communicating with another process. A remote buffer has no local piece (if it
   has a local piece it is a distributed buffer).

*********************
Buffer Considerations
*********************

.. _b_wrapping_other_tensor_libraries:

Wrapping other tensor libraries.
   In practice the literal data layouts of a tensor can be very complicated.
   Many existing tensor libraries have already optimized tensor operations on
   their data structures for some particular scenarios. We do not want to
   reinvent those optimizations and suggest ``Buffer`` should actually wrap
   those libraries' data structures.

.. _b_type_erasure:

Type erasure
   As consideration :ref:`b_wrapping_other_tensor_libraries` states, the
   ``Buffer`` objects do not just hold ``double*``, but rather tensor-like
   objects from existing tensor libraries. To allow the various tensor libraries
   to be somewhat interoperable we propose that the ``Buffer`` objects type-
   erase the backend they hold.

   - We will need to be able to unwrap the type-erased object too.
   - The set of methods exposed by the ``Buffer`` class needs to be very
     general so as to have analogs in every possible backend.

.. _b_data_location:

Data location
   How a user interacts with a ``Buffer`` depends on where the data lives.
   Data that can only be seen by the current process is usually treated
   differently than data which all processes can see.

   - Need to know if data is distributed, local, or replicated.
   - For distributed data, need to be able to replicate, get handles to remote
     data, and access local data.

.. _b_implicit_data:

Implicit data
   Particularly for high-rank tensors, we often do not store data explicitly.
   The ``Buffer`` must be able to work seamlessly when data is stored
   implicitly, also known as lazily.

   - In practice, implicit tensors usually need to store some state.

.. _b_asynchronous_support:

Asynchronous support
   We want the creation of a distributed tensor to be asynchronous. For this
   to work, we need the ability to have proxy ``Buffer`` objects. Such objects
   will eventually be filled in, but in the meantime control can continue to
   use them to build up an operation queue. Attempting to access such a
   ``Buffer`` results in waiting until the ``Buffer`` has been filled in.

.. _b_op_graph_evaluation:

OpGraph evaluation
   Once TensorWrapper has figured out what the user wants to do, it needs to
   tell the backend to do it. This process involves recording the operations to
   be done in an ``OpGraph`` object and passing that object to the buffer the
   result is to be assigned to.

Out of Scope
============

Tensor math
   While mathematical operations on tensors are arguably fundamental, the
   reality is each backend is going to approach those operations in a different
   manner. Attempting to unify these operations would be difficult. In our
   opinion a better solution is to queue up a set of operations to be done and
   then tell the backend to do them.

   - Evaluating a set of operations is in scope and is covered by
     :ref:`b_op_graph_evaluation`

Backend Allocation
   Literally making an object of the backend is a fundamental tensor operation;
   however, like "Tensor math" above, the creation of a backend object is
   going to be heavily dependent on the identity of the backend.

   - The responsibility for allocating ``Buffer`` objects is punted to the
     ``Allocator`` component. See :ref:`tw_designing_the_allocator` for
     more details.


*************
Proposed APIs
*************

Creating a Buffer
=================

Creating a ``Buffer`` is done through an allocator, the exact details for how
to create an allocator are beyond our current scope. For now we treat it as an
opaque type. Creation of a ``Buffer`` requires providing the allocator a
``Shape`` object (which is also opaque for our current purposes) in one of two
ways. The first invocation is just the shape:

.. code-block:: c++

   auto shape     = get_shape();
   auto allocator = get_allocator();
   auto buffer    = allocator.allocate(shape);

This invocation is suitable for initializing a ``Buffer`` to assign to.
Initializing a ``Buffer`` with actual data is done by also passing the allocator
a lambda like:

.. code-block:: c++

   auto shape     = get_shape();
   auto allocator = get_allocator();
   auto buffer    = allocator.allocate(shape, lambda_fxn);

The exact syntax of the lambda is an ``Allocator`` consideration.

Buffer Methods
==============

Once you have a ``Buffer`` you can inspect some basic properties:

.. code-block:: c++

   auto buffer = get_buffer();

   // Get the shape of the buffer
   auto shape = buffer.shape();

   // Get an enum representing the scalar elements of the buffer
   // N.B. Buffer also type erases this information
   auto scalar_type = buffer.element_type();


Retrieving the Wrapped Tensor
=============================

Until TensorWrapper is fleshed out we anticipate that users will need to
unwrap the buffer somewhat regularly we propose that this is done by:

.. code-block:: c++

   // Get the buffer object we want to unwrap
   auto buffer = make_buffer();

   // The type the backend uses as a tensor, e.g. for Eigen:
   using unwrapped_tensor_type = eigen::Tensor<double, 3>;

   auto eigen_t = Converter<unwrapped_tensor_type>::convert(buffer);

The ``Converter`` class is responsible for determining if the type-erased value
inside the ``Buffer`` is already of type ``unwrapped_tensor_type``. If it is
it just returns it; if it is not, then it either converts it to an object of
type ``unwrapped_tensor_type`` or throws an error.

Working with Distributed Buffers
================================

``DistributedBuffer`` extends the ``Buffer`` class to the scenario when the
underlying ``Buffer`` object has data potentially distributed across multiple
processes.

.. code-block:: c++

   auto dist_buffer = get_buffer();

   // Gets a handle to the part of the distributed buffer which is local to
   // the current process
   LocalBuffer my_buffer = dist_buffer.local_buffer();

   // Gets a handle to a part of the distributed buffer whose state is not
   // local to the current process. N.B. this does NOT make the data local
   // yet. We do assume that every process knows how to do this with no
   // communication though
   RemoteBuffer a_buffer = dist_buffer.at(range);

   // Actually pulls the data
   LocalBuffer = a_buffer.get();

   // To make the distributed buffer replicated
   ReplicatedBuffer replicated_buffer(dist_buffer);

Evaluating an OpGraph
=====================

When a series of operations are assigned to an ``AnnotatedTensor``,  this
triggers the creation of an ``OpGraph`` object. For our current purposes an
``OpGraph`` object is opaque (see :ref:`tw_designing_the_opgraph` for the full
design specification). For designing the ``Buffer`` the important part to note
is that the ``OpGraph`` must contain all necessary information about the
operation the backend needs to perform.

.. code-block:: c++

   auto buffer = get_buffer();

   auto graph = get_op_graph();

   buffer.compute(graph);


N.B. that we can use the visitor pattern to automatically downcast the buffer

*************
Buffer Design
*************

.. figure:: assets/buffer.png
   :align: center

   Design of the buffer.
