.. _tw_designing_the_buffer:

####################
Designing the Buffer
####################

*******************
What is the Buffer?
*******************

TensorWrapper is designed to wrap existing tensor implementations under a
unified API. We assume that each of those existing tensor implementations
already has some sort of data structure. The ``Buffer`` class is primarily
intended to type-erase those data structures, while still providing readily
accessible.


*********************
Buffer Considerations
*********************


.. _b_type_erasure:

Type erasure
   ``Buffer`` objects are primarily responsible for type-erasing the backend
   being used. Interactions among ``Buffer`` objects occur in a type-erased
   manner.

   - As a corollary, we need to be able to unwrap the type-erased object too.

.. _b_data_structure:

Data structure
   Buffer is primarily envisioned as the fundamental data structure behind the
   TensorWrapper DSL. As a data structure ``Buffer`` should have a fairly
   minimal set of operations. Advanced operations are done by unwrapping the
   ``Buffer`` and interacting with the wrapped tensor directly.

.. _b_fundamental_methods:

Fundamental methods
   Somewhat of a corollary to :ref:`_b_data_structure`, but all backends must,
   in theory, be able to implement all methods defined on the ``Buffer`` class.
   The methods should thus be very fundamental tensor operations. In practice,
   we can always throw if a backend doesn't implement a method, but we want to
   minimize such methods.

   - Methods should be read-only to avoid having to modify the underlying state
     of the backend.

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
     ``Allocator`` component. See :ref:`tw_desinging_the_allocator` for
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

Buffer Operations
=================

Once you have a ``Buffer`` you can inspect some basic properties:

.. code-block:: c++

   auto buffer = get_buffer();

   // Get the total number of elements in the buffer
   auto n_elements = buffer.size();

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
type ``unwrapped_tensor_type`` or throws.

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
