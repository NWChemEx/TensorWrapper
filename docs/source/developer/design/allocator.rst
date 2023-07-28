.. _tw_designing_the_allocator:

#######################
Designing the Allocator
#######################

************************
Allocator Considerations
************************

.. _a_runtime_aware:

Runtime aware
   The ``Allocator`` is charged with literally allocating ``Buffer`` objects.
   In order to do this in a performant manner, the ``Allocator`` will need
   access to runtime information such as the number of processes and the
   amount of available memory.

   - Tracking the state of the runtime is the responsibility of ParallelZone.

.. _a_input_initialization:

Input initialization
   Before any tensor operations can be performed some tensors will need to be
   filled in with values. Sometimes this an identity or zero tensor, but more
   often the initial value of an element depends on its indices. The
   ``Allocator`` must have a mechanism for filling in blocks of a ``Buffer``
   with arbitrary values.

.. _a_result_initialization:

Result initialization
   The ``Allocator`` will need to be able to initialize a ``Buffer`` in a way
   that is suitable for assignment. In general, this is a more lightweight
   initialization than input initialization.

.. _a_buffer_hierarchy_support:

Buffer hierarchy support
   Since ``Buffer`` objects are created by ``Allocator`` objects, the
   ``Allocator`` object must be aware of the ``Buffer`` hierarchy in order to
   create instances of
