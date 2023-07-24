
.. _designing_tensor_wrapper_class:

#############################
Designing TensorWrapper Class
#############################




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
