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

For computing purposes, tensors are really nothing more than a bunch of floating
point values and meta-data associated with those values. Conceptually, the
floating point values are typically arranged into |n|-dimensional rectangular
arrays, where |n| is the number of :ref:`term_mode` s in the tensor. A tensor's
shape describes this hyper-rectangular array's layout.

********************************
Why do we need a tensor's shape?
********************************

A tensor's shape is arguably the most primitive meta-data associated with the
tensor. Without the shape of the tensor we do not know how to access elements
or lay them out in memory.

********************
Shape Considerations
********************

Rank and extents
   The main data in the shape is the :ref:`term_rank` and the
   :ref:`term_extent` s of each mode.

   - Runtime determination of extents is a very common scenario.
   - Usually know the rank at compile time.

Nested
   While a :ref:`term_nested` tensor may seem exotic, in practice, distributed
   tensors are often implemented by nesting (ideally the user need not be aware
   of such nesting aside from possibly specifying it at construction). Nesting,
   also occurs naturally when discussing sparsity.

   - Nestings may be :ref:`term_smooth`  or :ref:`term_jagged`

Combining shapes
   As we do tensor operations we will need to work out the resulting shapes.
   This in general requires knowing the indices of the tensor.

Not in Scope
============

Sparsity
   The ``Shape`` component is targeted at describing the conceptual layout of
   the hyper-rectangular array of values. The conceptual layout is independent
   of the values of the elements. Sparsity is concerned with knowing which
   elements are zero.

   - Sparsity is punted to :ref:`sparsity_design`.


Permutational Symmetry
   In many cases the elements of a tensor are not all linearly-independent and
   optimizations are possible by avoiding redundant computation.

   - Antisymmetry, Hermitian, and anti-Hermitian all fall into this
     consideration too.
   - Symmetry is punted to :ref:`tw_designing_the_symmetry_component`.

Logical vs actual
   The user declares the tensor with some shape. That shape usually reflects the
   physical problem being modeled. Internally we may need to store the tensor
   as a different shape, for performance reasons. The shape describing how the
   user wants to interact with the tensor is the "logical" shape.

   - Both the logical and actual shapes are ``Shape`` objects.
   - It is the responsibility of the user creating ``Shape`` objects to track
     if they represent logical or actual shapes.


*************
Proposed APIs
*************

Constructing a ``Shape``
========================

Creating a non-nested shape just requires knowing the extent of each mode:

.. code-block:: c++

   Shape null_shape;              // No rank and no elements
   Shape rank0_shape{};           // A scalar
   Shape rank1_shape{10};         // 10 element vector
   Shape rank2_shape{10, 20};     // 10 by 20 matrix
   Shape rank3_shape{10, 20, 30}; // 10 by 20 by 30 tensor

Note that following usual C++ rules the first two lines actually call
different constructors (default ctor vs. initializer list). Using an initializer
list requires us to know the rank at compile time. If we want to determine the
rank at runtime we can use iterators:

.. code-block:: c++

   // Somehow create a vector of extents
   using size_type = Shape::size_type;
   std::vector<size_type> extents = get_extents();

   // Construct Shape from iterator pair
   Shape runtime_rank_shape(extents.begin(), extents.end());

Constructing a ``NestedShape``
==============================

Creating a :ref:`term_smooth` nested shape requires knowing the shape of
each layer's elements. To create two-layered smooth shapes:

.. code-block:: c++

   // Null shape (no nesting, no ranks)
   NestedShape null_nested;

   // A scalar viewed as a two-layer tensor
   NestedShape rank0_0{{}, {}};

   // A 10-element vector viewed as a two-layer tensor (index in layer 1)
   NestedShape rank0_1{{}, {10}};

   // A 10-element vector viewed as a two-layer tensor (index in layer 0)
   NestedShape rank1_0{{10}, {}};

   // Matrix viewed as a 10-element vector whose elements are 20-element vectors
   NestedShape rank1_1{{10}, {20}};

   // Same matrix viewed as a two-layer tensor with both indices in layer 1
   NestedShape rank0_2{{}, {10, 20}};

   // Same matrix viewed as a two-layer tensor with both indices in layer 0
   NestedShape rank2_0{{10, 20}, {}};

   // 10-element vector with 20 by 30 element matrices as elements
   NestedShape rank1_2{{10}, {20, 30}};

   // 10 by 20 element matrix with 30 element vectors as elements
   NestedShape rank2_1{{10, 20}, {30}};

   // 10 by 20 element matrix with 30 by 40 element matrices as elements
   NestedShape rank2_2{{10, 20}, {30, 40}};

Admittedly it's not immediately clear why one would want to be able to define
shapes where one layer is rank 0 (maybe just to get the number of layers to
line up?), but as shown it's no problem with our syntax. This easily
generalizes to more layers:

.. code-block:: c++

   // A matrix whose elements are matrices of matrices
   NestedShape rank2_2_2{{10, 20}, {30, 40}, {50, 60}};

In practice the inner initializer lists are used to initialize ``Shape``
objects so the previous initialization is equivalent to:

.. code-block:: c++

   // 10 by 20 element matrix with 30 by 40 element matrices as elements
   NestedShape rank2_2_2{Shape{10, 20}, Shape{30, 40}, Shape{50, 60}};

Like ``Shape``, we can determine the number of layers and ranks of each layer
at runtime using a range ctor:

.. code-block:: c++

   // Somehow get the extents for each layer
   using shape_type = NestedShape::shape_type;
   std::vector<shape_type> extents_per_layer = get_extents();

   // Make the shape from iterator pairs
   NestedShape runtime_layered(extents_per_layer.begin(),
                               extents_per_layer.end());

Jagged Construction
===================

:ref:`term_smooth`  tensor can be thought of as a special case of a
:ref:`term_jagged`` tensor. For a
.. code-block:: c++

   // For brevity define variables
   Shape s10({10}), s20({20}), s30({30});
   Shape s10_20({10, 20}), s30_40({30, 40}), s50_60({50, 60});
   Shape s10_20_30({10, 20, 30}), s40_50_60({40, 50, 60});

   // No elements, no rank
   JaggedShape null_shape;

   // A "jagged" scalar (only a single element, so it's also smooth)
   JaggedShape rank0_shape({});

   // A "jagged" vector (same as a smooth vector)
   JaggedShape rank1_shape({s10});

   // A jagged matrix with 3 rows; row 0 has 10 elements, row 1 has 20, row 2 30
   JaggedShape rank2_shape({s10, s20, s30});

   // A jagged rank 3 tensor with smooth matrices. Matrix 0 is 10 by 20,
   // matrix 1 is 30 by 40, and matrix 2 is 50 by 60
   JaggedShape rank3_shape({s10_20, s30_40, s50_60});

   // A jagged rank 3 tensor where elements are jagged matrices. Matrix 0 is
   // 1 by 10, matrix 2 has 20 columns in row 0 and 30 columns in row 2, and
   // matrix 3 has 30 columns in row 0, 10 columns in row 1, and 20 columns in
   // row 2
   JaggedShape rank3_shape2({{s10},
                             {s20, s30},
                             {s30, s10, s20}});

    // A jagged rank 4 tensor where the 0-th element of the 0-th mode is a
    // 10 by 20 by 30 smooth tensor and the 1-st element is a 40 by 50 by 60
    // smooth tensor
   JaggedShape rank4_shape({s10_20_30, s40_50_60});

   // A jagged rank 4 tensor where the elements are jagged rank 3 tensors.
   // Taking slices along the 0 and 1-st modes, the (0,0)-th slice is a 10 by 20
   // matrix, the (0,1)-th slice is a 30 by 40 matrix, the (1,0)-th slice is
   // a 30 by 40 matrix, the (1,1)-th slice is a 10 by 20 matrix, and the
   // (1,2)-th slice is a 50 by 60 matrix
   JaggedShape rank4_shape2({{s10_20, s30_40},
                             {s30_40, s10_20, s50_60}});

   // A jagged rank 4 tensors where the elements are jagged rank 3 tensors,
   // which have jagged matrices for elements. Taking slices along the 0, 1, and
   // 2 modes we have:
   // - (0,0,0) is a 10 element vector,
   // - (0,1,0) is a 20 element vector,
   // - (0,1,1) is a 30 element vector,
   // - (1,0,0) is a 10 element vector,
   // - (1,0,1) is a 30 element vector,
   // - (1,1,0) is a 20 element vector,
   // - (1,2,0) is a 10 element vector,
   // - (1,2,1) is a 20 element vector,
   // - (1,2,2) is a 30 element vector
   JaggedShape rank4_shape3({{{s10}, {s20, s30}},
                             {{s10, s30}, {s20}, {s10, s20, s30}}});

In general a ``JaggedShape`` is an initializer list of ``NestedShape`` objects.

.. code-block:: c++

   JaggedShape rank3_shape2({NestedShape({s10}), NestedShape({s20, s30}),
                             NestedShape({s30, s10, s20}))

Because the 0-th mode can not be jagged, it takes a minimum of two modes to be
truly jagged.

Shape Operations
================
