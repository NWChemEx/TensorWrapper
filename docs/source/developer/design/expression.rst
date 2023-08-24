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

.. _designing_the_expression_component:

##################################
Designing the Expression Component
##################################

This page documents the process of designing the expression component of
TensorWrapper.

*********************************
What is the expression component?
*********************************

Users of a tensor library have two main needs: to create tensors and to solve
tensor expressions involving those tensors. The expression component contains
the pieces of the :ref:`term_dsl` required to write succinct tensor expressions.

***************************************
Why do we need an expression component?
***************************************

Most libraries tend to be eagerly evaluated, meaning code is evaluated as it is
encountered. For simple, easily optimized code such a strategy is fine.
Unfortunately, tensor expressions tend to be significantly harder to optimize,
plus the optimal evaluation tends to be highly runtime dependent. A potential
solution to this problem is to evaluate the code lazily, instead of eagerly.
Lazily evaluated code is "recorded" until it is needed. Once it is needed it
is played back.

Most tensor operations are steps in a bigger algorithm. By lazily
evaluating tensor computations we can register the user's intention and
optimize our evaluation strategy accordingly. The point of the expression layer
is to provide a user-friendly DSL for lazy evaluation of tensors.

.. _ec_considerations:

***********************************
Expression Component Considerations
***********************************

.. _ec_compose_multiple_objects:

compose multiple objects
   While our goal is to ultimately be able to compose ``TensorWrapper`` objects,
   we can decompose the work needed to do so by ensuring we can compose the
   pieces of the ``TensorWrapper`` object. In particular we want to be able to
   compose objects from the shape, symmetry, sparsity, and allocator components.

.. _ec_generalized_einstein_notation:

generalized Einstein notation
   Many of the most common tensor operations are succinctly specified using
   generalized Einstein notation (indices appearing on the right side of
   an equation, but not the left are summed over).

   - There are some gotchas (see :ref:`eistein_summation_convention`).
   - Math operations include: addition, subtraction, scaling, multiplication,
     and division. The latter is missing from a number of existing tensor
     libraries.
   - Note that multiplication actually covers a number of operations including:
     element-wise product, matrix multiplication, contraction, trace, and direct
     product.
   - Could actually enable an alternative syntax with explicit summations.

.. _ec_non_einstein_math:

non-Einstein math
   There are a number of tensor operations which are not easily expressed using
   generalized Einstein notation, but are still important for tensor algebra.
   In particular:

   - Eigen system and linear equation solvers.
   - Factorizations/decompositions: Cholesky, QR, SVD.
   - Slicing/chipping
   - Matrix powers

.. _ec_template_meta_programming_cost:

template meta-programming cost
   In C++ lazy evaluation is typically implemented by using the template meta-
   programming technique known as expression templates. Ultimately expression
   templates map different expressions to different types. The type of an
   expression is usually a heavily nested instantiation of a template class,
   which can lead to significant compiler overhead (as well as nearly
   undecipherable compiler errors, though C++20 concepts helps a lot in this
   regard).

.. _ec_reusable_intermediates:

reusable intermediates
   In most tensor libraries the expression layer goes from a series of sources
   to a single sink. Along the way a series of temporary, unnamed intermediates
   is created. If any of these intermediates is common to more than one
   expression we want to make sure those intermediates are only formed once.

   - *N.b.* while an expression like ``A("i,j,k") + B("i,j,k")`` is clearly an
     intermediate, so is ``A("i,j,k")``. This is because ``A("i,j,k")``
     corresponds to tensor access of the "i,j,k"-th element and may be
     non-trivial (also note that behind the scenes, for performance,
     TensorWrapper may map the element access to "slice access followed by
     element access").

sparse maps
   The sparsity component :ref:`sparsity_design` realized that sparse maps are
   a mechanism for creating sparsity objects and thus should live above the
   sparsity component. Sparse maps are most naturally expressed as
   relationships among dummy indices, that is given an expression like
   ``C("i,j") = A("i,j,a,b") * B("i,j,b,a");`` we would define a sparse map, say
   which maps a given ``i,j`` pair to a set of corresponding, non-zero ``a,b``
   pairs. From that sparse map we can work out the sparsity of ``A``, ``B``,
   and ``C``. The trick is we need to know how the sparse map's modes map to
   the modes of ``A``, ``B``, and ``C``. This information is
   available in the expression layer.

Out of Scope
============

expression optimization
   The goal of the expression layer is simply to capture the information needed
   to perform the calculation, not to optimize the calculation, or to perform
   it. Optimizing the calculation can be done using the OpGraph component (see
   :ref:`tw_designing_the_opgraph`). Running the calculation is done by the
   backend of the Buffer component, given an OpGraph object (see
   :ref:`tw_designing_the_buffer`).

finding common intermediates
   While consideration :ref:`reusable_intermediates` concerns the expression
   layer being able to annotate common intermediates, actually finding said
   common intermediates is much harder. For now it is the user's job to identify
   common intermediates.

***************************
Expression Component Design
***************************

***********
Example API
***********

.. note::

   The examples in this section purposely use the real types from the expression
   layer. This is NOT what we expect a user to do. What a user sees is shown
   later (see :ref:`expression_user_api`).


The expression layer works basically the same for every composable object of
type ``T`` (``T`` being things like ``Shape``, ``Symmetry``, ``TensorWrapper``)
so we avoid specifying the value of ``T``.

.. _expression_construction:

Construction
============

Following from the :ref:`ec_generalized_einstein_notation` consideration we
expect that most users will enter into the expression layer by adding dummy
indices to an object. C++ wise this looks like:

.. code-block:: c++

   // Assume we have some T objects
   T a, b, c, d, e, g; // No f b/c variable would be "if"

   Indexed<T> ia = a("i,j,k");
   Indexed<T> ib = b("i,j,k");
   Indexed<T> ic = c("i,j,k");
   Indexed<T> id = d("i,j,k");
   Indexed<T> ie = e("i,j,k");
   Indexed<T> ig = g("i,j,k");

The ``Indexed<T>`` objects will then be composed pair-wise to form
``BinaryExpression<T>`` objects.

.. code-block:: c++

   // continues from last code block
   Addition<T> iapib = ia + ib;
   Multiplication<T> iatib = ia * ib;
   Subtraction<T> iasib = ia - ib;
   Division<T> iadib = ia / ib;

Note that unlike traditional expression templates which would end up with
types like ``Addition<Indexed<T>, Indexed<T>>`` we rely on the fact
the all of the pieces derive from ``Expression<T>``, which helps us address
consideration :ref:`ec_template_meta_programming_cost`.

Once we have built up terms they get assigned to an ``Indexed<T>`` object like:

.. code-block:: c++

   // continues from last two code blocks

   ic = iapib; // Assigns results of addition to C
   id = iatib; // Assigns results of multiplication to C
   ie = iasib; // Assigns results of subtraction to C
   ig = iadib; // Assigns results of division to C

It is worth noting, that it is somewhat trivial to satisfy consideration
:ref:`ec_reusable_intermediates` when interacting with the expression layer
directly. This is because each expression object is actually a node in the
:ref:`term_cst`, so by reusing the literal nodes we reuse the intermediates.

Obtaining an OpGraph
====================

The trick to avoid the nasty nested expression templates is to obtain the final
``OpGraph`` object via the base class's ``Expression<T>`` API. This can be
done via the visitor pattern and looks something like:

.. code-block:: c++

   // In practice e would be a pointer b/c Expression is an abstract base class
   Expression<Shape> e = get_expression();

   auto [graph, node] = e.add_to_graph(OpGraph{});

Then internally the ``add_to_graph`` method of the most derived class,
``Derived<T>``, is implemented something like:

.. code-block:: c++

   std::pair<OpGraph, Node> Derived<T>::add_to_graph(OpGraph g){
       // Assume Derived<T> inherits from Base<T>
       auto [subgraph, parent_node] = Base<T>::add_to_graph(g);

       // Create node corresponding to Derived<T> add to parent_node

       // Return new graph and new node
   }

This works because ``Expression<T>`` defines a virtual function
``std::pair<OpGraph, Node> add_to_graph(OpGraph g)`` which is overridden by each
of the derived classes. Each derived class calls the base class's
``add_to_graph`` method, which in turn returns the graph and the node just
added. Exactly what the nodes look like, and what information they contain is
punted to the OpGraph component (see :ref:`tw_designing_the_opgraph`).

.. _expression_user_api:

********
User API
********

The previous section showed how the expression objects actually interacted.
This section focuses on what the user actually writes.

Construction
============

If we view the main goal of the construction section to be the expressions
like ``ic = iapib;``, using the DSL the user would actually write this as:

.. code-block:: c++

   auto [a, b] = fill_in_a_and_b();
   T c, d, e, g; // No "f" to make connection to the example API section

   c("i,j,k") = a("i,j,k") + b("i,j,k");
   d("i,j,k") = a("i,j,k") * b("i,j,k");
   e("i,j,k") = a("i,j,k") - b("i,j,k");
   g("i,j,k") = a("i,j,k") / b("i,j,k");

This code is much more concise, but still contains all the contents shown
explicitly in :ref:`expression_construction`, but it now happens via a series
of unnamed temporary objects. For example the first line actually works by:

- ``a("i,j,k")`` creates an unnamed temporary ``Indexed<T>`` object,
- ``b("i,j,k")`` creates  another unnamed temporary ``Indexed<T>`` object,
- the ``Indexed<T>::operator+`` method is then called on the previous two
  temporary objects resulting in a third temporary of type ``Addition<T>``
- ``c("i,j,k")`` creates yet another temporary ``Indexed<T>`` object.
- Finally ``Indexed<T>::operator=`` is called assigning the ``Addition<T>``
  object to the the temporary resulting from ``c("i,j,k")``.

In order to satisfy :ref:`ec_reusable_intermediates`, we need to assign at least
one of the common intermediates to a named variable, *e.g.*:

.. code-block:: c++

   {
      auto aijk = a("i,j,k");
      c("i,j,k")  = aijk * b("i,j,k");
      d("i,j,k")  = aijk / b("i,j,k");
   }

In practice the ``Buffer`` objects actually assigned to ``c`` and ``d`` are
``FutureBuffer`` objects (see :ref:`tw_designing_the_buffer`). The
``FutureBuffer`` objects are tied to lifetime of the expression layer which
generated them. When all expression-layer objects involved in creating the
``FutureBuffer`` objects go out of scope evaluation begins. So if we want to
ensure that the above two equations are treated as a set of equations, and not
two individual equations, we need to make sure at least one of the expression-
layer objects is present in each equation (the ``{}`` are needed to establish a
scope for ``aijk``, ensuring it goes out scope after the second equation). While
it is theoretically possible for TensorWrapper to correctly identify the two
temporary objects resulting from ``b("i,j,k")`` to be identical, it is unlikely
that TensorWrapper will contain such optimizations in the near future. Hence
best practice will be to assign each common intermediate to a named variable,
*i.e.*, the above code block should really be written as:

.. code-block:: c++

   {
      auto aijk = a("i,j,k");
      auto bijk = b("i,j,k");
      c("i,j,k")  = aijk * bijk;
      d("i,j,k")  = aijk / bijk;
   }

so that the expression layer will identify ``b("i,j,k")`` as evaluating to the
same intermediate.

Non-Einstein Algebra
====================

.. code-block:: c++

   auto [L, Lt] = CholeskyDecomposition(A); // A = LLt
   auto [v, λ]  = EigenDecomposition(A, B); // Av = λBv

   auto a10_10 = A.slice(10, 10);



*******
Summary
*******

The above design satisfies the considerations raised in :ref:`ec_considerations`
by:

:ref:`ec_compose_multiple_objects`
   The entire expression layer is templated on the type of the object being
   composed. This allows the expression layer to be reused with various pieces
   of the ``TensorWrapper`` class (*e.g.*, the ``Shape`` class) in addition to
   the ``TensorWrapper`` class itself.

:ref:`ec_generalized_einstein_notation`
   The entry pont to the expression layer is, for most operations, is assigning
   indices to a tensor's modes. The resulting objects can then be composed using
   generalized Einstein notation.

:ref:`ec_non_einstein_math`


:ref:`ec_template_meta_programming_cost`
   Instead of templating the various pieces of the expression layer on the
   types of the sub-expressions, as is usually done, we only template the
   expression layer pieces on the types of the object being composed, *e.g.*,
   the template type parameter would be something like ``Shape`` or
   ``TensorWrapper`` instead of say a type like
   ``Addition<Indexed<Shape>, Indexed<Shape>>``.

:ref:`ec_reusable_intermediates`
   Each object in the expression layer is a node of a CST. Reusing the same
   object in multiple places reuses the same node of the CST.
