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

   - For expressions which involve more than one term, this notation may be
     ambiguous (TODO: add an example when I can recall how it's ambiguous).
   - Can "put the summation signs back in" to remove ambiguity. 
   - Math operations include: addition, subtraction, scaling, multiplication,
     and division. The latter is missing from a number of existing tensor
     libraries.
   - Note that multiplication actually covers a number of operations including:
     element-wise product, matrix multiplication, contraction, trace, and direct
     product.

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

multiple sources/sinks
   The expression layers in most existing tensor libraries tend to be limited
   to a single (final) sink, *i.e.*, the entire expression must be assigned to
   a single tensor and it is this assignment which forces the evaluation.

   - Avoiding the evaluation can be done by storing the expression in the tensor
     instead of assigning the state to the tensor. This however, complicates
     forcing evaluation.
   - An alternative is a different tensor type, say ``TempTensor``, which acts
     like a ``TensorWrapper`` object, but doesn't force evaluation.

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

common intermediates
   As we build an expression certain sub-expressions may appear multiple times.
   Identifying these common sub-expressions is a precursory step to optimization
   and is thus out of scope for the expression layer.

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

Construction
============

Following from the :ref:`ec_generalized_einstein_notation` consideration we
expect that most users will enter into the expression layer by adding dummy
indices to an object. C++ wise this looks like:

.. code-block:: c++

   // Assume we have some T objects
   T a, b, c;

   Indexed<T> ia = a("i,j,k");
   Indexed<T> ib = b("i,j,k");
   Indexed<T> ic = c("i,j,k");

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