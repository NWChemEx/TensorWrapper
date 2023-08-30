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

.. _tw_designing_the_opgraph:

#####################
Designing the OpGraph
#####################

.. |m| replace:: :math:`m`
.. |n| replace:: :math:`n`
.. |mn| replace:: :math:`mn`
.. |A| replace:: :math:`\mathbf{A}`
.. |B| replace:: :math:`\mathbf{B}`
.. |C| replace:: :math:`\mathbf{C}`
.. |D| replace:: :math:`\mathbf{D}`

******************************
What is the OpGraph Component?
******************************

**************************
Why do we need an OpGraph?
**************************

Expression Layer vs. OpGraph
============================

Perhaps what is less clear is why do we need the expression layer AND the
OpGraph component? This question is brought on by the fact that there is a fair
amount of redundancy in the classes. Ultimately, the answer is a separation of
concerns. The expression layer is purely meant to capture what the user wants to
do. The OpGraph component is supposed to represent what TensorWrapper wants the
backend to do. Generally speaking what the user wants to do will be expressed in
terms of much coarser operations than what TensorWrapper wants the backend to
do.

On a related note, another key difference between the Expression layer and the
OpGraph component is the the Expression layer is designed so that it is easy for
the user to express their intent, whereas the OpGraph is designed to be easy
for a backend to traverse. Thus the Expression layer is equation-based whereas
the OpGraph component is graph-based.

***************
Tensor Networks
***************

The graph representation of tensor algebra we've stumbled onto is not something
new, mathematicians have known it for years. The graphical representation of a
tensor algebra is usually termed a tensor network. Here's the basics (see
:ref:`og_references` for background/where I took this from):

- Nodes of a graph are tensors, edges are contractions.
- The shape of the node can be used to convey properties of the tensor.

  - There seems to be several conventions in the wild. We pick the one which is
    easiest to draw.
  - Squares are the default symbol and indicates that no additional properties
    are known/specified. Each square should have the same area.
  - Tensors obtained by combining other tensors are denoted with rectangles. The
    area of the rectangle should reflect the number of tensors consumed to form
    it.

- The number of edges going into a node equals its rank.

  - The thickness with which an edge is drawn is proportional to the number of
    modes comprising it. For example, reshaping an |m| by |n| matrix to a
    |mn|-element vector would result in an edge which is twice as thick as an
    edge denoting a single node.
  - Contraction over multiple indices is denoted with parallel edges.
  - The trace of a tensor (or product of tensors) is denoted with a loop.

- Additional operations, such as tensor products, addition, element-wise
  addition, etc. are usually specified with a single rectangular node labeled
  with the operation.

  - The area rules specified above apply to the resulting node.

Overall tensor networks seem to really be aimed at providing a graphical
representation of the contractions in a single term. The hierarchical nodes
(rectangular nodes replacing multiple square nodes) allow for incorporating
other operations into the graph. Ultimately, tensor networks are heavily focused
on expressing complicated contraction patterns, other operations are somewhat
awkward to express, particularly when they can not be expressed as binary
operations (in theory one could use hypergraphs to accommodate ternary
operations such as :math:`A_{ij}=C_{ik}B_{k}C_{kj}`).

**********************
OpGraph Considerations
**********************

data flow
   The graph needs to be capable of distinguishing between inputs and
   outputs. There conceivably could be cycles if loops are involved.

   - In terms of a tensor network this is represented by nesting of the nodes.
     The contents of a nested node convey where

tensors appear multiple times
   It is not uncommon for the same tensor to appear multiple times in a graph.
   For example many matrix decompositions are expressed in terms of a tensor
   and its transpose.

multiple source nodes
   The graph may have multiple inputs. Each input is assumed to be a
   ``TensorWrapper`` object.

Multiple sink nodes
   The graph may have multiple results. We assume the results have different
   types based on whether or not they are terminating the graph. The graph
   terminates on a ``TensorWrapper`` object.

References to actual data
   The ``OpGraph`` must either have handles to the actual data, or somehow be
   associated with the actual data in order for backends to translate the AST
   into results.

expression component compatibility
   The ``OpGraph`` object will ultimately be filled in by the objects in the
   expression component. Thus the interface of the ``OpGraph`` class needs to
   be designed so that it is compatible with the ``Expression`` component's
   implementation (see :ref:`designing_the_expression_component`).

****************
OpGraph Notation
****************

The graph represented by an ``OpGraph`` object is inspired by tensor network
notation, but has some differences to accommodate the extended set of use cases
the OpGraph component must deal with. The need for being able to visualize an
 ``OpGraph`` object graphically is useful for design, and is expected to also
 be useful for code analysis/optimization. To that end we propose the following
 notation:

- Nodes of the graph depict either tensors or operators.

  - Tensors are denoted with square nodes.
  - Operations with circles.
  - Nodes will be labeled with the name of the tensor or the operation.

- Edges denote modes.

  - Parallel edges are avoided by fusing indices, *i.e.*, each edge is labeled
    with all indices participating in that operation.
  - The number of fused modes is tracked by annotating the modes.
  - Annotations must be consistent throughout a graph, *i.e.*, if the the same
    annotation appears multiple times TensorWrapper will assume that the
    corresponding modes are spanned by the same basis set.

- Edges are directed.

  - The direction indicates data flow. Sources are input tensors. Sinks are
    outputs.
  - The rank of a tensor can be determined from the number of unique
    indices associated with it.

OpGraph Structure
=================

.. _fig_narity:

.. figure:: assets/narity.png
   :align: center

   Overview of how operations of different :ref:`term_arity` look using OpGraph
   graphical notation. For simplicity, mode annotations and operation labels are
   not specified.

:ref:`fig_narity` illustrates how an ``OpGraph`` representation looks for
operations of various :ref:`term_arity`. Graphs are grouped into a matrix such
that for row |m| (|m| is 1-based) the |n|-th column (|n| is 0-based) denotes an
operation returning |m| tensors given |n| tensors (|m| and |n| have different
bases because operations returning no tensors are not interesting).

The simplest, non-null, ``OpGraph`` stems from simply declaring a tensor. The
resulting "nullary" graph for a tensor |A|, is shown in :ref:`fig_narity`. From
the perspective of the OpGraph component, the actual declaration of a tensor
requires performing some opaque operation (such operations are denoted by purple
circles in :ref:`fig_narity`). For declaring a tensor this operation simply
returns the tensor (and does not require any input to do so, hence it is a
nullary operation). From the perspective of ``OpGraph``, the nullary operations
which create the source tensors must always be present and they are not usually
interesting (effectively being lambdas like `[](){return A;}`). Thus by
convention, and in an effort to simplify the representation of ``OpGraph``
objects, the nullary operations giving rise to the source tensors will usually
be implicit. The exception being when those nullary operations are interesting
(usually because they are on-demand generator functions). For the remaining
columns in :ref:`fig_narity` this convention applies. As shown in row 2 of
:ref:`fig_narity`

The next simplest ``OpGraph`` requires mapping an input tensor to an output
tensor via some intermediate operation. Such operations are "unary" and examples
include permuting the modes of a tensor, scaling a tensor, and raising a tensor
to a power. It is also possible that a unary operation returns multiple tensors,
*e.g.*, a standard eigen solver which returns the eigenvectors and the
eigenvalues. At this point, the basic structure of an operation should be clear,
nonetheless :ref:`fig_narity` shows examples of some other arities.

Basic Operations
================

.. _fig_basic_operations:

.. figure:: assets/opgraph_basic_ops.png
   :align: center

   Pictorial representations of the fundamental operations of the OpGraph
   component.

:ref:`fig_basic_operations` shows some of the basic operations which will be
comprise actual ``OpGraph`` instances. For simplicity we have focused on matrix
operations (most input/output edges have two annotations), but much of what is
in :ref:`fig_basic_operations` generalizes to other rank tensors in a
straightforward manner. Ignoring nullary operations, all operation nodes have
one or more inputs and one or more outputs. The goal is to establish a small set
of "fundamental" operations and to write all other operations in terms of these
operations. For example, we do not define a chip operation, but a chip operation
can be defined by a slice followed by a reshape.

As shown in :ref:`fig_basic_operations`, tensors acting as inputs to an
operation have their annotations associated with the edge connecting them to
the operation. Tensors resulting from an operation have their annotations
associated with the edge connecting the the operation to the tensor. In turn
permutations are signified by reordering the output indices relative to the
input indices.

The most questionable choice we have made is the "multiplication" operator. The
multiplication operator actually stands in for a number of operations including
trace, contraction, tensor product, and element-wise multiplication (though we
have also defined an element-wise multiplication operator for consistency with
the other element-wise operations). Our motivation here is that many of the
backend tensor libraries have already invested in infrastructure for handling
generalized Einstein summation convention (and/or tensor networks) and in the
first pass we intend to dispatch to the backend's implementations.

More Complicated OpGraphs
=========================

Ultimately, the state of an ``OpGraph`` is obtained by combining basic
operations from the previous subsection into larger graphs. From the
:ref:`designing_the_expression_component` section we had for example the set
of equations:

.. code-block:: c++

   {
      auto aijk = a("i,j,k");
      c("i,j,k")  = aijk * b("i,j,k");
      d("i,j,k")  = aijk / b("i,j,k");
   }



***********
OpGraph API
***********

The API of the ``OpGraph`` component is modeled after the Boost Graph Library
(see `here <https://www.boost.org/doc/libs/1_83_0/libs/graph/doc/>`__). This
is to lower the barrier to entry in case the user is already familiar with that
library and so that an actual graph library (like Boost Graph Library) can be
wrapped by ``OpGraph`` if needed for performance.

The ``OpGraph`` class serves the role of an overall container for the graph. A
similar role to say ``boost::adjacency_matrix`` or ``boost::adjacency_list``
classes.

.. code-block:: c++

   OpGraph g; // Default graph, no nodes

.. _og_references:

**********
References
**********

For the tensor network background we primarily relied on sources found in the
README of Google's
`TensorNetwork https://github.com/google/TensorNetwork#readme`__ project,
specifically:

- `https://iopscience.iop.org/article/10.1088/1751-8121/aa6dc3/pdf`
- `https://arxiv.org/pdf/1306.2164.pdf`
