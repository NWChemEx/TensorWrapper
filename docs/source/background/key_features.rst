#############################
Key Features of TensorWrapper
#############################

There are a lot of tensor libraries out there, the point of this page is to
keep track of what sets TensorWrapper a part.

***************
Design Features
***************

Wrapping
========

To the extent possible, TensorWrapper aims to graft the other features on this
list on to existing tensor libraries. Admittedly, this requires TensorWrapper
to "factor out" (in practice we have to reimplement since said features are
usually not modularized) some functionality.

Decoupled Expression Layer
==========================

Most C++ tensor libraries have an expression layer, which relies on
meta-template programming to build up a representation of what the user wants
before actually running the computation. From a computer language perspective,
the expression layer is a :ref:`term_dsl` which is represented as a
:ref:`term_cst`. To our knowledge, all existing C++ tensor libraries then
execute the CST (potentially after some optimization) in order to compute the
requested result. The problem with such an approach is that the optimizations
and implementations behind the individual operations become coupled to the
syntax of the tensor library's DSL (because CSTs are written in terms of the
DSL). TensorWrapper follows usual parsing procedures and converts the CST into
an :ref:`term_ast` before optimization and executing occur. In turn, the AST
serves as an intermediate representation which is not tied to the tensor
front end.

Logical vs Actual Layout
========================

Most distributed tensors have a concept of tiling because ultimately
operations on distributed data are done differently than on local data. In
practice tiling a tensor increases its rank, *e.g.*, if we tile a matrix along
the rows and columns it becomes a rank four tensor because we need two indices
to select the tile and two indices to select the element within the tile.
However, most tensor libraries still try to treat a tiled tensor as being the
same rank as an un-tiled tensor. In our opinion, this is somewhat awkward
because the tiling modes need to be treated differently than the original modes.
TensorWrapper's solution is to distinguish between the logical layout of the
tensor (*i.e.*, how the user declared it) and the actual layout of the tensor
(*i.e.*, including any additional modes added for performance reasons).

*************
Math Features
*************

Native Support for Nested Tensors
=================================

While we have a propensity to think of tensors as having scalar elements, this
need not be the case. For example, a matrix can also be thought of as a vector
of vectors. The key difference between these views is that in the second
providing a single offset is meaningful as it requests an element of the outer
vector. Most tensor libraries require the user to track alternative views
and map them to the more traditional view (in the matrix vs. vector of vector
example the user would have to request a particular slice of the tensor).
Native support for nesting allows slicing to be treated more naturally, it also
opens up the door for...

Native Support for Jagged Tensors
=================================

We can think of an :math:`n` by :math:`m` element matrix as an :math:`n`
element vector containing :math:`m` element vectors. In this scenario, each of
the inner vectors has the same shape. One can also imagine having a vector of
vectors where the inner vectors do not all have the same shape. If we view
such a vector of vectors as a matrix, the resulting matrix will have rows of
different lengths, *i.e.*, it is a "jagged" matrix. While jagged tensors may
seem exotic they occur quite naturally with sparse data with implicit zeros.
