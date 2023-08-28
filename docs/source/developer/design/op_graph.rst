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

**********************
OpGraph Considerations
**********************

directed graph
   The graph needs to be directed in order to distinguish between inputs and
   outputs. There conceivably could be cycles if loops are involved.

tensors appear multiple times
   It is not uncommon for the same tensor to appear multiple times in a graph.
   For example many matrix decompositions are expressed in terms of a tensor
   and its transpose.

   - To avoid needing to have multiple edges between nodes we can allow
     different node objects to refer to the same tensor. The nodes therefore
     track where in the sequence we are, and the tensor refers to

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



***********
OpGraph API
***********

The API of the ``OpGraph`` component is modeled after the Boost Graph Library
(see `here <https://www.boost.org/doc/libs/1_83_0/libs/graph/doc/>`__). This
is to lower the barrier to entry in case the user is already familiar with that
library and so that an actual graph library (like Boost Graph Library) can be
wrapped by ``OpGraph`` if needed for performance.

.. code-block:: c++

   OpGraph g; // Default graph, no nodes
